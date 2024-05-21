# Copyright (c) Meta Platforms, Inc. and affiliates.

import numpy as np
import torch
from torch.nn.functional import normalize

from maploc.models.bev_mapper import BEVMapper, build_query_grid
from maploc.models.ransac_matcher import (
    grid_refinement_batched,
    pose_scoring_many_batched,
    sample_transforms_ransac_batched,
)
from maploc.utils import grids
from maploc.utils.neural_cutout import neural_cutout
from maploc.utils.wrappers import Transform2D

from . import get_model
from .base import BaseModel
from .map_encoder import MapEncoder
from .metrics import (
    AngleError,
    AngleRecall,
    ExhaustiveEntropy,
    Location2DError,
    Location2DRecall,
)
from .voting import (
    argmax_xyr,
    conv2d_fft_batchwise,
    expectation_xyr,
    log_softmax_spatial,
    mask_yaw_prior,
    nll_loss_xyr,
    nll_loss_xyr_smoothed,
)


class OrienterNet(BaseModel):
    default_conf = {
        "image_encoder": "???",
        # "semantic_encoder": "???",
        "map_encoder": None,
        "aerial_encoder": None,
        "bev_mapper": None,
        # "bev_net": "???",
        "latent_dim": "???",
        "matching_dim": "???",
        "pixel_per_meter": "???",
        "num_rotations": "???",
        "add_temperature": False,
        "normalize_features": False,
        "padding_matching": "replicate",
        "apply_map_prior": False,
        "do_label_smoothing": False,
        "sigma_xy": 1,
        "sigma_r": 2,
        "use_map_cutout": False,
        "ransac_matcher": False,
        "clip_negative_scores": False,
        "num_pose_samples": 10_000,
        "num_pose_sampling_retries": 8,
        "ransac_grid_refinement": False,
        # deprecated
        "depth_parameterization": "scale",
        "norm_depth_scores": False,
        "normalize_scores_by_dim": False,
        "normalize_scores_by_num_valid": True,
        "prior_renorm": True,
        "retrieval_dim": None,
    }

    def _init(self, conf):
        assert not self.conf.norm_depth_scores
        assert self.conf.depth_parameterization == "scale"
        assert not self.conf.normalize_scores_by_dim
        assert self.conf.normalize_scores_by_num_valid
        assert self.conf.prior_renorm

        Encoder = get_model(conf.image_encoder.get("name", "feature_extractor_v2"))
        self.map_encoder = self.aerial_encoder = None
        if conf.map_encoder is not None:
            self.map_encoder = MapEncoder(conf.map_encoder)  # OSM maps
        if conf.aerial_encoder is not None:
            self.aerial_encoder = Encoder(conf.aerial_encoder.backbone)
        if conf.map_encoder is None and conf.aerial_encoder is None:
            raise ValueError("At least one map encoder must be created")

        self.bev_mapper = BEVMapper(conf.bev_mapper)

        if conf.add_temperature:
            temperature = torch.nn.Parameter(torch.tensor(0.0))
            self.register_parameter("temperature", temperature)

    def exhaustive_voting(self, f_bev, f_map, valid_bev, confidence_bev=None):
        if self.conf.normalize_features or self.conf.use_map_cutout:
            f_bev = normalize(f_bev, dim=1)
            f_map = normalize(f_map, dim=1)

        # Build the templates and exhaustively match against the map.
        if confidence_bev is not None:
            f_bev = f_bev * confidence_bev.unsqueeze(1)
        f_bev = f_bev.masked_fill(~valid_bev.unsqueeze(1), 0.0)
        templates = self.bev_mapper.template_sampler(f_bev)
        with torch.autocast("cuda", enabled=False):
            scores = conv2d_fft_batchwise(
                f_map.float(),
                templates.float(),
                padding_mode=self.conf.padding_matching,
            )
        if self.conf.add_temperature:
            scores = scores * torch.exp(self.temperature)

        # Reweight the different rotations based on the number of valid pixels in each
        # template. Axis-aligned rotation have the maximum number of valid pixels.
        valid_templates = self.bev_mapper.template_sampler(valid_bev.float()[None]) > (
            1 - 1e-4
        )
        num_valid = valid_templates.float().sum((-3, -2, -1))
        scores = scores / num_valid[..., None, None]
        return scores

    def compute_similarity(self, f_bev, f_map, valid_bev, confidence_bev=None):

        batch_size = len(f_bev)
        if self.conf.normalize_features or self.conf.use_map_cutout:
            f_bev = normalize(f_bev, dim=1)
            f_map = normalize(f_map, dim=1)
        f_bev_points = f_bev.movedim(-3, -1).reshape(batch_size, -1, f_bev.shape[-3])
        f_map_points = f_map.movedim(-3, -1)  # channel to last dim
        sim_points = torch.einsum("...nd,...ijd->...nij", f_bev_points, f_map_points)

        if self.conf.clip_negative_scores:
            sim_points = torch.nn.ReLU()(sim_points)

        if self.conf.add_temperature:
            sim_points *= torch.exp(self.temperature)

        sim_points = sim_points * valid_bev.reshape(batch_size, -1)[..., None, None]
        prob_points = torch.nn.Softmax(dim=-1)(
            sim_points.view(*sim_points.shape[:-2], -1)
        ).view(*sim_points.shape)
        prob_points = prob_points * valid_bev.reshape(batch_size, -1)[..., None, None]

        num_valid = (
            valid_bev.reshape(batch_size, -1).sum(-1).clamp(min=1)[:, None, None, None]
        )
        sim_points /= num_valid
        prob_points /= num_valid

        return sim_points, prob_points

    def ransac_voting(self, sim_points, prob_points, valid_bev, map_mask, map_T_cam_gt):
        """Sample correspondence pairs, compute poses, and score poses"""

        pose_scores = []
        map_T_cam_samples = []
        bev_ij_pool = []
        map_ij_pool = []

        # Temp. work-around for scoring large num of samples with small GPU
        num_iter = self.conf.num_pose_samples // 5_000

        for i in range(num_iter):
            map_T_cam_samples_sub, bev_ij_pool_sub, map_ij_pool_sub = (
                sample_transforms_ransac_batched(
                    self.bev_mapper.bev_ij_pts,
                    prob_points.detach(),
                    5_000,
                    self.conf.num_pose_sampling_retries,
                )
            )

            if i == 0:
                map_T_cam_samples_sub = torch.vmap(lambda *x: torch.cat(x, 0))(
                    map_T_cam_gt._data.unsqueeze(1), map_T_cam_samples_sub
                )

            pose_scores_sub = pose_scoring_many_batched(
                map_T_cam_samples_sub,  # B,num_poses,3
                sim_points,
                self.bev_mapper.bev_ij_pts,  # I, J, 2
                valid_bev,
                map_mask,
            )
            if i == 0:
                # Extract gt pose score and remove from poses
                gt_pose_score = pose_scores_sub[..., 0]
                pose_scores_sub = pose_scores_sub[..., 1:]
                map_T_cam_samples_sub = map_T_cam_samples_sub[..., 1:, :]

            pose_scores.append(pose_scores_sub)
            map_T_cam_samples.append(map_T_cam_samples_sub)
            bev_ij_pool.append(bev_ij_pool_sub)
            map_ij_pool.append(map_ij_pool_sub)

        batch_size = len(sim_points)
        pose_scores = torch.stack(pose_scores, -1).view(batch_size, -1)
        map_T_cam_samples = torch.stack(map_T_cam_samples, -2).view(batch_size, -1, 3)
        bev_ij_pool = torch.stack(bev_ij_pool, -3).view(batch_size, -1, 2, 2)
        map_ij_pool = torch.stack(map_ij_pool, -3).view(batch_size, -1, 2, 2)

        # Invalidate poses that fall outside the map bounds
        map_t_cam_samples = map_T_cam_samples[..., 1:]
        size = torch.tensor(sim_points.shape[-2:]).to(map_t_cam_samples)  # 256, 256
        valid = torch.all((map_t_cam_samples >= 0) & (map_t_cam_samples < size), -1)
        pose_scores = pose_scores * valid

        return map_T_cam_samples, pose_scores, bev_ij_pool, map_ij_pool, gt_pose_score

    def fuse_neural_maps(self, feature_maps):
        """Fuse aerial and semantic features maps with dropout"""
        # max pool feature maps
        # Apply Dropout. # todo: test this
        # ==
        # dropout_mask = torch.bernoulli(
        #     torch.full(len(planes), len(planes[0]), 0.5, device=planes[0].device)
        # )
        # dropout_mask = torch.where(
        #     dropout_mask.any(dim=0, keepdim=True),
        #     dropout_mask,
        #     torch.ones_like(dropout_mask)
        # )
        # features = [
        #     p.replace(
        #         valid=torch.where(m.unsqueeze(-1).unsqueeze(-1)), p.valid, torch.zeros_like(p.valid)
        #     )
        #     for p, m in zip(feature_maps, dropout_mask)
        # ]
        # features = torch.stack(features, dim=-2)
        # ==
        feature_maps = torch.stack(feature_maps, dim=0)
        f_map, _ = torch.max(feature_maps, dim=0)
        return f_map

    def _forward(self, data):

        pred = {}

        # Encode aerial/semantic maps
        # note: these maps are in memory layout
        feature_maps = []
        if self.map_encoder is not None:
            assert "semantic_map" in data
            pred["semantic_map"] = self.map_encoder(
                {**data, "map": data["semantic_map"]}
            )
            feature_maps.append(pred["semantic_map"]["map_features"][0])
        if self.aerial_encoder is not None:
            assert "aerial_map" in data, "Aerial map not found in data"
            pred["aerial_map"] = self.aerial_encoder({"image": data["aerial_map"]})[
                "feature_maps"
            ][0]
            feature_maps.append(pred["aerial_map"])

        # Fuse neural maps if multiple
        if len(feature_maps) == 1:
            f_map = feature_maps[0]
        elif len(feature_maps) > 1:
            f_map = self.fuse_neural_maps(feature_maps)
        else:
            raise ValueError(f"At least one feature map must be created")

        # this is an old grid version used only in ransac matching. will be removed
        if self.bev_mapper.bev_ij_pts is None:
            self.bev_mapper.bev_ij_pts = build_query_grid().to(f_map)

        # Predict BEV from image
        bev_mapper_pred = self.bev_mapper(data)
        pred.update({**bev_mapper_pred})

        f_bev, valid_bev, confidence_bev = [
            pred["bev"][key] for key in ["output", "valid_bev", "confidence"]
        ]

        all_valid_mask = torch.ones((f_map[:, 0, ...].shape)).to(valid_bev)
        map_mask = data.get("map_mask", all_valid_mask)

        if self.conf.use_map_cutout:  # for evaluating matchers
            f_bev, valid_cutout = neural_cutout(
                self.bev_mapper.bev_ij_pts, f_map, data["map_T_cam"]
            )
            valid_bev = valid_bev & valid_cutout
            if confidence_bev is not None:
                confidence_bev = pred["bev"]["confidence"] = (
                    torch.ones_like(confidence_bev) * valid_bev
                )  # / valid_bev.sum((-1, -2))
            pred["bev"]["output"] = f_bev

        log_prior = pred["semantic_map"]["log_prior"][0]
        if "semantic_map" in pred and "log_prior" in pred["semantic_map"]:
            log_prior = pred["semantic_map"]["log_prior"][0]

        if not self.conf.ransac_matcher:  # OrienterNet's Exhaustive Matching

            # Temporarily revert bev format. TODO: refactor template sampler.
            f_bev = pred["bev"]["output"] = torch.rot90(f_bev, 1, dims=(-2, -1))
            if confidence_bev is not None:
                confidence_bev = pred["bev"]["confidence"] = torch.rot90(
                    confidence_bev, 1, dims=(-2, -1)
                )
            valid_bev = torch.rot90(valid_bev, 1, dims=(-2, -1))

            scores = self.exhaustive_voting(f_bev, f_map, valid_bev, confidence_bev)
            scores = scores.moveaxis(1, -1)  # B,H,W,N

            if (
                "semantic_map" in pred
                and "log_prior" in pred["semantic_map"]
                and self.conf.apply_map_prior
            ):
                scores = scores + log_prior.unsqueeze(-1)
            # pred["scores_unmasked"] = scores.clone()
            scores.masked_fill_(~map_mask[..., None], -np.inf)
            if "yaw_prior" in data:  # TODO: refactor
                mask_yaw_prior(scores, data["yaw_prior"], self.conf.num_rotations)
            log_probs = log_softmax_spatial(scores)  # already rotated
            with torch.no_grad():
                uvr_max = argmax_xyr(scores).to(scores)
                uvr_avg, _ = expectation_xyr(log_probs.exp())

            # Convert rotated uv to ij
            ij_max = torch.flip(uvr_max[..., :2], dims=[-1])
            ij_avg = torch.flip(uvr_avg[..., :2], dims=[-1])
            yaw_max = 180 - uvr_max[..., 2][..., None]
            yaw_avg = 180 - uvr_avg[..., 2][..., None]
            map_T_cam_max = Transform2D.from_degrees(yaw_max, ij_max)
            map_T_cam_avg = Transform2D.from_degrees(yaw_avg, ij_avg)
            resolution = 1 / data["pixels_per_meter"][..., None]
            tile_T_cam_max = Transform2D.from_pixels(map_T_cam_max, resolution)
            tile_T_cam_avg = Transform2D.from_pixels(map_T_cam_avg, resolution)

            # Revert mem layout to snap's. TODO: Remove when template sampler is fixed
            f_bev = pred["bev"]["output"] = torch.rot90(f_bev, -1, dims=(-2, -1))
            if confidence_bev is not None:
                confidence_bev = pred["bev"]["confidence"] = torch.rot90(
                    confidence_bev, -1, dims=(-2, -1)
                )
            valid_bev = torch.rot90(valid_bev, -1, dims=(-2, -1))

            pred["scores"] = scores
            pred["log_probs"] = log_probs

        else:  # SNAP's RANSAC matcher

            # Compute similarity between each bev-map point pair
            sim_points, prob_points = self.compute_similarity(
                f_bev, f_map, valid_bev, confidence_bev
            )

            # Sample correspondences, compute poses and score them
            map_T_cam_samples, pose_scores, _, _, gt_pose_score = self.ransac_voting(
                sim_points, prob_points, valid_bev, map_mask, data["map_T_cam"]
            )

            # Extract pose with best score
            _, max_indices = torch.max(pose_scores, dim=-1)
            select_fn = torch.vmap(lambda x, i: x[i[None]][0])
            map_T_cam_max = select_fn(map_T_cam_samples, max_indices)

            # Refine the pose within a small grid around best pose
            if self.conf.ransac_grid_refinement:
                pred["map_T_cam_ransac"] = map_T_cam_max
                map_T_cam_max, pred["refined_score"], pred["scores_grid_refine"] = (
                    grid_refinement_batched(
                        map_T_cam_max,
                        sim_points,
                        self.bev_mapper.bev_ij_pts,
                        valid_bev,
                        map_mask,
                    )
                )

            map_T_cam_max = Transform2D(map_T_cam_max)
            resolution = 1 / data["pixels_per_meter"][..., None]
            tile_T_cam_max = Transform2D.from_pixels(map_T_cam_max, resolution)

            # Dummy
            tile_T_cam_avg = tile_T_cam_max
            map_T_cam_avg = map_T_cam_max
            pred["log_probs"] = torch.zeros((1, 256, 256, 360))

            pred.update(
                {
                    # "bev_ij_pool": bev_ij_pool,
                    # "map_ij_pool": map_ij_pool,
                    "map_T_cam_samples": map_T_cam_samples,
                    "pose_scores": pose_scores,
                    "max_indices": max_indices,
                    "prob_points": prob_points,
                    "gt_pose_score": gt_pose_score,
                }
            )

        return {
            **pred,
            "tile_T_cam_max": tile_T_cam_max,
            "tile_T_cam_expectation": tile_T_cam_avg,
            "map_T_cam_max": map_T_cam_max,
            "map_T_cam_expectation": map_T_cam_avg,
            "features_map": f_map,
            "features_bev": f_bev,
            "valid_bev": valid_bev.squeeze(1),
        }

    def loss(self, pred, data):

        # Revert refactored outputs to original. TODO: update sample_xyr
        ij_gt = data["map_T_cam"].t
        uv_gt = ij_gt.clone()
        uv_gt = torch.flip(ij_gt, dims=[-1])
        yaw_gt = (180 - data["map_T_cam"].angle.squeeze(-1)) % 360
        log_probs = pred["log_probs"]

        map_mask = data.get("map_mask")
        if map_mask is not None:
            map_mask = torch.rot90(map_mask, 1, dims=(-2, -1))

        if self.conf.do_label_smoothing:
            nll = nll_loss_xyr_smoothed(
                log_probs,
                uv_gt,
                yaw_gt,
                self.conf.sigma_xy / self.conf.pixel_per_meter,
                self.conf.sigma_r,
                mask=data.get("map_mask"),
            )
        else:
            nll = nll_loss_xyr(log_probs, uv_gt, yaw_gt)
        loss = {"total": nll, "nll": nll}
        if self.training and self.conf.add_temperature:
            loss["temperature"] = self.temperature.expand(len(nll))
        return loss

    def metrics(self):
        metrics = {}
        if not self.conf.ransac_matcher:
            metrics["exhaustive_entropy"] = ExhaustiveEntropy()

        return {
            **metrics,
            "xy_max_error": Location2DError("tile_T_cam_max"),
            "xy_expectation_error": Location2DError("tile_T_cam_expectation"),
            "yaw_max_error": AngleError("tile_T_cam_max"),
            "xy_recall_2m": Location2DRecall(2.0, key="tile_T_cam_max"),
            "xy_recall_5m": Location2DRecall(5.0, key="tile_T_cam_max"),
            "yaw_recall_2°": AngleRecall(2.0, "tile_T_cam_max"),
            "yaw_recall_5°": AngleRecall(5.0, "tile_T_cam_max"),
        }
