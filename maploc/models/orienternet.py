# Copyright (c) Meta Platforms, Inc. and affiliates.

import numpy as np
import torch
from torch.nn.functional import normalize
from torch.profiler import ProfilerActivity, profile, record_function

from maploc.models.ransac_matcher import (
    grid_refinement_batched,
    pose_scoring_many_batched,
    sample_transforms_ransac_batched,
)
from maploc.utils.neural_cutout import neural_cutout
from maploc.utils.wrappers import Transform2D

from . import get_model
from .base import BaseModel
from .bev_net import BEVNet
from .bev_projection import CartesianProjection, PolarProjectionDepth
from .map_encoder import MapEncoder
from .metrics import (
    AngleError,
    AngleRecall,
    Location2DError,
    Location2DRecall,
    SamplesRecall,
)
from .voting import (
    TemplateSampler,
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
        "map_encoder": "???",
        "bev_net": "???",
        "latent_dim": "???",
        "matching_dim": "???",
        "scale_range": [0, 9],
        "num_scale_bins": "???",
        "z_min": None,
        "z_max": "???",
        "x_max": "???",
        "pixel_per_meter": "???",
        "num_rotations": "???",
        "add_temperature": False,
        "normalize_features": False,
        "padding_matching": "replicate",
        "apply_map_prior": False,
        "do_label_smoothing": False,
        "sigma_xy": 1,
        "sigma_r": 2,
        "use_map_cutout": True,
        "ransac_matcher": True,
        "clip_negative_scores": True,
        "num_pose_samples": 10_000,
        "num_pose_sampling_retries": 8,
        "ransac_grid_refinement": True,
        "profiler_mode": False,
        "overall_profiler": False,
        "compute_sim_relu_and_numvalid": False,
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
        self.image_encoder = Encoder(conf.image_encoder.backbone)
        self.map_encoder = MapEncoder(conf.map_encoder)
        self.bev_net = None if conf.bev_net is None else BEVNet(conf.bev_net)

        ppm = conf.pixel_per_meter
        self.projection_polar = PolarProjectionDepth(
            conf.z_max,
            ppm,
            conf.scale_range,
            conf.z_min,
        )
        self.projection_bev = CartesianProjection(
            conf.z_max, conf.x_max, ppm, conf.z_min
        )
        self.template_sampler = TemplateSampler(
            self.projection_bev.grid_xz, ppm, conf.num_rotations
        )

        self.scale_classifier = torch.nn.Linear(conf.latent_dim, conf.num_scale_bins)
        if conf.bev_net is None:
            self.feature_projection = torch.nn.Linear(
                conf.latent_dim, conf.matching_dim
            )
        if conf.add_temperature:
            temperature = torch.nn.Parameter(torch.tensor(0.0))
            self.register_parameter("temperature", temperature)

        self.bev_ij_pts = None

    def build_query_grid(self, f_bev):
        """Computes coordinates for each pixel in f_bev"""

        h, w = f_bev.shape[-2:]  # h = 129, w = 64
        bev_t_cam = torch.tensor([(h - 1) / 2, 0.0])

        bev_ij_pts = (
            torch.stack(torch.unravel_index(torch.arange(h * w), (h, w)), -1)
            - bev_t_cam
        )

        # BEV faces east in the map frame by default, so we rotate the coords by 90deg
        bev_ij_pts = (
            Transform2D.from_degrees(torch.tensor([-90]), torch.zeros(2)) @ bev_ij_pts
        )

        return bev_ij_pts.view(h, w, 2)

    def exhaustive_voting(self, f_bev, f_map, valid_bev, confidence_bev=None):
        if self.conf.normalize_features or self.conf.use_map_cutout:
            f_bev = normalize(f_bev, dim=1)
            f_map = normalize(f_map, dim=1)

        # Build the templates and exhaustively match against the map.
        if confidence_bev is not None:
            f_bev = f_bev * confidence_bev.unsqueeze(1)
        f_bev = f_bev.masked_fill(~valid_bev.unsqueeze(1), 0.0)

        if self.conf.profiler_mode:
            torch.cuda.reset_peak_memory_stats()
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                profile_memory=True,
            ) as prof:
                with record_function("exhaustive_template_sampler"):
                    templates = self.template_sampler(f_bev)
            cuda_time = prof.key_averages()[0].cuda_time / 1000
            stats = torch.cuda.memory_stats()
            peak_bytes = stats["allocated_bytes.all.peak"] / 1024**3
            print(
                f"[exhaustive_template_sampler]: {cuda_time:.3f} ms, {peak_bytes:.2f} GB"
            )
        else:
            templates = self.template_sampler(f_bev)

        with torch.autocast("cuda", enabled=False):
            if self.conf.profiler_mode:
                torch.cuda.reset_peak_memory_stats()
                with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    profile_memory=True,
                ) as prof:
                    with record_function("exhaustive_conv_fft"):
                        scores = conv2d_fft_batchwise(
                            f_map.float(),
                            templates.float(),
                            padding_mode=self.conf.padding_matching,
                        )
                cuda_time = prof.key_averages()[0].cuda_time / 1000
                stats = torch.cuda.memory_stats()
                peak_bytes = stats["allocated_bytes.all.peak"] / 1024**3
                print(f"[exhaustive_conv_fft]: {cuda_time:.3f} ms, {peak_bytes:.2f} GB")
            else:
                scores = conv2d_fft_batchwise(
                    f_map.float(),
                    templates.float(),
                    padding_mode=self.conf.padding_matching,
                )

        if self.conf.add_temperature:
            scores = scores * torch.exp(self.temperature)

        # Reweight the different rotations based on the number of valid pixels in each
        # template. Axis-aligned rotation have the maximum number of valid pixels.
        valid_templates = self.template_sampler(valid_bev.float()[None]) > (1 - 1e-4)
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

        if self.conf.profiler_mode:
            torch.cuda.reset_peak_memory_stats()
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                profile_memory=True,
            ) as prof:
                with record_function("compute_similarity_einsum"):
                    sim_points = torch.einsum(
                        "...nd,...ijd->...nij", f_bev_points, f_map_points
                    )
            cuda_time = prof.key_averages()[0].cuda_time / 1000
            stats = torch.cuda.memory_stats()
            peak_bytes = stats["allocated_bytes.all.peak"] / 1024**3
            print(
                f"[compute_similarity_einsum]: {cuda_time:.3f} ms, {peak_bytes:.2f} GB"
            )
        else:
            sim_points = torch.einsum(
                "...nd,...ijd->...nij", f_bev_points, f_map_points
            )

        # if self.conf.clip_negative_scores:
        if self.conf.compute_sim_relu_and_numvalid:
            sim_points = torch.nn.functional.relu(sim_points)
        if self.conf.add_temperature:
            sim_points *= torch.exp(self.temperature)

        if self.conf.profiler_mode:
            torch.cuda.reset_peak_memory_stats()
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                profile_memory=True,
            ) as prof:
                with record_function("compute_similarity_softmax"):
                    prob_points = torch.nn.functional.softmax(
                        sim_points.view(*sim_points.shape[:-2], -1), dim=-1
                    ).view(*sim_points.shape)
            cuda_time = prof.key_averages()[0].cuda_time / 1000
            stats = torch.cuda.memory_stats()
            peak_bytes = stats["allocated_bytes.all.peak"] / 1024**3
            print(
                f"[compute_similarity_softmax]: {cuda_time:.3f} ms, {peak_bytes:.2f} GB"
            )
        else:
            prob_points = torch.nn.functional.softmax(
                sim_points.view(*sim_points.shape[:-2], -1), dim=-1
            ).view(*sim_points.shape)

        prob_points = prob_points * valid_bev.reshape(batch_size, -1)[..., None, None]

        if self.conf.compute_sim_relu_and_numvalid:
            num_valid = (
                valid_bev.reshape(batch_size, -1)
                .sum(-1)
                .clamp(min=1)[:, None, None, None]
            )
            sim_points /= num_valid
            prob_points /= num_valid  # Probably not needed as cumsum searchsorted works without weights adding to 1

        return sim_points, prob_points

    def ransac_voting(self, sim_points, prob_points, valid_bev, map_mask, map_T_cam_gt):
        """Sample correspondence pairs, compute poses, and score poses"""

        # pose_scores = []
        # map_T_cam_samples = []
        # bev_ij_pool = []
        # map_ij_pool = []

        # Temp. work-around for scoring large num of samples with small GPU
        # num_iter = self.conf.num_pose_samples // 5_000

        # for i in range(num_iter):

        map_T_cam_samples, bev_ij_pool, map_ij_pool = sample_transforms_ransac_batched(
            self.bev_ij_pts,
            prob_points.detach(),
            self.conf.num_pose_samples,  # 5_000,
            self.conf.num_pose_sampling_retries,
            self.conf.profiler_mode,
        )

        # if i == 0:
        map_T_cam_samples = torch.vmap(lambda *x: torch.cat(x, 0))(
            map_T_cam_gt._data.unsqueeze(1), map_T_cam_samples
        )

        if self.conf.profiler_mode:
            torch.cuda.reset_peak_memory_stats()
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                profile_memory=True,
            ) as prof:
                with record_function("pose_scoring"):
                    pose_scores = pose_scoring_many_batched(
                        map_T_cam_samples,  # B,num_poses,3
                        sim_points,
                        self.bev_ij_pts,  # I, J, 2
                        valid_bev,
                        map_mask,
                    )
            cuda_time = prof.key_averages()[0].cuda_time / 1000
            stats = torch.cuda.memory_stats()
            peak_bytes = stats["allocated_bytes.all.peak"] / 1024**3
            print(f"[pose_scoring]: {cuda_time:.3f} ms, {peak_bytes:.2f} GB")
        else:
            pose_scores = pose_scoring_many_batched(
                map_T_cam_samples,  # B,num_poses,3
                sim_points,
                self.bev_ij_pts,  # I, J, 2
                valid_bev,
                map_mask,
            )

        # if i == 0:
        # Extract gt pose score and remove from poses
        gt_pose_score = pose_scores[..., 0]
        pose_scores = pose_scores[..., 1:]
        map_T_cam_samples = map_T_cam_samples[..., 1:, :]

        # pose_scores.append(pose_scores_sub)
        # map_T_cam_samples.append(map_T_cam_samples_sub)
        # bev_ij_pool.append(bev_ij_pool_sub)
        # map_ij_pool.append(map_ij_pool_sub)

        # batch_size = len(sim_points)
        # pose_scores = torch.stack(pose_scores, -1).view(batch_size, -1)
        # map_T_cam_samples = torch.stack(map_T_cam_samples, -2).view(batch_size, -1, 3)
        # bev_ij_pool = torch.stack(bev_ij_pool, -3).view(batch_size, -1, 2, 2)
        # map_ij_pool = torch.stack(map_ij_pool, -3).view(batch_size, -1, 2, 2)

        # Invalidate poses that fall outside the map bounds
        map_t_cam_samples = map_T_cam_samples[..., 1:]
        size = torch.tensor(sim_points.shape[-2:]).to(map_t_cam_samples)  # 256, 256
        valid = torch.all((map_t_cam_samples >= 0) & (map_t_cam_samples < size), -1)
        pose_scores = pose_scores * valid

        return map_T_cam_samples, pose_scores, bev_ij_pool, map_ij_pool, gt_pose_score

    def _forward(self, data):

        # Revert memory layout back to spatial layout
        spatial_map = torch.rot90(data["map"], 1, dims=(-2, -1))

        pred = {}
        pred_map = pred["map"] = self.map_encoder({**data, "map": spatial_map})
        f_map = pred_map["map_features"][0]
        batch_size = f_map.shape[0]

        # Extract image features.
        level = 0
        f_image = self.image_encoder(data)["feature_maps"][level]
        camera = data["camera"].scale(1 / self.image_encoder.scales[level])
        camera = camera.to(data["image"].device, non_blocking=True)

        # Estimate the monocular priors.
        pred["pixel_scales"] = scales = self.scale_classifier(f_image.moveaxis(1, -1))
        f_polar = self.projection_polar(f_image, scales, camera)

        # Map to the BEV.
        with torch.autocast("cuda", enabled=False):
            f_bev, valid_bev, _ = self.projection_bev(
                f_polar.float(), None, camera.float()
            )

        if self.conf.bev_net is None:
            # channel last -> classifier -> channel first
            f_bev = self.feature_projection(f_bev.moveaxis(1, -1)).moveaxis(-1, 1)
        else:
            pred_bev = pred["bev"] = self.bev_net({"input": f_bev})
            f_bev = pred_bev["output"]

        # Convert bev to memory layout
        f_bev = pred["bev"]["output"] = torch.rot90(
            f_bev, -1, dims=(-2, -1)
        )  # B, C, I, J
        confidence_bev = pred_bev.get("confidence")  # B, I, J
        if confidence_bev is not None:
            confidence_bev = pred["bev"]["confidence"] = torch.rot90(
                confidence_bev, -1, dims=(-2, -1)
            )
        valid_bev = torch.rot90(valid_bev, -1, dims=(-2, -1))  # B, I, J
        f_map = pred["map"]["map_features"][0] = torch.rot90(f_map, -1, dims=(-2, -1))
        map_mask = data.get(
            "map_mask", torch.ones((batch_size, *f_map.shape[-2:])).to(valid_bev)
        )

        if self.bev_ij_pts is None:
            self.bev_ij_pts = self.build_query_grid(f_bev).to(f_map)

        if self.conf.use_map_cutout:  # for evaluating matchers
            f_bev, valid_cutout = neural_cutout(
                self.bev_ij_pts, f_map, data["map_T_cam"]
            )
            valid_bev = valid_bev & valid_cutout
            if confidence_bev is not None:
                confidence_bev = pred_bev["confidence"] = (
                    torch.ones_like(confidence_bev) * valid_bev
                )  # / valid_bev.sum((-1, -2))
            pred["bev"]["output"] = f_bev

        if "log_prior" in pred["map"]:
            log_prior = pred["map"]["log_prior"][0] = torch.rot90(
                pred_map["log_prior"][0], -1, dims=(-2, -1)
            )

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
            if "log_prior" in pred_map and self.conf.apply_map_prior:
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

            # TODO: Revert mem layout to snap's. Remove when template sampler is fixed
            f_bev = pred["bev"]["output"] = torch.rot90(f_bev, -1, dims=(-2, -1))
            if confidence_bev is not None:
                confidence_bev = pred["bev"]["confidence"] = torch.rot90(
                    confidence_bev, -1, dims=(-2, -1)
                )
            valid_bev = torch.rot90(valid_bev, -1, dims=(-2, -1))
            pred["scores"] = scores
            pred["log_probs"] = log_probs

        else:  # SNAP's RANSAC matcher

            sim_points, prob_points = self.compute_similarity(
                f_bev, f_map, valid_bev, confidence_bev
            )

            map_T_cam_samples, pose_scores, bev_ij_pool, map_ij_pool, gt_pose_score = (
                self.ransac_voting(
                    sim_points, prob_points, valid_bev, map_mask, data["map_T_cam"]
                )
            )

            _, max_indices = torch.max(pose_scores, dim=-1)
            select_fn = torch.vmap(lambda x, i: x[i[None]][0])
            map_T_cam_max = select_fn(map_T_cam_samples, max_indices)

            if self.conf.ransac_grid_refinement:
                if self.conf.profiler_mode:
                    torch.cuda.reset_peak_memory_stats()
                    with profile(
                        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                        profile_memory=True,
                    ) as prof:
                        with record_function("grid_refinement"):
                            pred["map_T_cam_ransac"] = map_T_cam_max
                            (
                                map_T_cam_max,
                                pred["refined_score"],
                                pred["scores_grid_refine"],
                            ) = grid_refinement_batched(
                                map_T_cam_max,
                                sim_points,
                                self.bev_ij_pts,
                                valid_bev,
                                map_mask,
                            )

                    cuda_time = prof.key_averages()[0].cuda_time / 1000
                    stats = torch.cuda.memory_stats()
                    peak_bytes = stats["allocated_bytes.all.peak"] / 1024**3
                    print(f"[grid_refinement]: {cuda_time:.3f} ms, {peak_bytes:.2f} GB")

                else:
                    pred["map_T_cam_ransac"] = map_T_cam_max
                    (
                        map_T_cam_max,
                        pred["refined_score"],
                        pred["scores_grid_refine"],
                    ) = grid_refinement_batched(
                        map_T_cam_max,
                        sim_points,
                        self.bev_ij_pts,
                        valid_bev,
                        map_mask,
                    )

            map_T_cam_max = Transform2D(map_T_cam_max)
            resolution = 1 / data["pixels_per_meter"][..., None]
            tile_T_cam_max = Transform2D.from_pixels(map_T_cam_max, resolution)
            tile_T_cam_samples = Transform2D.from_pixels(
                Transform2D(map_T_cam_samples), resolution
            )

            # TODO: if we want to apply map prior, then we need to grid sample the
            # logprob at sampled pose locations

            # Dummy
            tile_T_cam_avg = tile_T_cam_max
            map_T_cam_avg = map_T_cam_max
            pred["log_probs"] = torch.zeros((1, 256, 256, 360))

            pred.update(
                {
                    "bev_ij_pool": bev_ij_pool,
                    "map_ij_pool": map_ij_pool,
                    "map_T_cam_samples": map_T_cam_samples,
                    "tile_T_cam_samples": tile_T_cam_samples,
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
            "features_image": f_image,
            "features_bev": f_bev,
            "valid_bev": valid_bev.squeeze(1),
        }

    def loss(self, pred, data):

        # Revert refactored outputs to original
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
        ransac_only_metrics = {}
        if self.conf.ransac_matcher:
            ransac_only_metrics["samples_recall_8m_10°"] = SamplesRecall(
                xy_thresh=8, yaw_thresh=5, key="tile_T_cam_samples"
            )
            ransac_only_metrics["samples_recall_8m_5°"] = SamplesRecall(
                xy_thresh=8, yaw_thresh=5, key="tile_T_cam_samples"
            )
            ransac_only_metrics["samples_recall_4m_10°"] = SamplesRecall(
                xy_thresh=4, yaw_thresh=5, key="tile_T_cam_samples"
            )
            ransac_only_metrics["samples_recall_4m_5°"] = SamplesRecall(
                xy_thresh=4, yaw_thresh=5, key="tile_T_cam_samples"
            )

        return {
            **ransac_only_metrics,
            "xy_max_error": Location2DError("tile_T_cam_max"),
            "xy_expectation_error": Location2DError("tile_T_cam_expectation"),
            "yaw_max_error": AngleError("tile_T_cam_max"),
            "xy_recall_2m": Location2DRecall(2.0, key="tile_T_cam_max"),
            "xy_recall_5m": Location2DRecall(5.0, key="tile_T_cam_max"),
            "yaw_recall_2°": AngleRecall(2.0, "tile_T_cam_max"),
            "yaw_recall_5°": AngleRecall(5.0, "tile_T_cam_max"),
        }
