# Copyright (c) Meta Platforms, Inc. and affiliates.

import numpy as np
import torch
from torch.nn.functional import normalize

from maploc.models.bev_mapper import BEVMapper
from maploc.utils.wrappers import Transform2D
from maploc.utils.grids import grid_refinement_orienternet_batched
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
        "multiscale": False,
        # "semantic_encoder": "???",
        "map_encoder": None,
        "aerial_encoder": None,
        "bev_mapper": None,
        # "bev_net": "???",
        "grid_refinement": False,
        "latent_dim": "???",
        "matching_dim": "???",
        "pixel_per_meter": "???",
        "num_rotations": "???",
        "add_temperature": False,
        "normalize_features": False,
        "padding_matching": "replicate",
        "apply_map_prior": True,
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
        "chop_bev": False,
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

    def exhaustive_voting(
        self, template_sampler, f_bev, f_map, valid_bev, confidence_bev=None
    ):
        if self.conf.normalize_features or self.conf.use_map_cutout:
            f_bev = normalize(f_bev, dim=1)
            f_map = normalize(f_map, dim=1)

        # Build the templates and exhaustively match against the map.
        if confidence_bev is not None:
            f_bev = f_bev * confidence_bev.unsqueeze(1)
        f_bev = f_bev.masked_fill(~valid_bev.unsqueeze(1), 0.0)
        templates = template_sampler(f_bev)
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
        valid_templates = template_sampler(valid_bev.float()[None]) > (1 - 1e-4)
        num_valid = valid_templates.float().sum((-3, -2, -1))
        scores = scores / num_valid[..., None, None]
        return scores

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

        # TODO: for later, make data = {32.0: {keys}}, where the shared values can be referenced to avoid increas memory usage
        # This will make data and pred consistent

        # NOTE: for random scale selection, selected choices should be
        # forwarded through args and not through data.

        # Predict BEV from image
        pred = self.bev_mapper(data)

        # Encode aerial/semantic maps
        # note: these maps are in memory layout
        feature_maps = {k: [] for k in self.conf.bev_mapper.z_max}
        if self.map_encoder is not None:
            assert "semantic_map" in data
            semantic_map = self.map_encoder({"map": data["semantic_map"]})
            for i, k in enumerate(semantic_map):
                pred[k]["semantic_map"] = semantic_map[k]
                feature_maps[k].append(pred[k]["semantic_map"]["map_features"][0])

        if self.aerial_encoder is not None:
            assert "aerial_map" in data, "Aerial map not found in data"
            # Todo: make aerial encoder compatible w multiple inputs
            pred[k]["aerial_map"] = self.aerial_encoder({"image": data["aerial_map"]})[
                "feature_maps"
            ][0]
            feature_maps.append(pred["aerial_map"])

        for i, k in enumerate(self.conf.bev_mapper.z_max):

            # Fuse neural maps if semantic and aerial
            if len(feature_maps[k]) == 1:
                f_map = feature_maps[k][0]
            elif len(feature_maps) > 1:
                f_map = self.fuse_neural_maps(feature_maps[k])
            else:
                raise ValueError(f"At least one feature map must be created")

            f_bev, valid_bev, confidence_bev = [
                pred[k]["bev"][key] for key in ["output", "valid_bev", "confidence"]
            ]

            if self.conf.chop_bev:
                half_depth = f_bev.shape[-1] // 2
                valid_bev[..., half_depth:] = False

            all_valid_mask = {k: torch.ones((f_map[:, 0, ...].shape)).to(valid_bev)}
            map_mask = data.get("map_mask", all_valid_mask)[k]

            if map_mask.shape[-2:] != f_map.shape[-2:]:
                nan_mask = torch.where(map_mask, 0, torch.nan)
                nan_mask = torch.nn.functional.interpolate(
                    nan_mask.unsqueeze(1),
                    size=tuple(f_map.shape[-2:]),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
                map_mask = ~torch.isnan(nan_mask)
            # pred[k]["map_mask"] = map_mask if "map_mask" in data else None

            # OrienterNet's Exhaustive Matching

            # Temporarily revert bev format. TODO: refactor template sampler.
            f_bev = pred[k]["bev"]["output"] = torch.rot90(f_bev, 1, dims=(-2, -1))
            if confidence_bev is None or confidence_bev == None:
                raise ValueError
            if confidence_bev is not None:
                confidence_bev = pred[k]["bev"]["confidence"] = torch.rot90(
                    confidence_bev, 1, dims=(-2, -1)
                )
            valid_bev = torch.rot90(valid_bev, 1, dims=(-2, -1))

            template_sampler = self.bev_mapper.template_sampler[i]

            scores = self.exhaustive_voting(
                template_sampler, f_bev, f_map, valid_bev, confidence_bev
            )
            scores = scores.moveaxis(1, -1)  # B,H,W,N

            if (
                "semantic_map" in pred[k]
                and "log_prior" in pred[k]["semantic_map"]
                and self.conf.apply_map_prior
            ):
                log_prior = pred[k]["semantic_map"]["log_prior"][0]
                scores = scores + log_prior.unsqueeze(-1)
            # scores_unmasked = scores.clone()
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

            resolution = 1 / self.conf.pixel_per_meter[i]
            tile_T_cam_max = Transform2D.from_pixels(map_T_cam_max, resolution)
            tile_T_cam_avg = Transform2D.from_pixels(map_T_cam_avg, resolution)

            # Revert mem layout to snap's. TODO: Remove when template sampler is fixed
            f_bev = pred[k]["bev"]["output"] = torch.rot90(f_bev, -1, dims=(-2, -1))
            if confidence_bev is not None:
                confidence_bev = pred[k]["bev"]["confidence"] = torch.rot90(
                    confidence_bev, -1, dims=(-2, -1)
                )
            valid_bev = torch.rot90(valid_bev, -1, dims=(-2, -1))

            if self.conf.grid_refinement:
                bev_ij_pts = self.bev_mapper.cam_xy_pts[i] / resolution
                # BEV faces east in the map frame by default, so we rotate the coords by 90deg
                bev_ij_pts = Transform2D(torch.Tensor([-90, 0, 0])) @ bev_ij_pts
                delta_p = 0.5  # m
                range_p = 2  # m
                delta_r = 1.0  # deg
                range_r = 5.0  # deg
                map_T_cam_max_refined, _, _, _ = grid_refinement_orienternet_batched(
                    map_T_cam_max._data,
                    f_map,
                    f_bev,
                    bev_ij_pts.to(
                        f_map
                    ),  # TODO: construct this for forward and inverse bev mapper
                    valid_bev,
                    map_mask,
                    delta_p / resolution,  # px
                    range_p / resolution,  # px
                    delta_r,
                    range_r,
                )
                tile_T_cam_max_refined = Transform2D.from_pixels(
                    Transform2D(map_T_cam_max_refined), resolution
                )
                pred[k].update(
                    {
                        "tile_T_cam_max_refined": tile_T_cam_max_refined,
                        "map_T_cam_max_refined": map_T_cam_max_refined,
                    }
                )

            pred[k].update(
                {
                    "tile_T_cam_max": tile_T_cam_max,
                    "tile_T_cam_expectation": tile_T_cam_avg,
                    "map_T_cam_max": map_T_cam_max,
                    "map_T_cam_expectation": map_T_cam_avg,
                    "features_map": f_map,
                    "features_bev": f_bev,
                    "valid_bev": valid_bev.squeeze(1),
                    "scores": scores,
                    # "scores_unmasked": scores_unmasked,
                    "log_probs": log_probs,
                }
            )

        return pred

    def loss(self, pred, data):

        loss = {}

        # Revert refactored outputs to original. TODO: update sample_xyr
        for i, k in enumerate(self.conf.bev_mapper.z_max):
            # TODO: if random scale, we can add a continue block - not elegant though
            # if True:
            # k = self.conf.bev_mapper.z_max
            # ij_gt = Transform2D.to_pixels(
            #     data["tile_T_cam"],
            #     1 / data["bev_ppm"].float(),  # self.conf.pixel_per_meter
            # )
            ij_gt = Transform2D.to_pixels(
                data["tile_T_cam"][k], 1 / self.conf.pixel_per_meter[i]
            ).t
            uv_gt = ij_gt.clone()
            uv_gt = torch.flip(ij_gt, dims=[-1])
            yaw_gt = (180 - data["tile_T_cam"][k].angle.squeeze(-1)) % 360
            # log_probs = pred[i]["log_probs"]
            log_probs = pred[k]["log_probs"]

            map_mask = data.get("map_mask")
            if map_mask is not None:
                map_mask[k] = torch.rot90(map_mask[k], 1, dims=(-2, -1))

            if self.conf.do_label_smoothing:
                nll = nll_loss_xyr_smoothed(
                    log_probs,
                    uv_gt,
                    yaw_gt,
                    self.conf.sigma_xy / self.conf.pixel_per_meter,
                    self.conf.sigma_r,
                    mask=map_mask[k],
                )
            else:
                nll = nll_loss_xyr(log_probs, uv_gt, yaw_gt)

            loss[f"nll_{str(int(k))}"] = nll

        loss["total"] = sum(loss.values())

        if self.training and self.conf.add_temperature:
            loss["temperature"] = self.temperature.expand(len(nll))

        return loss

    def metrics(self):
        metrics = {}
        scales = self.conf.bev_mapper.z_max
        if isinstance(scales, (float, int)):
            scales = [scales]
        for s in scales:
            metrics.update(
                {
                    f"exhaustive_entropy_{int(s)}": ExhaustiveEntropy("log_probs", s),
                    f"xy_max_error_{int(s)}": Location2DError("tile_T_cam_max", s),
                    f"yaw_max_error_{int(s)}": AngleError("tile_T_cam_max", s),
                    f"xy_recall_0_5m_{int(s)}": Location2DRecall(
                        0.5, "tile_T_cam_max", s
                    ),
                    f"xy_recall_01m_{int(s)}": Location2DRecall(
                        1.0, "tile_T_cam_max", s
                    ),
                    f"xy_recall_02m_{int(s)}": Location2DRecall(
                        2.0, "tile_T_cam_max", s
                    ),
                    f"xy_recall_05m_{int(s)}": Location2DRecall(
                        5.0, "tile_T_cam_max", s
                    ),
                    f"xy_recall_10m_{int(s)}": Location2DRecall(
                        10.0, "tile_T_cam_max", s
                    ),
                    f"xy_recall_20m_{int(s)}": Location2DRecall(
                        20.0, "tile_T_cam_max", s
                    ),
                    f"yaw_recall_0_5°_{int(s)}": AngleRecall(0.5, "tile_T_cam_max", s),
                    f"yaw_recall_01°_{int(s)}": AngleRecall(1.0, "tile_T_cam_max", s),
                    f"yaw_recall_02°_{int(s)}": AngleRecall(2.0, "tile_T_cam_max", s),
                    f"yaw_recall_05°_{int(s)}": AngleRecall(5.0, "tile_T_cam_max", s),
                    f"yaw_recall_10°_{int(s)}": AngleRecall(10.0, "tile_T_cam_max", s),
                    f"yaw_recall_20°_{int(s)}": AngleRecall(20.0, "tile_T_cam_max", s),
                }
            )
            if self.conf.grid_refinement:
                metrics.update(
                    {
                        f"xy_max_error_{int(s)}_refined": Location2DError(
                            "tile_T_cam_max_refined", s
                        ),
                        f"yaw_max_error_{int(s)}_refined": AngleError(
                            "tile_T_cam_max_refined", s
                        ),
                        f"xy_recall_0_5m_{int(s)}_refined": Location2DRecall(
                            0.5, "tile_T_cam_max_refined", s
                        ),
                        f"xy_recall_01m_{int(s)}_refined": Location2DRecall(
                            1.0, "tile_T_cam_max_refined", s
                        ),
                        f"xy_recall_02m_{int(s)}_refined": Location2DRecall(
                            2.0, "tile_T_cam_max_refined", s
                        ),
                        f"xy_recall_05m_{int(s)}_refined": Location2DRecall(
                            5.0, "tile_T_cam_max_refined", s
                        ),
                        f"xy_recall_10m_{int(s)}_refined": Location2DRecall(
                            10.0, "tile_T_cam_max_refined", s
                        ),
                        f"xy_recall_20m_{int(s)}_refined": Location2DRecall(
                            20.0, "tile_T_cam_max_refined", s
                        ),
                        f"yaw_recall_0_5°_{int(s)}_refined": AngleRecall(
                            0.5, "tile_T_cam_max_refined", s
                        ),
                        f"yaw_recall_01°_{int(s)}_refined": AngleRecall(
                            1.0, "tile_T_cam_max_refined", s
                        ),
                        f"yaw_recall_02°_{int(s)}_refined": AngleRecall(
                            2.0, "tile_T_cam_max_refined", s
                        ),
                        f"yaw_recall_05°_{int(s)}_refined": AngleRecall(
                            5.0, "tile_T_cam_max_refined", s
                        ),
                        f"yaw_recall_10°_{int(s)}_refined": AngleRecall(
                            10.0, "tile_T_cam_max_refined", s
                        ),
                        f"yaw_recall_20°_{int(s)}_refined": AngleRecall(
                            20.0, "tile_T_cam_max_refined", s
                        ),
                    }
                )

        return metrics
