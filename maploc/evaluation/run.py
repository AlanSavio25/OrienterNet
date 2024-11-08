# Copyright (c) Meta Platforms, Inc. and affiliates.

import functools
from itertools import islice
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf, read_write
from pytorch_lightning import seed_everything
from torchmetrics import MetricCollection
from tqdm import tqdm

from lightning_utilities.core.apply_func import apply_to_collection
from maploc.utils.wrappers import Transform2D
from ..utils.geo import BoundaryBox

from .. import EXPERIMENTS_PATH, logger
from ..data.torch import collate, unbatch_to_device
from ..models.sequential import GPSAligner, RigidAligner
from ..models.voting import argmax_xyr, fuse_gps, log_softmax_spatial
from ..utils.grids import grid_refinement_orienternet_batched
from maploc.utils import grids
from ..module import GenericModule
from ..utils.io import DATA_URL, download_file, read_json
from .utils import write_dump
from .viz import plot_example_sequential, plot_example_single
from ..models.metrics import (
    AngleError,
    AngleRecall,
    ExhaustiveEntropy,
    Location2DError,
    Location2DRecall,
    LateralLongitudinalError,
)
from copy import deepcopy

from ..osm.viz import Colormap
from ..utils.viz_2d import features_to_RGB, plot_images, save_plot

pretrained_models = dict(
    OrienterNet_MGL=("orienternet_mgl.ckpt", dict(num_rotations=256)),
)


def resolve_checkpoint_path(experiment_or_path: str) -> Path:
    path = Path(experiment_or_path)
    if not path.exists():
        # provided name of experiment
        path = Path(EXPERIMENTS_PATH, *experiment_or_path.split("/"))
        if not path.exists():
            if experiment_or_path in set(p for p, _ in pretrained_models.values()):
                download_file(f"{DATA_URL}/{experiment_or_path}", path)
            else:
                raise FileNotFoundError(path)
    if path.is_file():
        return path
    # provided only the experiment name
    maybe_path = path / "last-step.ckpt"
    if not maybe_path.exists():
        maybe_path = path / "step.ckpt"
    if not maybe_path.exists():
        raise FileNotFoundError(f"Could not find any checkpoint in {path}.")
    return maybe_path


@torch.no_grad()
def evaluate_single_image_chain(
    dataloader: torch.utils.data.DataLoader,
    models: List[GenericModule],
    num: Optional[int] = None,
    callback: Optional[Callable] = None,
    progress: bool = True,
    mask_index: Optional[Tuple[int]] = None,
    has_gps: bool = False,
    **kwargs,
):

    # ppm = models[0].model.conf.pixel_per_meter
    # metrics = MetricCollection(models[0].model.metrics())
    metrics = {}
    for model in models:
        metrics.update(model.model.metrics())
    # metrics = models[0].model.metrics()
    # if models[0].model.conf.bev_mapper.multiscale:
    modes = [""]
    if models[0].model.conf.grid_refinement:
        modes += ["_refined"]
    values = [
        ("0_5", 0.5),
        ("01", 1.0),
        ("02", 2.0),
        ("05", 5.0),
        ("10", 10.0),
        ("20", 20.0),
    ]
    metrics.update(
        {
            "exhaustive_entropy_chain": ExhaustiveEntropy("log_probs", f"chain"),
        }
    )
    for mode in modes:
        metrics.update(
            {
                f"xy_max_error_chain{mode}": Location2DError(
                    "tile_T_cam_max", f"chain{mode}"
                ),
                f"yaw_max_error_chain{mode}": AngleError(
                    "tile_T_cam_max", f"chain{mode}"
                ),
            }
        )
        for val_str, val_float in values:
            metrics.update(
                {
                    f"xy_recall_{val_str}m_chain{mode}": Location2DRecall(
                        val_float, "tile_T_cam_max", f"chain{mode}"
                    ),
                    f"yaw_recall_{val_str}°_chain{mode}": AngleRecall(
                        val_float, "tile_T_cam_max", f"chain{mode}"
                    ),
                }
            )

    metrics = MetricCollection(metrics)
    metrics = metrics.to(models[0].device)

    names = []

    for i, batch_ in enumerate(
        islice(tqdm(dataloader, total=num, disable=not progress), num)
    ):

        preds = []
        batches = []
        # batches_ = []
        if kwargs.get("selected_images"):
            if batch_["name"][0] not in kwargs.get("selected_images"):
                continue
        # for scale_idx, model in enumerate(models):

        if kwargs["singlemodel_randomscale"]:
            model = models[0]
            bev_depths = batch_["z_max"]
            for bev_depth in bev_depths:
                # the index is where this is in the bev model.
                scale_idx = models[0].model.conf.bev_mapper.z_max.index(bev_depth)
                batch_["scale_idx"] = torch.tensor([scale_idx])
                batch = model.transfer_batch_to_device(
                    deepcopy(batch_), model.device, i
                )
                pred = model(batch)
                preds.append(pred)
                batches.append(batch)
        else:
            for model in models:
                # batch_["scale_idx"] = torch.tensor([scale_idx])
                # scale_idx = list(batch_["bev_ppm"].values()).index(
                #     model.model.conf.bev_mapper.pixel_per_meter
                # )
                # batch_["scale_idx"] = torch.tensor([scale_idx])

                batch = model.transfer_batch_to_device(batch_, model.device, i)
                model_batch = deepcopy(batch)
                # batch has to have only the current model's z_max
                z_max = model.model.conf.bev_mapper.z_max[
                    0
                ]  # in this function, we are only dealing with multiscale models with single z_max
                for key in model_batch:
                    if isinstance(model_batch[key], dict) and z_max in model_batch[key]:
                        for k in [
                            non_model_k
                            for non_model_k in model_batch[key]
                            if non_model_k != z_max
                        ]:
                            del model_batch[key][k]

                # del batch["scale_idx"], batch["z_max"]  # , batch["bev_ppm"]
                pred = model(model_batch)
                preds.append(pred)
                batches.append(batch)
                # batches_.append(batch_)

        # scores = [preds[i][k]["scores"] for i,k in enumerate(batch_["z_max"])]
        # h = w = max([score.shape[-2] for score in scores])
        # scores = [
        #     torch.nn.functional.interpolate(
        #         score.moveaxis(-1, -3), size=(h, w), mode="bilinear"
        #     ).moveaxis(-3, -1)
        #     for score in scores
        # ]

        # log_probs = [log_softmax_spatial(score) for score in scores]
        # log_probs_chained = log_softmax_spatial(torch.stack(log_probs).sum(0))

        # pred = preds[0]
        pred = {}
        for p in preds:
            pred.update(p)
        model = models[0]
        batch = batches[0]
        # batch_ = batches_[0]

        # Multiply probability volumes of all branches to "chain" results

        if models[0].cfg.data.add_map_mask:
            scores = [
                pred[k]["scores_unmasked"] for k in pred if isinstance(k, (float, int))
            ]
        else:
            scores = [pred[k]["scores"] for k in pred if isinstance(k, (float, int))]
        crop_size_meters = models[0].cfg.data.crop_size_meters[0]
        upsample_ppm = max(models[0].model.conf.pixel_per_meter)
        h = w = (
            crop_size_meters * 2 * upsample_ppm
        )  # max([score.shape[-2] for score in scores])
        scores = [
            torch.nn.functional.interpolate(
                score.moveaxis(-1, -3), size=(int(h), int(w)), mode="bilinear"
            ).moveaxis(-3, -1)
            for score in scores
        ]

        if models[0].cfg.data.add_map_mask:
            # map_mask = batch["map_mask"][32.0]
            map_mask = pred[min([k for k in pred if isinstance(k, (float, int))])][
                "map_mask"
            ]
            scores = [
                score.masked_fill_(~map_mask[..., None], -np.inf) for score in scores
            ]

        log_probs = [
            weight * log_softmax_spatial(score)
            for score, weight in zip(scores, kwargs["chain_weights"])
        ]
        log_probs_chained = log_softmax_spatial(torch.stack(log_probs).sum(0))

        uvr_max = argmax_xyr(log_probs_chained)
        ij_max = torch.flip(uvr_max[..., :2], dims=[-1])
        yaw_max = 180 - uvr_max[..., -1]
        map_T_max = Transform2D.from_degrees(yaw_max.unsqueeze(-1), ij_max)

        # pred["map_T_cam_max"] = map_T_max
        # upsample_ppm = max(model.cfg.data.bev_ppm)
        # pred["tile_T_cam_max"] = Transform2D.from_pixels(
        #     map_T_max, 1 / upsample_ppm
        # )
        # pred["log_probs"] = log_probs_chained

        pred["chain"] = {}
        pred["chain"]["map_T_cam_max"] = map_T_max
        pred["chain"]["tile_T_cam_max"] = tile_T_cam_max_chained = (
            Transform2D.from_pixels(map_T_max, 1 / upsample_ppm)
        )
        pred["chain"]["log_probs"] = log_probs_chained
        # TODO: this is probably a problem

        batch["tile_T_cam"]["chain"] = batch["tile_T_cam"][
            models[0].model.conf.bev_mapper.z_max[0]
        ]

        if model.model.conf.grid_refinement:
            delta_p = 0.5  # m
            range_p = 2  # m
            delta_r = 1.0  # deg
            range_r = 5.0  # deg

            poses = []
            pose_scores_list = []
            for idx, k in enumerate(model.cfg.data.z_max):
                # for idx, k in enumerate(models[0].model.conf.bev_mapper.z_max):

                resolution = 1 / models[idx].model.conf.pixel_per_meter[0]  # [idx]
                map_T_cam_max_chained = Transform2D.to_pixels(
                    tile_T_cam_max_chained, resolution
                )
                bev_ij_pts = models[idx].model.bev_mapper.cam_xy_pts[0] / resolution
                bev_ij_pts = Transform2D(torch.Tensor([-90, 0, 0])) @ bev_ij_pts

                # Perform grid refinement
                _, _, map_T_cam_samples, pose_scores = (
                    grid_refinement_orienternet_batched(
                        map_T_cam_max_chained._data,
                        preds[idx][k]["features_map"],
                        preds[idx][k]["features_bev"],
                        bev_ij_pts.to(preds[idx][k]["features_map"]),
                        preds[idx][k]["valid_bev"],
                        batches[idx].get(
                            "map_mask",
                            {
                                k: torch.ones(
                                    (preds[idx][k]["features_map"][:, 0, ...].shape)
                                ).to(preds[idx][k]["valid_bev"])
                            },
                        )[k],
                        delta_p / resolution,
                        range_p / resolution,
                        delta_r,
                        range_r,
                    )
                )
                # convert pose scores to probabilities
                pose_log_probs = torch.nn.functional.log_softmax(pose_scores.flatten())
                poses.append(map_T_cam_samples)
                pose_scores_list.append(pose_log_probs)

            # Sum up log probs. Then,
            # chained_pose_scores = [pose_log_probs for pose_log_probs in pose_scores_list]
            # TODO: add weighting?
            pose_scores_list = [
                weight * score
                for score, weight in zip(pose_scores_list, kwargs["chain_weights"])
            ]
            chained_pose_scores = torch.stack(pose_scores_list).sum(0)
            _, best_idx = torch.max(chained_pose_scores, dim=-1)
            map_T_cam_chain_refined = poses[0].squeeze(0)[best_idx].unsqueeze(0)

            pred["chain_refined"] = {}
            pred["chain_refined"]["map_T_cam_max"] = Transform2D(
                map_T_cam_chain_refined
            )
            tile_T_cam_chain_refined = Transform2D.from_pixels(
                Transform2D(map_T_cam_chain_refined), 1 / upsample_ppm
            )
            pred["chain_refined"]["tile_T_cam_max"] = tile_T_cam_chain_refined
            batch["tile_T_cam"]["chain_refined"] = batch["tile_T_cam"][
                model.model.conf.bev_mapper.z_max[0]
            ]

        names += batch["name"]

        results = metrics(pred, batch)
        if callback is not None:
            callback(
                i, model, unbatch_to_device(pred), unbatch_to_device(batch), results
            )
        del batches, preds, results

    return metrics.cpu(), names


@torch.no_grad()
def evaluate_single_image(
    dataloader: torch.utils.data.DataLoader,
    model: GenericModule,
    num: Optional[int] = None,
    callback: Optional[Callable] = None,
    progress: bool = True,
    mask_index: Optional[Tuple[int]] = None,
    has_gps: bool = True,
    **kwargs,
):
    ppm = model.model.conf.pixel_per_meter
    metrics = model.model.metrics()
    # if model.model.conf.bev_mapper.multiscale:
    modes = [""]
    if model.model.conf.grid_refinement:
        modes += ["_refined"]
    values = [
        ("0_5", 0.5),
        ("01", 1.0),
        ("02", 2.0),
        ("05", 5.0),
        ("10", 10.0),
        ("20", 20.0),
    ]
    metrics.update(
        {
            "exhaustive_entropy_chain": ExhaustiveEntropy("log_probs", f"chain"),
        }
    )
    for mode in modes:
        metrics.update(
            {
                f"xy_max_error_chain{mode}": Location2DError(
                    "tile_T_cam_max", f"chain{mode}"
                ),
                f"yaw_max_error_chain{mode}": AngleError(
                    "tile_T_cam_max", f"chain{mode}"
                ),
            }
        )
        for val_str, val_float in values:
            metrics.update(
                {
                    f"xy_recall_{val_str}m_chain{mode}": Location2DRecall(
                        val_float, "tile_T_cam_max", f"chain{mode}"
                    ),
                    f"yaw_recall_{val_str}°_chain{mode}": AngleRecall(
                        val_float, "tile_T_cam_max", f"chain{mode}"
                    ),
                }
            )

    # metrics["directional_error"] = LateralLongitudinalError()
    if has_gps:
        scale_choice_idx = 0
        scale_choice = list(model.model.conf.bev_mapper.z_max)[scale_choice_idx]
        for val_str, val_float in values:
            metrics.update(
                {
                    f"xy_gps_recall_{val_str}m": Location2DRecall(
                        val_float, "tile_t_gps", scale_choice
                    ),
                    f"xy_gps_fused_recall_{val_str}m": Location2DRecall(
                        val_float, "tile_T_fused", scale_choice
                    ),
                    f"yaw_gps_fused_recall_{val_str}°": AngleRecall(
                        val_float, "tile_T_fused", scale_choice
                    ),
                }
            )
        metrics["xy_gps_error"] = Location2DError("tile_t_gps", scale_choice)
        metrics["xy_gps_fused_error"] = Location2DError("tile_T_fused", scale_choice)
        metrics["yaw_gps_fused_error"] = AngleError("tile_T_fused", scale_choice)
    metrics = MetricCollection(metrics)
    metrics = metrics.to(model.device)

    names = []
    for i, batch_ in enumerate(
        islice(tqdm(dataloader, total=num, disable=not progress), num)
    ):
        batch = model.transfer_batch_to_device(batch_, model.device, i)

        if kwargs.get("selected_images"):
            if batch["name"][0] not in kwargs.get("selected_images"):
                continue

        # Ablation: mask semantic classes
        if mask_index is not None:
            mask = batch["map"][0, mask_index[0]] == (mask_index[1] + 1)
            batch["map"][0, mask_index[0]][mask] = 0
        pred = model(batch)

        names += batch["name"]

        # if model.model.conf.bev_mapper.multiscale:
        # Multiply probability volumes of all branches to "chain" results

        if model.cfg.data.add_map_mask:
            scores = [
                pred[k]["scores_unmasked"] for k in pred if isinstance(k, (float, int))
            ]
        else:
            scores = [pred[k]["scores"] for k in pred if isinstance(k, (float, int))]
        crop_size_meters = model.cfg.data.crop_size_meters[0]
        upsample_ppm = max(model.model.conf.pixel_per_meter)
        h = w = (
            crop_size_meters * 2 * upsample_ppm
        )  # max([score.shape[-2] for score in scores])
        scores = [
            torch.nn.functional.interpolate(
                score.moveaxis(-1, -3), size=(int(h), int(w)), mode="bilinear"
            ).moveaxis(-3, -1)
            for score in scores
        ]

        if model.cfg.data.add_map_mask:
            # map_mask = batch["map_mask"][32.0]
            map_mask = pred[min([k for k in pred if isinstance(k, (float, int))])][
                "map_mask"
            ]
            scores = [
                score.masked_fill_(~map_mask[..., None], -np.inf) for score in scores
            ]

        log_probs = [
            weight * log_softmax_spatial(score)
            for score, weight in zip(scores, kwargs["chain_weights"])
        ]
        log_probs_chained = log_softmax_spatial(torch.stack(log_probs).sum(0))

        # pred = preds[0]
        # model = models[0]
        # batch = batches[0]

        uvr_max = argmax_xyr(log_probs_chained)
        ij_max = torch.flip(uvr_max[..., :2], dims=[-1])
        yaw_max = 180 - uvr_max[..., -1]
        map_T_max = Transform2D.from_degrees(yaw_max.unsqueeze(-1), ij_max)
        pred["chain"] = {}
        pred["chain"]["map_T_cam_max"] = map_T_max
        pred["chain"]["tile_T_cam_max"] = tile_T_cam_max_chained = (
            Transform2D.from_pixels(map_T_max, 1 / upsample_ppm)
        )
        pred["chain"]["log_probs"] = log_probs_chained
        batch["tile_T_cam"]["chain"] = batch["tile_T_cam"][
            model.model.conf.bev_mapper.z_max[0]
        ]
        # batch["features_map"]["chain"] = batch["features_map"][32.0]
        # features_bev, valid_bev, pixel_Scales, semantic_map have to be added for visualization
        # for now, we skip visualization, and just focus on the numbers
        # if "tile_t_gps" in batch:
        # batch["tile_t_gps"]["chain"] = batch["tile_t_gps"][32.0]

        if has_gps:
            # Evaluate either on a single map (each z_max maps to a different map)
            map_t_gps = batch["map_t_gps"][scale_choice]
            pred[scale_choice]["log_probs_fused"] = fuse_gps(
                pred["chain"]["log_probs"],
                map_t_gps,
                ppm[scale_choice_idx],
                sigma=batch["accuracy_gps"][scale_choice],
                gaussian=True,
                refactored=True,
            )  # memory_layout
            # TODO: refactor code for scale_choice_idx and upsample ppm to be same
            uvr_gps_max = argmax_xyr(pred[scale_choice]["log_probs_fused"])
            ij_gps_max = torch.flip(uvr_gps_max[..., :2], dims=[-1])
            yaw_gps_max = 180 - uvr_gps_max[..., -1]
            map_T_gps = Transform2D.from_degrees(yaw_gps_max.unsqueeze(-1), ij_gps_max)
            pred[scale_choice]["tile_T_fused"] = tile_T_gps_fused_max = (
                Transform2D.from_pixels(map_T_gps, 1 / ppm[scale_choice_idx])
            )
            pred[scale_choice]["tile_t_gps"] = Transform2D.from_pixels(
                map_t_gps, 1 / ppm[scale_choice_idx]
            )

        if model.model.conf.grid_refinement:
            delta_p = 0.5  # m
            range_p = 2  # m
            delta_r = 1.0  # deg
            range_r = 5.0  # deg

            poses = []
            pose_scores_list = []
            for idx, k in enumerate(model.model.conf.bev_mapper.z_max):

                resolution = 1 / model.model.conf.pixel_per_meter[idx]
                map_T_cam_max_chained = Transform2D.to_pixels(
                    tile_T_cam_max_chained, resolution
                )
                bev_ij_pts = model.model.bev_mapper.cam_xy_pts[idx] / resolution
                bev_ij_pts = Transform2D(torch.Tensor([-90, 0, 0])) @ bev_ij_pts

                # Perform grid refinement
                _, _, map_T_cam_samples, pose_scores = (
                    grid_refinement_orienternet_batched(
                        map_T_cam_max_chained._data,
                        pred[k]["features_map"],
                        pred[k]["features_bev"],
                        bev_ij_pts.to(pred[k]["features_map"]),
                        pred[k]["valid_bev"],
                        batch.get(
                            "map_mask",
                            {
                                k: torch.ones(
                                    (pred[k]["features_map"][:, 0, ...].shape)
                                ).to(pred[k]["valid_bev"])
                            },
                        )[k],
                        delta_p / resolution,
                        range_p / resolution,
                        delta_r,
                        range_r,
                    )
                )
                if (
                    model.model.conf.add_temperature
                    and model.model.conf.apply_temperature
                ):
                    temp = torch.exp(-model.model.temperature[idx])
                else:
                    temp = 1.0
                # convert pose scores to probabilities
                pose_log_probs = torch.nn.functional.log_softmax(
                    temp * pose_scores.flatten()
                )
                poses.append(map_T_cam_samples)
                pose_scores_list.append(pose_log_probs)

            # Sum up log probs. Then,
            # chained_pose_scores = [pose_log_probs for pose_log_probs in pose_scores_list]
            pose_scores_list = [
                weight * score
                for score, weight in zip(pose_scores_list, kwargs["chain_weights"])
            ]
            chained_pose_scores = torch.stack(pose_scores_list).sum(0)
            _, best_idx = torch.max(chained_pose_scores, dim=-1)
            map_T_cam_chain_refined = poses[0].squeeze(0)[best_idx].unsqueeze(0)

            pred["chain_refined"] = {}
            pred["chain_refined"]["map_T_cam_max"] = Transform2D(
                map_T_cam_chain_refined
            )
            tile_T_cam_chain_refined = Transform2D.from_pixels(
                Transform2D(map_T_cam_chain_refined), 1 / upsample_ppm
            )
            pred["chain_refined"]["tile_T_cam_max"] = tile_T_cam_chain_refined
            batch["tile_T_cam"]["chain_refined"] = batch["tile_T_cam"][
                model.model.conf.bev_mapper.z_max[0]
            ]

        results = metrics(pred, batch)

        # if not (results["xy_max_error_chain"] < 4 and results["xy_max_error_chain"] < results["xy_max_error_128"] < results["xy_max_error_32"]):
        #     continue

        # mining good examples for thesis
        if results["xy_max_error_chain"] > 2:
            continue

        if callback is not None:
            callback(
                i,
                model,
                unbatch_to_device(pred),
                unbatch_to_device(batch),
                results,
                return_plots=True,
            )
        del batch_, batch, pred, results

    return metrics.cpu(), names


# def set_key_value_to_none(batch, key_to_set_none):
#     """Turn off values in batch to force single branch forward pass"""
#     dict_with_none_value = {key: ({k: (None if k == key_to_set_none else v) for k, v in batch[key].items()}
#                 if isinstance(batch[key], dict) else batch[key]) for key in batch}
#     return dict_with_none_value


def disable_key(batch, target_keys, remove_key=False):
    if not isinstance(target_keys, list):
        target_keys = [target_keys]
    ret = {}
    for key in batch:
        if isinstance(batch[key], dict):
            ret.update(
                {
                    key: {
                        k: (None if k in target_keys else v)
                        for k, v in batch[key].items()
                        if not (remove_key and k in target_keys)
                    }
                }
            )
        else:
            ret.update({key: batch[key]})
    return ret


@torch.no_grad()
def evaluate_hierarchical(
    dataloader: torch.utils.data.DataLoader,
    model: GenericModule,  # our multi-scale model
    num: Optional[int] = None,
    callback: Optional[Callable] = None,
    progress: bool = True,
    mask_index: Optional[Tuple[int]] = None,
    has_gps: bool = True,
    prior_model: GenericModule = None,  # coarse prior model
    **kwargs,
):
    """Evaluate single image with minimal memory requirement.
    Match prior model (coarse res, large depth) on a large map first (optional),
    then hierarchically match the coarse and fine BEVs on the best map crops.

    Default:
        Prior model: Res=4mpp, BEVDepth=256m,
        Model: Res=(0.5mpp, 2mpp), BEVDepth=(32m,128m)
        256px map radius corresponds to {64,256,1024}m at {0.5,2,4}mpp, respectively.
    """

    # Setup metrics for chain
    metrics = model.model.metrics()
    modes = [""]
    values = [
        ("0_5", 0.5),
        ("01", 1.0),
        ("02", 2.0),
        ("05", 5.0),
        ("10", 10.0),
        ("20", 20.0),
    ]
    metrics.update(
        {
            "exhaustive_entropy_chain": ExhaustiveEntropy("log_probs", f"chain"),
        }
    )
    for mode in modes:
        metrics.update(
            {
                f"xy_max_error_chain{mode}": Location2DError(
                    "tile_T_cam_max", f"chain{mode}"
                ),
                f"yaw_max_error_chain{mode}": AngleError(
                    "tile_T_cam_max", f"chain{mode}"
                ),
            }
        )
        for val_str, val_float in values:
            metrics.update(
                {
                    f"xy_recall_{val_str}m_chain{mode}": Location2DRecall(
                        val_float, "tile_T_cam_max", f"chain{mode}"
                    ),
                    f"yaw_recall_{val_str}°_chain{mode}": AngleRecall(
                        val_float, "tile_T_cam_max", f"chain{mode}"
                    ),
                }
            )

    if has_gps:
        scale_choice_idx = 0
        scale_choice = list(model.model.conf.bev_mapper.z_max)[scale_choice_idx]
        for val_str, val_float in values:
            metrics.update(
                {
                    f"xy_gps_recall_{val_str}m": Location2DRecall(
                        val_float, "tile_t_gps", scale_choice
                    ),
                    f"xy_gps_fused_recall_{val_str}m": Location2DRecall(
                        val_float, "tile_T_fused", scale_choice
                    ),
                    f"yaw_gps_fused_recall_{val_str}°": AngleRecall(
                        val_float, "tile_T_fused", scale_choice
                    ),
                }
            )
        metrics["xy_gps_error"] = Location2DError("tile_t_gps", scale_choice)
        metrics["xy_gps_fused_error"] = Location2DError("tile_T_fused", scale_choice)
        metrics["yaw_gps_fused_error"] = AngleError("tile_T_fused", scale_choice)
    metrics = MetricCollection(metrics)
    metrics = metrics.to(model.device)
    do_prior = prior_model is not None
    if do_prior:
        p_metrics = prior_model.model.metrics()
        p_metrics = MetricCollection(p_metrics).to(prior_model.device)
        prior_ppm = prior_model.model.conf.pixel_per_meter[0]

    ppm = model.model.conf.pixel_per_meter

    names = []

    # Naming convention: prior=prior model. coarse is coarse branch and fine is fine branch of multiscale model.
    # ptile is prior tile, ctile is coarse tile, ftile is fine tile

    for i, batch_ in enumerate(
        islice(tqdm(dataloader, total=num, disable=not progress), num)
    ):
        batch = model.transfer_batch_to_device(batch_, model.device, i)
        if kwargs.get("selected_images"):  # for plotting
            if batch["name"][0] not in kwargs.get("selected_images"):
                continue

        # TODO: currently, input csm for finer res are non-zero. make zero.

        # config for topk
        num_k_coarse = 3 if do_prior else 1
        num_k_fine = 3 if not do_prior else 1
        csm_coarse = 256
        csm_fine = 64
        # pmap_crad is radius of a cmap in pmap coords
        # for masking - NMS
        pmap_crad = csm_coarse * prior_ppm if do_prior else None
        cmap_frad = csm_fine * ppm[1]

        topk_poses_fine = []
        p_topk_coords = []  # for plotting only

        if prior_model is not None:
            # batch has all 3 scales, set the others to none for a forward pass
            batch_256m = disable_key(batch, [32.0, 128.0], remove_key=True)
            pred_256m = prior_model(batch_256m)
            pscores = pred_256m[256.0]["scores"].clone()
            world_T_ptile = Transform2D.from_Rt(
                torch.eye(2), batch_256m["canvas"][256.0][0].bbox.min_
            ).float()

        # Caching avoids redundant network forward passes
        bev_cache_fine = None
        bev_cache_coarse = None

        # Loop through the topk small tiles on the Prior Map
        for k_idx_coarse in range(num_k_coarse):

            batch_128m = disable_key(batch, [32.0, 256.0])
            if do_prior:
                if k_idx_coarse == 0:
                    pmap_T_maxcam = pred_256m[256.0]["map_T_cam_max"].float().squeeze(0)
                    ptile_T_maxcam = pred_256m[256.0]["tile_T_cam_max"].float()
                else:
                    # Select best pose in NMS-masked score volume
                    uvr_max = argmax_xyr(pscores).to(pscores)
                    pmax_score = pscores.flatten(-3).max(-1).values
                    ij_max = torch.flip(uvr_max[..., :2], dims=[-1])
                    yaw_max = 180 - uvr_max[..., 2][..., None]
                    pmap_T_maxcam = (
                        Transform2D.from_degrees(yaw_max, ij_max).float().squeeze(0)
                    )
                    ptile_T_maxcam = (
                        Transform2D.from_pixels(pmap_T_maxcam, 1 / prior_ppm)
                        .float()
                        .unsqueeze(0)
                    )

                world_T_maxcam = (
                    world_T_ptile.to(ptile_T_maxcam.device) @ ptile_T_maxcam
                ).squeeze(0)

                # Prepare coarse raster bbox
                ctile_manager = dataloader.dataset.tile_managers[batch["scene"][0]][
                    1
                ]  # use index 0 for the 32m ppm
                min_ = np.maximum(
                    world_T_maxcam.t.cpu().numpy() - csm_coarse, world_T_ptile.t
                )
                min_ = np.maximum(min_, ctile_manager.bbox.min_)
                min_ = np.minimum(
                    min_, batch_256m["canvas"][256.0][0].bbox.max_ - csm_coarse * 2
                )
                min_ = np.minimum(min_, ctile_manager.bbox.max_ - (csm_coarse + 1) * 2)
                max_ = min_ + 2 * csm_coarse
                bbox_ctile = BoundaryBox(min_, max_)

                # Query coarse raster
                ccanvas = ctile_manager.query(bbox_ctile)
                craster = torch.from_numpy(np.ascontiguousarray(ccanvas.raster)).long()
                craster = torch.rot90(craster, -1, dims=(-2, -1))

                # Assign queried coarse raster to the batch for forward pass
                craster = craster.unsqueeze(0).to(model.device)
                batch_128m["semantic_map"][128.0] = craster

                pmap_min = Transform2D.to_pixels(
                    world_T_ptile.inv() @ bbox_ctile.min_, 1 / prior_ppm
                ).squeeze()

                # plot the topk tiles
                p_topk_coords.append((pmap_min, pmap_crad * 2, pmap_crad * 2))

                # mask scores - NMS
                x_start, y_start = (pmap_min + 0.25 * pmap_crad).to(int).numpy()
                x_end, y_end = (pmap_min + 1.75 * pmap_crad).to(int).numpy()
                pscores[..., x_start:x_end, y_start:y_end, :] = pred_256m[256.0][
                    "scores"
                ].min()
                x_start, y_start = (
                    (pmap_T_maxcam.t - 0.75 * pmap_crad).cpu().to(int).numpy()
                )
                x_end, y_end = (
                    (pmap_T_maxcam.t + 0.75 * pmap_crad).cpu().to(int).numpy()
                )
                pscores[..., x_start:x_end, y_start:y_end, :] = pred_256m[256.0][
                    "scores"
                ].min()

                world_T_ctile = Transform2D.from_Rt(
                    torch.eye(2), bbox_ctile.min_
                ).float()
                ctile_T_ptile = world_T_ctile.inv() @ world_T_ptile
                ctile_T_cam = (
                    ctile_T_ptile.to(model.device) @ batch_256m["tile_T_cam"][256.0]
                )
                cmap_T_cam = Transform2D.to_pixels(ctile_T_cam, 1 / ccanvas.ppm)
                # TODO: mask edges of the prior scores?
            else:
                ccanvas = None
                craster = None
                bbox_ctile = batch_128m["canvas"][128.0][0].bbox
                ctile_T_cam = batch_128m["tile_T_cam"][128.0].cpu()

                world_T_ctile = Transform2D.from_Rt(
                    torch.eye(2), bbox_ctile.min_
                ).float()

            # disable fine branch of network
            with read_write(model.model.bev_mapper.conf):
                model.model.bev_mapper.conf.z_max[0] = None
                model.model.bev_mapper.conf.z_max[1] = 128.0

            # Coarse Localization
            model.model.bev_cache = bev_cache_coarse
            pred_128m = model(batch_128m)
            bev_cache_coarse = model.model.bev_cache
            cscores = pred_128m[128.0]["scores"].clone()

            c_topk_coords = []

            # Loop through the topk fine maps
            for k_idx_fine in range(num_k_fine):
                if k_idx_fine == 0:
                    cmap_T_maxcam = pred_128m[128.0]["map_T_cam_max"].float().squeeze(0)
                    ctile_T_maxcam = pred_128m[128.0]["tile_T_cam_max"].float()
                else:
                    uvr_max = argmax_xyr(cscores).to(cscores)
                    cmax_score = cscores.flatten(-3).max(-1).values
                    ij_max = torch.flip(uvr_max[..., :2], dims=[-1])
                    yaw_max = 180 - uvr_max[..., 2][..., None]
                    cmap_T_maxcam = (
                        Transform2D.from_degrees(yaw_max, ij_max).float().squeeze(0)
                    )
                    ctile_T_maxcam = (
                        Transform2D.from_pixels(cmap_T_maxcam, 1 / ppm[1])
                        .float()
                        .unsqueeze(0)
                    )
                world_T_maxcam = (
                    world_T_ctile.to(ctile_T_maxcam.device) @ ctile_T_maxcam
                ).squeeze(0)

                # Prepare fine bbox
                ftile_manager = dataloader.dataset.tile_managers[batch["scene"][0]][
                    0
                ]  # use index 0 for the 32m ppm
                min_ = np.maximum(
                    world_T_maxcam.t.cpu().numpy() - csm_fine, world_T_ctile.t
                )
                min_ = np.maximum(min_, ftile_manager.bbox.min_)
                min_ = np.minimum(min_, world_T_ctile.t + csm_coarse * 2 - csm_fine * 2)
                min_ = np.minimum(min_, ftile_manager.bbox.max_ - (csm_fine + 0.5) * 2)
                max_ = min_ + 2 * csm_fine
                bbox_ftile = BoundaryBox(min_, max_)

                # Query fine raster
                fcanvas = ftile_manager.query(bbox_ftile)
                fraster = torch.from_numpy(np.ascontiguousarray(fcanvas.raster)).long()
                fraster = torch.rot90(fraster, -1, dims=(-2, -1))
                batch_32m = disable_key(batch, [128.0, 256.0])
                batch_32m["semantic_map"][32.0] = fraster.unsqueeze(0).to(model.device)

                cmap_min = Transform2D.to_pixels(
                    world_T_ctile.inv() @ bbox_ftile.min_, 1 / ppm[1]
                ).squeeze()

                # plot topk fine maps
                c_topk_coords.append((cmap_min, cmap_frad * 2, cmap_frad * 2))

                # mask scores - NMS
                # x_start, y_start = (cmap_min + 0.25*cmap_frad).to(int).numpy()
                # x_end, y_end = (cmap_min + 1.75*cmap_frad).to(int).numpy()
                x_start, y_start = (
                    (cmap_T_maxcam.t - 0.75 * cmap_frad).cpu().to(int).numpy()
                )
                x_end, y_end = (
                    (cmap_T_maxcam.t + 0.75 * cmap_frad).cpu().to(int).numpy()
                )
                cscores[..., x_start:x_end, y_start:y_end, :] = pred_128m[128.0][
                    "scores"
                ].min()

                # disable coarse branch for forward pass
                with read_write(model.model.bev_mapper.conf):
                    model.model.bev_mapper.conf.z_max[0] = 32.0
                    model.model.bev_mapper.conf.z_max[1] = None

                # Forward pass through fine branch
                model.model.bev_cache = bev_cache_fine
                pred_32m = model(batch_32m)
                bev_cache_fine = model.model.bev_cache
                fscores = pred_32m[32.0]["scores"]

                # Sample coarse scores for all fine points.
                width = depth = csm_fine * 2
                cell_size = 1 / ppm[0]
                grid = grids.Grid2D.from_extent_meters((width, depth), cell_size)
                ftile_xy_pts = grid.index_to_xyz(grid.grid_index())
                fmap_xy_pts = Transform2D.to_pixels(ftile_xy_pts, 1 / ppm[0])
                world_T_ftile = Transform2D.from_Rt(
                    torch.eye(2), bbox_ftile.min_
                ).float()
                ctile_T_ftile = world_T_ctile.inv() @ world_T_ftile
                ctile_xy_pts = ctile_T_ftile @ ftile_xy_pts
                cmap_xy_pts = Transform2D.to_pixels(ctile_xy_pts, 1 / ppm[1])
                cmap_interp, _, _ = grids.interpolate_nd(
                    pred_128m[128.0]["scores"].moveaxis(-1, -3).squeeze(0),
                    cmap_xy_pts.to(cscores.device).reshape(-1, 2),  # 8, H, W  # I*J, 2
                )
                cmap_interp = (
                    cmap_interp.unsqueeze(0).moveaxis(-2, -1).reshape(1, 256, 256, 64)
                )

                # Chain probabilities of the fine and coarse
                log_probs = [
                    log_softmax_spatial(fscores),
                    log_softmax_spatial(cmap_interp),
                ]
                log_probs_chained = log_softmax_spatial(torch.stack(log_probs).sum(0))

                # Extract joint best pose
                uvr_max = argmax_xyr(log_probs_chained)
                ij_max = torch.flip(uvr_max[..., :2], dims=[-1])
                yaw_max = 180 - uvr_max[..., -1]
                fmap_T_maxcam = Transform2D.from_degrees(yaw_max.unsqueeze(-1), ij_max)
                ftile_T_maxcam = Transform2D.from_pixels(fmap_T_maxcam, 1 / ppm[0])

                # debug
                # special_points = {}
                # pred["chain"] = {}
                # pred["chain"]["map_T_cam_max"] = map_T_max
                # pred["chain"]["tile_T_cam_max"] = tile_T_cam_max_chained = (
                #     Transform2D.from_pixels(map_T_max, 1 / upsample_ppm)
                # )
                # pred["chain"]["log_probs"] = log_probs_chained
                # batch["tile_T_cam"]["chain"] = batch["tile_T_cam"][
                #     model.model.conf.bev_mapper.z_max[0]
                # ]

                max_score = (fscores + cmap_interp).flatten(-3).max(-1).values

                # Store predicted pose and score.
                topk_poses_fine.append(
                    (
                        {
                            128.0: deepcopy(pred_128m[128.0]),
                            32.0: deepcopy(pred_32m[32.0]),
                            "chain": {
                                "tile_T_cam_max": ftile_T_maxcam,
                                "map_T_cam_max": fmap_T_maxcam,
                                "max_score": max_score,
                                "log_probs": log_probs_chained,
                            },
                        },
                        fcanvas,
                        fraster.unsqueeze(0).to(model.device),
                        bbox_ftile,
                        ccanvas,
                        craster,
                        bbox_ctile,
                        c_topk_coords,
                    )
                )

            # Reset the cache for the next forward pass
            bev_cache_fine = None

        bev_cache_coarse = None

        # Select best pose out of topk
        (
            best_pred,
            fcanvas,
            fraster,
            bbox_ftile,
            ccanvas,
            craster,
            bbox_ctile,
            c_topk_coords,
        ) = max(topk_poses_fine, key=lambda x: x[0]["chain"]["max_score"])

        pred = {}
        pred["features_image"] = pred_128m["features_image"]
        pred[128.0] = best_pred[128.0]
        pred[32.0] = best_pred[32.0]
        pred["chain"] = best_pred["chain"]

        # debug
        # pred[32.0]['special_points'] = fmap_xy_pts[:100,:100,:]
        # pred[128.0]['special_points'] = cmap_xy_pts[:100,:100,:]

        if do_prior:
            pred_256m[256.0]["topk"] = p_topk_coords
        pred[128.0]["topk"] = c_topk_coords

        # Transform fine GT to smaller map's frame
        # world_T_ctile = Transform2D.from_Rt(torch.eye(2), bbox_ctile.min_).float()
        if do_prior:
            batch["tile_T_cam"][128.0] = ctile_T_cam.to(model.device)
            batch["map_T_cam"][128.0] = cmap_T_cam.to(model.device)
            batch["semantic_map"][128.0] = craster
        world_T_ftile = Transform2D.from_Rt(torch.eye(2), bbox_ftile.min_).float()
        ftile_T_ctile = world_T_ftile.inv() @ world_T_ctile
        ftile_T_cam = ftile_T_ctile.to(model.device) @ ctile_T_cam.to(model.device)
        fmap_T_cam = Transform2D.to_pixels(ftile_T_cam, 1 / fcanvas.ppm)
        batch["tile_T_cam"][32.0] = ftile_T_cam.to(model.device)
        batch["map_T_cam"][32.0] = fmap_T_cam.to(model.device)
        batch["semantic_map"][32.0] = fraster

        # GPS
        ftile_t_gps = ftile_T_ctile.to(model.device).t + batch_128m["tile_t_gps"][128.0]
        fmap_t_gps = Transform2D.to_pixels(ftile_t_gps, 1 / fcanvas.ppm)
        batch["tile_t_gps"][32.0] = ftile_t_gps.to(model.device)
        batch["map_t_gps"][32.0] = fmap_t_gps.to(model.device)
        batch["tile_T_cam"]["chain"] = batch["tile_T_cam"][32.0]

        if model.cfg.data.add_map_mask:
            scores = [
                pred[k]["scores_unmasked"] for k in pred if isinstance(k, (float, int))
            ]
        else:
            scores = [pred[k]["scores"] for k in pred if isinstance(k, (float, int))]

        if has_gps:  # TODO: temporarily turn this off.
            # Evaluate either on a single map (each z_max maps to a different map)
            map_t_gps = batch["map_t_gps"][scale_choice]
            pred[scale_choice]["log_probs_fused"] = fuse_gps(
                pred["chain"]["log_probs"],
                map_t_gps,
                ppm[scale_choice_idx],
                sigma=batch["accuracy_gps"][scale_choice],
                gaussian=True,
                refactored=True,
            )  # memory_layout
            # TODO: refactor code for scale_choice_idx and upsample ppm to be same
            uvr_gps_max = argmax_xyr(pred[scale_choice]["log_probs_fused"])
            ij_gps_max = torch.flip(uvr_gps_max[..., :2], dims=[-1])
            yaw_gps_max = 180 - uvr_gps_max[..., -1]
            map_T_gps = Transform2D.from_degrees(yaw_gps_max.unsqueeze(-1), ij_gps_max)
            pred[scale_choice]["tile_T_fused"] = tile_T_gps_fused_max = (
                Transform2D.from_pixels(map_T_gps, 1 / ppm[scale_choice_idx])
            )
            pred[scale_choice]["tile_t_gps"] = Transform2D.from_pixels(
                map_t_gps, 1 / ppm[scale_choice_idx]
            )

        if model.model.conf.grid_refinement:
            delta_p = 0.5  # m
            range_p = 2  # m
            delta_r = 1.0  # deg
            range_r = 5.0  # deg

            poses = []
            pose_scores_list = []
            for idx, k in enumerate(model.model.conf.bev_mapper.z_max):

                resolution = 1 / model.model.conf.pixel_per_meter[idx]
                map_T_cam_max_chained = Transform2D.to_pixels(
                    tile_T_cam_max_chained, resolution
                )
                bev_ij_pts = model.model.bev_mapper.cam_xy_pts[idx] / resolution
                bev_ij_pts = Transform2D(torch.Tensor([-90, 0, 0])) @ bev_ij_pts

                # Perform grid refinement
                _, _, map_T_cam_samples, pose_scores = (
                    grid_refinement_orienternet_batched(
                        map_T_cam_max_chained._data,
                        pred[k]["features_map"],
                        pred[k]["features_bev"],
                        bev_ij_pts.to(pred[k]["features_map"]),
                        pred[k]["valid_bev"],
                        batch.get(
                            "map_mask",
                            {
                                k: torch.ones(
                                    (pred[k]["features_map"][:, 0, ...].shape)
                                ).to(pred[k]["valid_bev"])
                            },
                        )[k],
                        delta_p / resolution,
                        range_p / resolution,
                        delta_r,
                        range_r,
                    )
                )
                if (
                    model.model.conf.add_temperature
                    and model.model.conf.apply_temperature
                ):
                    temp = torch.exp(-model.model.temperature[idx])
                else:
                    temp = 1.0
                # convert pose scores to probabilities
                pose_log_probs = torch.nn.functional.log_softmax(
                    temp * pose_scores.flatten()
                )
                poses.append(map_T_cam_samples)
                pose_scores_list.append(pose_log_probs)

            # Sum up log probs. Then,
            # chained_pose_scores = [pose_log_probs for pose_log_probs in pose_scores_list]
            pose_scores_list = [
                weight * score
                for score, weight in zip(pose_scores_list, kwargs["chain_weights"])
            ]
            chained_pose_scores = torch.stack(pose_scores_list).sum(0)
            _, best_idx = torch.max(chained_pose_scores, dim=-1)
            map_T_cam_chain_refined = poses[0].squeeze(0)[best_idx].unsqueeze(0)

            pred["chain_refined"] = {}
            pred["chain_refined"]["map_T_cam_max"] = Transform2D(
                map_T_cam_chain_refined
            )
            tile_T_cam_chain_refined = Transform2D.from_pixels(
                Transform2D(map_T_cam_chain_refined), 1 / upsample_ppm
            )
            pred["chain_refined"]["tile_T_cam_max"] = tile_T_cam_chain_refined
            batch["tile_T_cam"]["chain_refined"] = batch["tile_T_cam"][
                model.model.conf.bev_mapper.z_max[0]
            ]

        results = metrics(pred, batch)
        if do_prior:
            prior_results = p_metrics(pred_256m, batch_256m)

        # if not (results["xy_max_error_chain"] < 4 and results["xy_max_error_chain"] < results["xy_max_error_128"] < results["xy_max_error_32"]):
        #     continue

        # mining good examples for thesis
        # if results["xy_max_error_chain"] > 10:
        #     continue

        # mine good examples
        if not (
            results["xy_max_error_chain"] < 2
            and results["xy_max_error_chain"]
            < results["xy_max_error_32"]
            < results["xy_max_error_128"]
        ):
            continue

        names += batch["name"]

        if prior_model is not None:
            if callback is not None:
                callback(
                    i,
                    prior_model,
                    unbatch_to_device(pred_256m),
                    unbatch_to_device(batch_256m),
                    prior_results,
                    return_plots=True,
                )
        if callback is not None:
            callback(
                i,
                model,
                unbatch_to_device(pred),
                unbatch_to_device(batch),
                results,
                return_plots=True,
            )

        del batch_, batch, pred, results

    if prior_model is not None:
        p_metrics = p_metrics.cpu().compute()
        logger.info(f"Prior model results: {p_metrics}")
    return metrics.cpu(), names


def select_images_from_log(log_paths):
    """Return list of images to plot"""

    if len(log_paths) == 0:
        raise ValueError("At least one log path must be provided")
    elif len(log_paths) == 1:
        log_data = read_json(Path(log_paths[0]))
        # best or worst or both with median?
        logs = list(zip(log_data["errors"]["xy_max_error"], log_data["names"]))

        logs = [name for _, name in sorted(logs, key=lambda x: x[0])]

        # logs = [name for (err, name) in logs if 4.0 < err < 9.0]
        selected_images = logs[:15] + logs[-30:]

    if len(log_paths) > 1:

        logs = {}

        sorted_names = None
        for i, log_path in enumerate(log_paths):
            log_data = read_json(Path(log_path))
            if not log_data:
                raise ValueError("Log data is empty")
            if i >= 1:  # todo: remove this
                sorted_names = sorted(log_data["names"])
            # logs[i] = list(
            #     zip(
            #         log_data["errors"]["xy_max_error_32"],
            #         log_data["errors"]["xy_max_error_128"],
            #         log_data["errors"]["xy_max_error_chain"],
            #         log_data["names"],
            #     )
            # )
            if i == 0:
                # logs[i] = list(zip(log_data["errors"]["xy_max_error"], log_data["names"]))
                logs[i] = [log_data["errors"]["xy_max_error"]]  # Orienternet
            else:
                logs[i] = list(log_data["errors"]["xy_max_error_chain"])
                # logs[i] = list(
                #     zip(
                #         log_data["errors"]["xy_max_error_32"],
                #         log_data["errors"]["xy_max_error_128"],
                #         log_data["errors"]["xy_max_error_chain"],
                #         log_data["names"],
                #     )
                # )
            # logs[i] = list(log_data["errors"]["xy_max_error_chain"])

            # We must sort
            # logs[i] = [
            #     x[:-1] for x in sorted(logs[i], key=lambda x: x[-1])
            # ]  # skip the last which is the name

        # selected_images = [n for (n, f, c) in list(zip(sorted_names, logs[0], logs[len(log_paths)-1])) if c < 5 and f > 12]
        # selected_images = [n for (n, f, c, C) in list(zip(sorted_names, logs[0], logs[1], logs[2])) if (f > 0.5 and c <= 0.5) or (f > 1 and c <= 1) or (f > 2 and c <= 2)]
        # selected_images = [
        #     n
        #     for (n, f, c, C) in list(zip(sorted_names, logs[0], logs[1], logs[2]))
        #     if f <= 4
        # diff = -np.array(logs[0]) + np.array(logs[len(log_paths) - 1])
        # selected_images = [
        #     n for value, n in sorted(list(zip(diff, sorted_names)), key=lambda x: x[0])
        # ]

        # selected_images = [
        #     n
        #     for (n, single, multiscale1) in list(zip(sorted_names, logs[0], logs[1]))
        #     # if (
        #     #     single[0] > 15
        #     #     and single[1] > 15  # 32m
        #     #     and single[2] > 15  # 128m
        #     #     and  # chain
        #     #     # multiscale1[0] > 15 and
        #     #     # multiscale1[1] > 15 and
        #     #     multiscale1[2] < 15
        #     # )
        #     if (
        #         # # single[0] > 20
        #         # and single[1] > 15  # 32m
        #         # and single[2] > 15  # 128m
        #         # and
        #         # multiscale1[0] > 15 and
        #         # multiscale1[1] < 3
        #         # and single[2] > 8
        #         # and multiscale1[1] < single[1]
        #         # and multiscale1[0] < single[0]
        #         ##
        #         multiscale1[2] < 5 < 15 < single[2]
        #         ##
        #         # single[2] > 20
        #     )
        # ] # [:25]  # + [
        # #     n
        # #     for (n, single, multiscale) in list(zip(sorted_names, logs[0], logs[1]))
        # #     if (single > 20 and multiscale < 5)
        # # ][
        # #     :25
        # # ]

        selected_images = [
            n
            for (n, orienternet, ours) in list(zip(sorted_names, logs[0], logs[1]))
            if ours < 2.5 < 20 < orienternet
        ]

    return selected_images  # [:50]


def evaluate_chain(
    experiments: List[str],
    cfgs: List[DictConfig],
    dataset,
    split: str,
    output_dir: Optional[Path] = None,
    callback: Optional[Callable] = None,
    num_workers: int = 1,
    viz_kwargs=None,
    **kwargs,
):

    logger.info("Evaluating models %s", experiments)

    models = []
    for experiment, cfg in zip(experiments, cfgs):
        checkpoint_path = resolve_checkpoint_path(experiment)
        model = GenericModule.load_from_checkpoint(
            checkpoint_path, cfg=cfg, find_best=not experiment.endswith(".ckpt")
        )
        model = model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
        models.append(model)

    dataset.prepare_data()
    dataset.setup()

    plot_images = kwargs.get("plot_images")
    if output_dir is not None:
        output_dir.mkdir(exist_ok=True, parents=True)
        if callback is None and plot_images:
            callback = plot_example_single
            callback = functools.partial(
                callback, out_dir=output_dir, return_plots=True, **(viz_kwargs or {})
            )
    kwargs = {**kwargs, "callback": callback}

    if kwargs.get("select_images_from_logs"):
        kwargs["selected_images"] = select_images_from_log(
            kwargs.get("select_images_from_logs")
        )
    seed_everything(dataset.cfg.seed)

    loader = dataset.dataloader(split, shuffle=True, num_workers=num_workers)
    metrics, names = evaluate_single_image_chain(loader, models, **kwargs)

    results = metrics.compute()
    logger.info("All results: %s", results)
    if output_dir is not None and not plot_images:
        write_dump(output_dir, experiments[0], cfg, results, metrics, names)
        logger.info("Outputs have been written to %s.", output_dir)
    return metrics


# def evaluate_chain(
#     experiments: List[str],
#     cfgs: List[DictConfig],
#     dataset,
#     split: str,
#     output_dir: Optional[Path] = None,
#     callback: Optional[Callable] = None,
#     num_workers: int = 1,
#     viz_kwargs=None,
#     **kwargs,
# ):

#     logger.info("Evaluating models %s", experiments)

#     models = []
#     for experiment, cfg in zip(experiments, cfgs):
#         checkpoint_path = resolve_checkpoint_path(experiment)
#         model = GenericModule.load_from_checkpoint(
#             checkpoint_path, cfg=cfg, find_best=not experiment.endswith(".ckpt")
#         )
#         model = model.eval()
#         if torch.cuda.is_available():
#             model = model.cuda()
#         models.append(model)

#     dataset.prepare_data()
#     dataset.setup()

#     plot_images = kwargs.get("plot_images")
#     if output_dir is not None:
#         output_dir.mkdir(exist_ok=True, parents=True)
#         if callback is None and plot_images:
#             callback = plot_example_single
#             callback = functools.partial(
#                 callback, out_dir=output_dir, return_plots=True, **(viz_kwargs or {})
#             )
#     kwargs = {**kwargs, "callback": callback}

#     if kwargs.get("select_images_from_logs"):
#         kwargs["selected_images"] = select_images_from_log(
#             kwargs.get("select_images_from_logs")
#         )
#     seed_everything(dataset.cfg.seed)

#     loader = dataset.dataloader(split, shuffle=True, num_workers=num_workers)
#     metrics, names = evaluate_single_image_chain(loader, models, **kwargs)

#     results = metrics.compute()
#     logger.info("All results: %s", results)
#     if output_dir is not None and not plot_images:
#         write_dump(output_dir, experiments[0], cfg, results, metrics, names)
#         logger.info("Outputs have been written to %s.", output_dir)
#     return metrics


def evaluate(
    experiment: str,
    cfg: DictConfig,
    dataset,
    split: str,
    sequential: bool = False,
    output_dir: Optional[Path] = None,
    callback: Optional[Callable] = None,
    num_workers: int = 1,
    viz_kwargs=None,
    hierarchical: bool = False,
    prior_model: str = None,
    prior_model_cfg_path: str = None,
    **kwargs,
):
    if experiment in pretrained_models:
        experiment, cfg_override = pretrained_models[experiment]
        cfg = OmegaConf.merge(OmegaConf.create(dict(model=cfg_override)), cfg)

    logger.info("Evaluating model %s with config %s", experiment, cfg)
    checkpoint_path = resolve_checkpoint_path(experiment)
    model = GenericModule.load_from_checkpoint(
        checkpoint_path, cfg=cfg, find_best=not experiment.endswith(".ckpt")
    )
    model = model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    if prior_model is not None:
        if prior_model_cfg_path is not None:
            cfg_prior = OmegaConf.load(prior_model_cfg_path)
        else:
            cfg_prior = cfg
        OmegaConf.resolve(cfg_prior)
        if prior_model in pretrained_models:
            prior_model, cfg_override_prior = pretrained_models[prior_model]
            cfg_prior = OmegaConf.merge(
                OmegaConf.create(dict(model=cfg_override_prior)), cfg_prior
            )

        logger.info(
            "Evaluating model %s (PRIOR) with config %s", prior_model, cfg_prior
        )
        checkpoint_path = resolve_checkpoint_path(prior_model)
        prior_model = GenericModule.load_from_checkpoint(
            checkpoint_path, cfg=cfg_prior, find_best=not prior_model.endswith(".ckpt")
        )
        prior_model = prior_model.eval()
        if torch.cuda.is_available():
            prior_model = prior_model.cuda()

    dataset.prepare_data()
    dataset.setup()

    plot_images = kwargs.get("plot_images")
    if output_dir is not None:
        output_dir.mkdir(exist_ok=True, parents=True)
        if callback is None and plot_images:
            callback = plot_example_single
            callback = functools.partial(
                callback, out_dir=output_dir, return_plots=True, **(viz_kwargs or {})
            )
    kwargs = {**kwargs, "callback": callback}

    if kwargs.get("select_images_from_logs"):
        kwargs["selected_images"] = select_images_from_log(
            kwargs.get("select_images_from_logs")
        )
    seed_everything(dataset.cfg.seed)

    loader = dataset.dataloader(split, shuffle=True, num_workers=num_workers)
    if hierarchical:
        # dset, chunk2idx = dataset.sequence_dataset(split, **cfg.chunking)
        # metrics = evaluate_sequential(dset, chunk2idx, model, **kwargs)
        metrics, names = evaluate_hierarchical(
            loader, model, prior_model=prior_model, **kwargs
        )
    else:
        metrics, names = evaluate_single_image(loader, model, **kwargs)

    results = metrics.compute()
    logger.info("All results: %s", results)
    if output_dir is not None and not plot_images:
        write_dump(output_dir, experiment, cfg, results, metrics, names)
        logger.info("Outputs have been written to %s.", output_dir)
    return metrics
