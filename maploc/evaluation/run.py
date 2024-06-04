# Copyright (c) Meta Platforms, Inc. and affiliates.

import functools
from itertools import islice
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything
from torchmetrics import MetricCollection
from tqdm import tqdm

from maploc.utils.wrappers import Transform2D

from .. import EXPERIMENTS_PATH, logger
from ..data.torch import collate, unbatch_to_device
from ..models.metrics import AngleError, LateralLongitudinalError, Location2DError
from ..models.sequential import GPSAligner, RigidAligner
from ..models.voting import argmax_xyr, fuse_gps
from ..module import GenericModule
from ..utils.io import DATA_URL, download_file, read_json
from .utils import write_dump
from .viz import plot_example_sequential, plot_example_single

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
def evaluate_single_image(
    dataloader: torch.utils.data.DataLoader,
    model: GenericModule,
    num: Optional[int] = None,
    callback: Optional[Callable] = None,
    progress: bool = True,
    mask_index: Optional[Tuple[int]] = None,
    has_gps: bool = False,
    **kwargs
):
    ppm = model.model.conf.pixel_per_meter
    metrics = MetricCollection(model.model.metrics())
    metrics["directional_error"] = LateralLongitudinalError()
    if has_gps:
        metrics["xy_gps_error"] = Location2DError("tile_t_gps")
        metrics["xy_fused_error"] = Location2DError("tile_T_fused")
        metrics["yaw_fused_error"] = AngleError("tile_T_fused")
    metrics = metrics.to(model.device)

    names = []
    for i, batch_ in enumerate(
        islice(tqdm(dataloader, total=num, disable=not progress), num)
    ):
        batch = model.transfer_batch_to_device(batch_, model.device, i)

        if kwargs.get("selected_images"):
            if batch["name"] in kwargs.get("selected_images"):
                continue

        # Ablation: mask semantic classes
        if mask_index is not None:
            mask = batch["map"][0, mask_index[0]] == (mask_index[1] + 1)
            batch["map"][0, mask_index[0]][mask] = 0
        pred = model(batch)

        if has_gps:
            map_t_gps = pred["map_t_gps"] = batch["map_t_gps"]
            pred["log_probs_fused"] = fuse_gps(
                pred["log_probs"],
                map_t_gps,
                ppm,
                sigma=batch["accuracy_gps"],
                gaussian=True,
                refactored=True,
            )  # memory_layout

            # argmax_xyr returns the "uv" coordinates on the memory layout
            uvt_fused = argmax_xyr(pred["log_probs_fused"])

            # Note: the rotation dimension (last dim) of log_probs_fused is
            # still ordered acc. to north-clockwise yaw convention.

            ij_fused = torch.flip(uvt_fused[..., :2], dims=[-1])
            yaw_fused = 90 - uvt_fused[..., -1]
            map_T_fused = Transform2D.from_degrees(yaw_fused.unsqueeze(-1), ij_fused)
            pred["tile_T_fused"] = Transform2D.from_pixels(map_T_fused, 1 / ppm)

            pred["tile_t_gps"] = Transform2D.from_pixels(map_t_gps, 1 / ppm)
            del ij_fused, uvt_fused, yaw_fused
        names += batch["name"]

        results = metrics(pred, batch)
        if callback is not None:
            callback(
                i, model, unbatch_to_device(pred), unbatch_to_device(batch_), results
            )
        del batch_, batch, pred, results

    return metrics.cpu(), names


@torch.no_grad()
def evaluate_sequential(
    dataset: torch.utils.data.Dataset,
    chunk2idx: Dict,
    model: GenericModule,
    num: Optional[int] = None,
    shuffle: bool = False,
    callback: Optional[Callable] = None,
    progress: bool = True,
    num_rotations: int = 512,
    mask_index: Optional[Tuple[int]] = None,
    has_gps: bool = False,
):
    chunk_keys = list(chunk2idx)
    if shuffle:
        chunk_keys = [chunk_keys[i] for i in torch.randperm(len(chunk_keys))]
    if num is not None:
        chunk_keys = chunk_keys[:num]
    lengths = [len(chunk2idx[k]) for k in chunk_keys]
    logger.info(
        "Min/max/med lengths: %d/%d/%d, total number of images: %d",
        min(lengths),
        np.median(lengths),
        max(lengths),
        sum(lengths),
    )
    viz = callback is not None

    metrics = MetricCollection(model.model.metrics())
    ppm = model.model.conf.pixel_per_meter
    metrics["directional_error"] = LateralLongitudinalError(ppm)
    metrics["xy_seq_error"] = Location2DError("uv_seq", ppm)
    metrics["yaw_seq_error"] = AngleError("yaw_seq")
    metrics["directional_seq_error"] = LateralLongitudinalError(ppm, key="uv_seq")
    if has_gps:
        metrics["xy_gps_error"] = Location2DError("uv_gps", ppm)
        metrics["xy_gps_seq_error"] = Location2DError("uv_gps_seq", ppm)
        metrics["yaw_gps_seq_error"] = AngleError("yaw_gps_seq")
    metrics = metrics.to(model.device)

    keys_save = ["uvr_max", "uv_max", "yaw_max", "uv_expectation"]
    if has_gps:
        keys_save.append("uv_gps")
    if viz:
        keys_save.append("log_probs")

    for chunk_index, key in enumerate(tqdm(chunk_keys, disable=not progress)):
        indices = chunk2idx[key]
        aligner = RigidAligner(track_priors=viz, num_rotations=num_rotations)
        if has_gps:
            aligner_gps = GPSAligner(track_priors=viz, num_rotations=num_rotations)
        batches = []
        preds = []
        for i in indices:
            data = dataset[i]
            data = model.transfer_batch_to_device(data, model.device, 0)
            pred = model(collate([data]))

            canvas = data["canvas"]
            data["xy_geo"] = xy = canvas.to_xy(data["uv"].double())
            data["yaw"] = yaw = data["roll_pitch_yaw"][-1].double()
            aligner.update(pred["log_probs"][0], canvas, xy, yaw)

            if has_gps:
                (uv_gps) = pred["uv_gps"] = data["uv_gps"][None]
                xy_gps = canvas.to_xy(uv_gps.double())
                aligner_gps.update(xy_gps, data["accuracy_gps"], canvas, xy, yaw)

            if not viz:
                data.pop("image")
                data.pop("map")
            batches.append(data)
            preds.append({k: pred[k][0] for k in keys_save})
            del pred

        xy_gt = torch.stack([b["xy_geo"] for b in batches])
        yaw_gt = torch.stack([b["yaw"] for b in batches])
        aligner.compute()
        xy_seq, yaw_seq = aligner.transform(xy_gt, yaw_gt)
        if has_gps:
            aligner_gps.compute()
            xy_gps_seq, yaw_gps_seq = aligner_gps.transform(xy_gt, yaw_gt)
        results = []
        for i in range(len(indices)):
            preds[i]["uv_seq"] = batches[i]["canvas"].to_uv(xy_seq[i]).float()
            preds[i]["yaw_seq"] = yaw_seq[i].float()
            if has_gps:
                preds[i]["uv_gps_seq"] = (
                    batches[i]["canvas"].to_uv(xy_gps_seq[i]).float()
                )
                preds[i]["yaw_gps_seq"] = yaw_gps_seq[i].float()
            results.append(metrics(preds[i], batches[i]))
        if viz:
            callback(chunk_index, model, batches, preds, results, aligner)
        del aligner, preds, batches, results
    return metrics.cpu()


def select_images_from_log(log_paths):


    if len(log_paths) == 0:
        raise ValueError("At least one log path must be provided")
    elif len(log_paths) == 1:
        raise NotImplementedError
    if len(log_paths) > 1:

        logs = {}

        sorted_names = None
        for i, log_path in enumerate(log_paths):
            log_data = read_json(Path(log_path))
            if not log_data:
                raise ValueError("Log data is empty")
            print(log_data)
            sorted_names = sorted(log_data["names"])
            logs[i] = list(zip(log_data["errors"]["xy_max_error"], log_data["names"]))
            logs[i] = [err for err, _ in sorted(logs[i], key=lambda x: x[1])]

        diff = np.array(logs[0]) - np.array(logs[len(log_paths)-1])
        selected_images = [n for value, n in zip(diff, sorted_names)]

    return selected_images[:20]

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

    dataset.prepare_data()
    dataset.setup()

    plot_images = kwargs.get("plot_images")
    if output_dir is not None:
        output_dir.mkdir(exist_ok=True, parents=True)
        if callback is None and plot_images:
            if sequential:
                callback = plot_example_sequential
            else:
                callback = plot_example_single
            callback = functools.partial(
                callback, out_dir=output_dir, return_plots=True, **(viz_kwargs or {})
            )
    kwargs = {**kwargs, "callback": callback}

    if kwargs.get("select_images_from_logs"):
        kwargs["selected_images"] = select_images_from_log(kwargs.get("select_images_from_logs"))
    seed_everything(dataset.cfg.seed)
    if sequential:
        dset, chunk2idx = dataset.sequence_dataset(split, **cfg.chunking)
        metrics = evaluate_sequential(dset, chunk2idx, model, **kwargs)
    else:
        loader = dataset.dataloader(split, shuffle=True, num_workers=num_workers)
        metrics, names = evaluate_single_image(loader, model, **kwargs)

    results = metrics.compute()
    logger.info("All results: %s", results)
    if output_dir is not None and not plot_images:
        write_dump(output_dir, experiment, cfg, results, metrics, names)
        logger.info("Outputs have been written to %s.", output_dir)
    return metrics
