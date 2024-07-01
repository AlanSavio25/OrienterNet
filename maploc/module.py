# Copyright (c) Meta Platforms, Inc. and affiliates.

from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
import numpy as np
from lightning_fabric.utilities.apply_func import move_data_to_device
from lightning_utilities.core.apply_func import apply_to_collection
from omegaconf import DictConfig, OmegaConf, open_dict
from torchmetrics import MeanMetric, MetricCollection

from maploc.evaluation.viz import plot_example_single

from . import logger
from .models import get_model


class AverageKeyMeter(MeanMetric):
    def __init__(self, key, *args, **kwargs):
        self.key = key
        super().__init__(*args, **kwargs)

    def update(self, dict):
        value = dict[self.key]
        value = value[torch.isfinite(value)]
        return super().update(value)


class GenericModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        name = cfg.model.get("name")
        name = "orienternet" if name in ("localizer_bev_depth", None) else name
        self.model = get_model(name)(cfg.model)
        self.cfg = cfg
        self.save_hyperparameters(cfg)
        self.metrics_val = MetricCollection(self.model.metrics(), prefix="val/")
        self.losses_val = None  # we do not know the loss keys in advance

    def forward(self, batch):
        return self.model(batch)

    def training_step(self, batch):
        pred = self(batch)
        losses = self.model.loss(pred, batch)
        self.log_dict(
            {f"loss/{k}/train": v.mean() for k, v in losses.items()},
            prog_bar=True,
            rank_zero_only=True,
        )
        return losses["total"].mean()

    def validation_step(self, batch, batch_idx):
        pred = self(batch)
        losses = self.model.loss(pred, batch)
        if self.losses_val is None:
            self.losses_val = MetricCollection(
                {k: AverageKeyMeter(k).to(self.device) for k in losses},
                prefix="loss/",
                postfix="/val",
            )
        results = self.metrics_val(pred, batch)
        self.log_dict(self.metrics_val, sync_dist=True)
        self.losses_val.update(losses)
        self.log_dict(self.losses_val, sync_dist=True)
        if batch_idx == 0 or batch_idx == 20:
            batch = move_data_to_device(batch, "cpu")
            batch = apply_to_collection(batch, torch.Tensor, lambda x: x[0])
            pred = move_data_to_device(pred, "cpu")
            pred = apply_to_collection(pred, torch.Tensor, lambda x: x[0])
            results = {k[4:]: results[k] for k in results}
            plots = plot_example_single(
                0,
                self,
                pred,
                batch,
                results,
                out_dir=None,
                show_gps=True,
                return_plots=True,
            )
            for i, plot in enumerate(plots):
                self.logger.experiment.add_image(
                    f"Visualizations/{batch_idx}/{i}", plot, self.global_step
                )

    def validation_epoch_start(self, batch):
        self.losses_val = None

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.training.lr)
        ret = {"optimizer": optimizer}
        cfg_scheduler = self.cfg.training.get("lr_scheduler")
        if cfg_scheduler is not None:
            scheduler = getattr(torch.optim.lr_scheduler, cfg_scheduler.name)(
                optimizer=optimizer, **cfg_scheduler.get("args", {})
            )
            ret["lr_scheduler"] = {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "loss/total/val",
                "strict": True,
                "name": "learning_rate",
            }
        return ret

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path,
        map_location=None,
        hparams_file=None,
        strict=True,
        cfg=None,
        find_best=False,
    ):
        assert hparams_file is None, "hparams are not supported."

        checkpoint = torch.load(
            checkpoint_path, map_location=map_location or (lambda storage, loc: storage)
        )
        if find_best:
            best_score, best_name = None, None
            modes = {"min": torch.lt, "max": torch.gt}
            for key, state in checkpoint["callbacks"].items():
                if not key.startswith("ModelCheckpoint"):
                    continue
                mode = eval(key.replace("ModelCheckpoint", ""))["mode"]
                if best_score is None or modes[mode](
                    state["best_model_score"], best_score
                ):
                    best_score = state["best_model_score"]
                    best_name = Path(state["best_model_path"]).name
            logger.info("Loading best checkpoint %s", best_name)
            if best_name != checkpoint_path:
                return cls.load_from_checkpoint(
                    Path(checkpoint_path).parent / best_name,
                    map_location,
                    hparams_file,
                    strict,
                    cfg,
                    find_best=False,
                )

        logger.info(
            "Using checkpoint %s from epoch %d and step %d.",
            checkpoint_path.name,
            checkpoint["epoch"],
            checkpoint["global_step"],
        )
        cfg_ckpt = checkpoint[cls.CHECKPOINT_HYPER_PARAMS_KEY]
        if list(cfg_ckpt.keys()) == ["cfg"]:  # backward compatibility
            cfg_ckpt = cfg_ckpt["cfg"]
        cfg_ckpt = OmegaConf.create(cfg_ckpt)

        if cfg is None:
            cfg = {}
        if not isinstance(cfg, DictConfig):
            cfg = OmegaConf.create(cfg)
        with open_dict(cfg_ckpt):
            cfg = OmegaConf.merge(cfg_ckpt, cfg)

        return pl.core.saving._load_state(cls, checkpoint, strict=strict, cfg=cfg)

    def transfer_batch_to_device(self, batch, device, dataloader_idx) -> Any:

        if isinstance(
            batch["pixels_per_meter"], dict
        ):  # TODO: this needs to be something else
            if self.training:
                scale_idx = int(
                    np.random.choice(np.arange(len(self.cfg.model.bev_mapper.z_max)))
                )
            else:
                if batch.get("scale_idx", None) is not None:
                    scale_idx = batch.get("scale_idx")[0].item()
                else:
                    scale_idx = 1 if len(self.cfg.model.bev_mapper.z_max) > 1 else 0 # Fixed for validation
            z_max = self.cfg.model.bev_mapper.z_max[scale_idx]
            batch["scale_idx"] = torch.tensor(scale_idx).unsqueeze(0)
            keys = [
                "map_mask",
                "map_t_gps",
                "tile_t_gps",
                "accuracy_gps",
                "semantic_map",
                "tile_T_cam",
                "map_T_cam",
                "map_t_init",
                "pixels_per_meter",
                "canvas",
                "z_max",
                "bev_ppm",
            ]
            for k in keys:
                if k not in batch:
                    continue
                batch[k] = batch[k][z_max]

        return super().transfer_batch_to_device(batch, device, dataloader_idx)
