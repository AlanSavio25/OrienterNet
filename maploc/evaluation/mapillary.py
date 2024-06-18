# Copyright (c) Meta Platforms, Inc. and affiliates.

import argparse
from pathlib import Path
from typing import Optional, Tuple
from hydra import compose, initialize

from omegaconf import DictConfig, OmegaConf

from .. import logger
from ..conf import data as conf_data_dir
from ..data import MapillaryDataModule
from .run import evaluate, evaluate_chain

split_overrides = {
    "val": {
        "scenes": [
            "sanfrancisco_soma",
            "sanfrancisco_hayes",
            "amsterdam",
            "berlin",
            "lemans",
            "montrouge",
            "toulouse",
            "nantes",
            "vilnius",
            "avignon",
            "helsinki",
            "milan",
            "paris",
        ],
    },
}
data_cfg_train = OmegaConf.load(Path(conf_data_dir.__file__).parent / "mapillary.yaml")
data_cfg = OmegaConf.merge(
    data_cfg_train,
    {
        "return_gps": True,
        "add_map_mask": False,
        "max_init_error": 32,
        "loading": {"val": {"batch_size": 1, "num_workers": 0}},
    },
)
default_cfg_single = OmegaConf.create({"data": data_cfg})
default_cfg_sequential = OmegaConf.create(
    {
        **default_cfg_single,
        "chunking": {
            "max_length": 10,
        },
    }
)


def run(
    split: str,
    experiment: str,
    cfg_path: Path = None,
    cfg: Optional[DictConfig] = None,
    sequential: bool = False,
    thresholds: Tuple[int] = (1, 3, 5),
    **kwargs,
):

    default = default_cfg_sequential if sequential else default_cfg_single
    default = OmegaConf.merge(default, dict(data=split_overrides[split]))

    if cfg_path is not None:
        cfgs = []
        if isinstance(cfg, dict):
            cfg = OmegaConf.create(cfg)
        dataset = MapillaryDataModule(OmegaConf.merge(default, cfg).get("data", {}))
        for conf_p in cfg_path:
            conf = OmegaConf.load(conf_p)
            OmegaConf.resolve(conf)
            conf = OmegaConf.merge(default, conf)
            conf = OmegaConf.merge(conf, cfg)
            cfgs.append(conf)
    else:
        cfg = cfg or {}
        if isinstance(cfg, dict):
            cfg = OmegaConf.create(cfg)

        cfg = OmegaConf.merge(default, cfg)
        cfgs = [cfg]
        dataset = MapillaryDataModule(cfg.get("data", {}))

    if len(experiment) == 1:
        metrics = evaluate(
            experiment[0], cfgs[0], dataset, split, sequential=sequential, **kwargs
        )
    else:
        metrics = evaluate_chain(experiment, cfgs, dataset, split, **kwargs)

    keys = [
        "xy_max_error",
        "xy_gps_error",
        "yaw_max_error",
    ]
    if sequential:
        keys += [
            "xy_seq_error",
            "xy_gps_seq_error",
            "yaw_seq_error",
            "yaw_gps_seq_error",
        ]
    for k in keys:
        if k not in metrics:
            logger.warning("Key %s not in metrics.", k)
            continue
        rec = metrics[k].recall(thresholds).double().numpy().round(2).tolist()
        logger.info("Recall %s: %s at %s m/°", k, rec, thresholds)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", nargs="*", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["val"])
    parser.add_argument("--sequential", action="store_true")
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--num", type=int)
    parser.add_argument("--plot_images", action="store_true")
    parser.add_argument("--select_images_from_logs", nargs="*", type=Path)
    parser.add_argument("--cfg_path", nargs="*", type=Path)
    parser.add_argument("dotlist", nargs="*")
    args = parser.parse_args()
    cfg = OmegaConf.from_cli(args.dotlist)
    run(
        args.split,
        args.experiment,
        args.cfg_path,
        cfg,
        args.sequential,
        output_dir=args.output_dir,
        num=args.num,
        plot_images=args.plot_images,
        select_images_from_logs=args.select_images_from_logs,
    )
