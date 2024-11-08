# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch
import torch.nn as nn
from torch.nn.functional import interpolate

from .base import BaseModel
from .feature_extractor import AdaptationBlock, FeatureExtractor


class MapEncoder(BaseModel):
    """Outputs neural map with semantic/aerial inputs."""

    default_conf = {
        "map_types": ["semantic"],
        "num_encoders": 1,
        "embedding_dim": "???",  # semantic
        "num_classes": "???",  # semantic
        "output_dim": None,  # final output = matching dim
        "backbone": "???",
        "unary_prior": False,
    }
    mean = [0.485, 0.456, 0.406]  # for aerial images
    std = [0.229, 0.224, 0.225]

    def _init(self, conf):

        # Semantic
        if "semantic" in conf.map_types:
            self.embeddings = torch.nn.ModuleDict(
                {
                    k: torch.nn.Embedding(n + 1, conf.embedding_dim)
                    for k, n in conf.num_classes.items()
                }
            )

        # Aerial
        if "aerial" in conf.map_types:
            self.register_buffer("mean_", torch.tensor(self.mean), persistent=False)
            self.register_buffer("std_", torch.tensor(self.std), persistent=False)
            self.conv1x1 = torch.nn.Conv2d(
                3,
                conf.embedding_dim,  # * len(conf.num_classes), # match the size of osm embedding
                kernel_size=1,
                padding=0,
                bias=True,
            )

        # Early fusion combines both inputs before passing through Feature Extractor
        input_dim = (
            len(conf.num_classes) + (1 if "aerial" in conf.map_types else 0)
        ) * conf.embedding_dim  # * len(conf.map_types)
        output_dim = conf.output_dim  # 8
        if output_dim is None:
            output_dim = conf.backbone.output_dim
        if conf.unary_prior:
            output_dim += 1
        if conf.backbone is None:
            self.encoder = nn.ModuleList(
                nn.Conv2d(input_dim, output_dim, 1) for _ in range(conf.num_encoders)
            )
        elif conf.backbone == "simple":
            self.encoder = nn.ModuleList(
                nn.Sequential(
                    nn.Conv2d(input_dim, 128, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 128, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, output_dim, 3, padding=1),
                )
                for _ in range(conf.num_encoders)
            )
        else:
            self.encoder = nn.ModuleList(
                FeatureExtractor(  # this takes as input both maps
                    {
                        **conf.backbone,
                        "input_dim": input_dim,
                        # "output_dim": output_dim,
                    }
                )
                for _ in range(conf.num_encoders)
            )

        # Adaptation layers
        adaptation = []
        for _ in conf.backbone.output_scales:
            block = AdaptationBlock(conf.backbone.decoder[-1], output_dim)
            adaptation.append(block)
        self.adaptation = nn.ModuleList(adaptation)

    def fuse_neural_maps(self, feature_maps):
        """Fuse aerial and semantic features maps with dropout"""
        """Not used currently because we switched to early fusion"""
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
        pred = {
            k: {}
            for k in [map_dict for map_dict in data.values() if map_dict is not None][0]
        }
        # assert (
        #     self.conf.num_encoders == len(self.conf.backbone.output_scales) == len(pred)
        # )  # TODO: remove this
        for idx, k in enumerate(pred):
            features = []
            # Semantic
            if data.get("semantic_map"):
                if data["semantic_map"][k] is None:
                    continue
                features += [
                    self.embeddings[key](data["semantic_map"][k][:, i]).permute(
                        0, 3, 1, 2
                    )
                    for i, key in enumerate(("areas", "ways", "nodes"))
                ]  # 48 dim

            # Aerial
            # Conv 1x1 to simply change the channel dims
            if data.get("aerial_map"):
                if data["aerial_map"][k] is None:
                    continue
                aerial_map = (
                    data["aerial_map"][k] - self.mean_[:, None, None]
                ) / self.std_[:, None, None]
                features += [self.conv1x1(aerial_map)]

            features = torch.cat(features, dim=-3)

            if isinstance(self.encoder[0], BaseModel):
                features = self.encoder[idx]({"image": features, "encoder_idx": idx})[
                    "feature_maps"
                ]
                features = self.adaptation[idx](features)
            else:
                features = self.encoder[idx](features)

            if self.conf.unary_prior:
                pred[k]["log_prior"] = [f[:, -1] for f in features]
                features = [f[:, :-1] for f in features]

            pred[k]["map_features"] = features
        return pred
