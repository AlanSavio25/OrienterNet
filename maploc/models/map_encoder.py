# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch
import torch.nn as nn
from torch.nn.functional import interpolate

from .base import BaseModel
from .feature_extractor import FeatureExtractor


class MapEncoder(BaseModel):
    default_conf = {
        "num_encoders": 1,
        "embedding_dim": "???",
        "output_dim": None,
        "num_classes": "???",
        "backbone": "???",
        "unary_prior": False,
    }

    def _init(self, conf):
        if conf.num_encoders > 1:
            self.embeddings = nn.ModuleList(
                torch.nn.ModuleDict(
                    {
                        k: torch.nn.Embedding(n + 1, conf.embedding_dim)
                        for k, n in conf.num_classes.items()
                    }
                )
                for _ in range(conf.num_encoders)
            )
        else:
            self.embeddings = torch.nn.ModuleDict(
                {
                    k: torch.nn.Embedding(n + 1, conf.embedding_dim)
                    for k, n in conf.num_classes.items()
                }
            )

        input_dim = len(conf.num_classes) * conf.embedding_dim
        output_dim = conf.output_dim
        if output_dim is None:
            output_dim = conf.backbone.output_dim
        if conf.unary_prior:
            output_dim += 1
        if conf.backbone is None:
            self.encoder = nn.Conv2d(input_dim, output_dim, 1)
        elif conf.backbone == "simple":
            self.encoder = nn.Sequential(
                nn.Conv2d(input_dim, 128, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, output_dim, 3, padding=1),
            )
        else:
            if conf.num_encoders > 1:
                self.encoder = nn.ModuleList(
                    [
                        FeatureExtractor(
                            {
                                **conf.backbone[i],
                                "input_dim": input_dim,
                                "output_dim": output_dim,
                            }
                        )
                        for i in range(conf.num_encoders)
                    ]
                )
            else:
                self.encoder = FeatureExtractor(
                    {
                        **conf.backbone,
                        "input_dim": input_dim,
                        "output_dim": output_dim,
                    }
                )

    def _forward(self, data):
        # pred = {"map_features": {}}
        pred = {k: {} for k in data["map"]}
        for idx, k in enumerate(data["map"]):
            if self.conf.num_encoders > 1:
                embeddings = [
                    self.embeddings[idx][key](data["map"][k][:, i])
                    for i, key in enumerate(("areas", "ways", "nodes"))
                ]
            else:
                embeddings = [
                    self.embeddings[key](data["map"][k][:, i])
                    for i, key in enumerate(("areas", "ways", "nodes"))
                ]
            embeddings = torch.cat(embeddings, dim=-1).permute(0, 3, 1, 2)
            if isinstance(self.encoder, BaseModel) or isinstance(
                self.encoder, nn.ModuleList
            ):
                if self.conf.num_encoders > 1:
                    assert len(self.encoder) == len(
                        data["map"]
                    ), "Number of maps does not match num encoders"
                    encoder = self.encoder[idx]
                else:
                    encoder = self.encoder
                features = encoder(
                    {
                        "image": embeddings,
                        "out_scale_idx": 0 if self.conf.num_encoders > 1 else idx,
                    }
                )["feature_maps"]
            else:
                # if self.conf.num_encoders > 1:
                #     features = [self.encoder[idx](embeddings)]
                # else:
                features = [self.encoder(embeddings)]

            if self.conf.unary_prior:
                pred[k]["log_prior"] = [f[:, -1] for f in features]
                features = [f[:, :-1] for f in features]

            pred[k]["map_features"] = features
        return pred
