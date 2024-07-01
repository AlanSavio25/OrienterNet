# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch.nn as nn
from torchvision.models.resnet import Bottleneck

from maploc.models.mlp import MLP

from .base import BaseModel
from .feature_extractor import AdaptationBlock
from .utils import checkpointed


class BEVNet(BaseModel):
    default_conf = {
        "pretrained": True,
        "num_blocks": "???",
        "latent_dim": "???",
        "input_dim": "${.latent_dim}",
        "output_dim": "${.latent_dim}",
        "confidence": False,
        "norm_layer": "nn.BatchNorm2d",  # normalization ind decoder blocks
        "checkpointed": False,  # whether to use gradient checkpointing
        "padding": "zeros",
        "mlp": None,
    }

    def _init(self, conf):
        blocks = []
        Block = checkpointed(Bottleneck, do=conf.checkpointed)
        for i in range(conf.num_blocks):
            dim = conf.input_dim if i == 0 else conf.latent_dim
            blocks.append(
                Block(
                    dim,
                    conf.latent_dim // Bottleneck.expansion,
                    norm_layer=eval(conf.norm_layer),
                )
            )
        self.blocks = nn.Sequential(*blocks)  # 71k params

        # When net is multi-scale, we cannot use conv2d, so we use mlp
        if conf.mlp is not None:
            self.mlp = MLP(conf.mlp)  # 66k params

        self.output_layer = AdaptationBlock(
            conf.latent_dim, conf.output_dim
        )  # 1032 params

        if conf.confidence:
            self.confidence_layer = AdaptationBlock(conf.latent_dim, 1)

        def update_padding(module):
            if isinstance(module, nn.Conv2d):
                module.padding_mode = conf.padding

        if conf.padding != "zeros":
            self.blocks.apply(update_padding)

    def _forward(self, data):
        features = self.blocks(data["input"])
        if self.conf.mlp is not None:
            features = self.mlp(features.moveaxis(-3, -1)).moveaxis(-1, -3)
        pred = {
            "output": self.output_layer(features),
        }
        if self.conf.confidence:
            pred["confidence"] = self.confidence_layer(features).squeeze(1).sigmoid()
        return pred

    def loss(self, pred, data):
        raise NotImplementedError

    def metrics(self, pred, data):
        raise NotImplementedError
