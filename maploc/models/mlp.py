import torch.nn as nn
from torch.nn.functional import normalize

from .base import BaseModel

class MLP(BaseModel):
    """A simple MLP"""

    default_conf = {
        "activation": "ReLU",
        "layers": [], # outputdim for each layer
        "apply_input_activation": True,
        "input_dim": "???",
        "output_dim": "???"
    }

    def _init(self, conf):

        # initialize torch nn linear layer
        activation = getattr(nn, conf.activation)
        input_dim = conf.input_dim
        
        layers = []
        for i, output_dim in enumerate(conf.layers):
            if i > 0 or conf.apply_input_activation:
                layers.append(activation())
            layers.append(nn.Linear(input_dim, output_dim))
            input_dim = output_dim
        self.layers = nn.Sequential(*layers)

    def _forward(self, data):
        return self.layers(data)
