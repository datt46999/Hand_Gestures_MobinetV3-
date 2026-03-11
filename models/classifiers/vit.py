import torch
import torchvision
from torch import nn, Tensor

from collections import OrderedDict
from typing import Dict


def VitB16(**kwargs):
    return Vit(patch_size = 16, **kwargs)

class Vit(nn.Module):
    def __init__(self, num_class: Dict, pretrained: bool= False, patch_size: int = 16, *args, **kwargs):
        super().__init__()
        self.hagrid_model = getattr(torchvision.models, f"vit_{patch_size}" )(pretrained)
        heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
        heads_layers["head"] = nn.Linear(in_features = self.hagrid_model.hidden_dim, out_features = num_class)
        self.hagrid_model = nn.Sequential(heads_layers)

        nn.init.zeros_(self.hagrid_model.heads.head_weight)
        nn.init.zeros_(self.hagrid_model.heads.bias)

    def forward(self, x:Tensor):
        gesture = self.hagrid_model(x)
        return gesture
        