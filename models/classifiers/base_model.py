# input: images: Tensor, targets: dict
# output: criterion(loss val)


import torch 
from torch import nn, Tensor
from typing import Dict
from omegaconf import DictConfig
from models.model import Hagrid

class classifierModel(Hagrid):
    def __init__(self, model: nn.Module, **kwargs):
        super().__init__()
        self.hagrid_model = model(**kwargs)
        self.criterion = None 
    
    def __call__(self, images: Tensor, targets: Dict = None) -> Dict:
        images_tensor = torch.stack(images)
        model_output = self.hagrid_model(images_tensor)
        model_output = {"labels": model_output}
        if targets is None:
            return model_output
        else:
            targets_tensor = torch.stack([target["labels"] for target in targets])
            return self.criterion(model_output["lables"], targets_tensor) 
    def criterion(self, model_output : Dict, targets: DictConfig):
        loss_val = self.criterion(model_output, targets)
        return loss_val