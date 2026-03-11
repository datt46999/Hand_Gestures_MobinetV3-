import random
from collections import defaultdict
from time import gmtime, strftime
from typing import Dict

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from omegaconf import DictConfig
from torchmetrics import F1Score

from models import classifiers_list
import os
TORCH_VERSION = torch.__version__
import os
from datetime import datetime


class F1ScoreWithLogging:
    def __init__(self, task, num_classes):
        self.F1_score = F1Score(task = task, num_classes = num_classes )
    def to(self, device):
        self.F1_score = self.F1_score.to(device)
        return self
    def __call__(self, predict, targets):
        target = torch.stack([t["labels"] for t in targets])
        results = self.F1_score(predict["labels"].argmax(1), target)
        return {"F1Score": results}


class Logger:
    def __init__(self, train_state: str, max_epochs: int, dataloader_len: int, log_every: int, gpu_id: int):
        self.dataloader_len = dataloader_len
        self.max_epochs = max_epochs
        self.train_state = train_state
        self.log_every = log_every
        self.gpu_id = gpu_id
        self.loss_averager = LossAverager()
        self.metric_averager = MetricAverager()

    def log_iteration(self, iteration: int, epoch: int, loss: float = None, metrics: dict = None):
        if self.gpu_id != 0:
            return
        if (iteration % self.log_every == 0) or (iteration == self.dataloader_len):
            log_str = f"Time: {strftime('%Y-%m-%d %H:%M:%S', gmtime())} "
            log_str += f"{self.train_state} ---- Epoch [{epoch}/{self.max_epochs}], Iteration [{iteration}/{self.dataloader_len}]:"
            if self.train_state == "Train" and loss is not None:
                self.loss_averager.update(loss)
                log_str += f" Loss: {self.loss_averager.value}"
            if self.train_state in ["Eval", "Test"] and metrics is not None:
                try:
                    del metrics["classes"]
                except KeyError:
                    pass
                self.metric_averager.update(metrics)
                if iteration == self.dataloader_len:
                    for metric_name, metric_value in self.metric_averager.value.items():
                        log_str += f" {metric_name}: {metric_value}"
            print(log_str)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass




class MetricAverager:
    def __init__(self):
        self.current_total = defaultdict(float)
        self.iterations = 0

    def update(self, values: Dict):
        for key, value in values.items():
            self.current_total[key] += value.item()
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            metrics = {key: value / self.iterations for key, value in self.current_total.items()}
            return metrics


class LossAverager:
    def __init__(self):
        self.iterations = 0
        self.current_total = 0

    def update(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return self.current_total / self.iterations


def get_transform(transform_config: DictConfig, model_type: str):
    transforms_list = [getattr(A, key)(**params) for key, params in transform_config.items()]
    transforms_list.append(ToTensorV2())
    return A.Compose(transforms_list)

def build_model(config: DictConfig):
    model_name = config.model.name
    model_config = {"num_classes": 5, "pretrained": config.model.pretrained}
    if model_name in classifiers_list:
        model = classifiers_list[model_name](**model_config)
        model.criterion = getattr(torch.nn, config.criterion)()
        model.type = "classifier"
    else:
        raise Exception(f"Unknown model {model_name}")

    return model
def set_random_seed(seed: int = 42, deterministic: bool = False) -> int:

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        if torch.backends.cudnn.benchmark:
            print(
                "torch.backends.cudnn.benchmark is going to be set as "
                "`False` to cause cuDNN to deterministically select an "
                "algorithm"
            )
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        if TORCH_VERSION >= "1.10.0":
            torch.use_deterministic_algorithms(True)
    return seed
class Tee:
    def __init__(self, file):
        self.file = file
        self.stdout = sys.stdout

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

