import os
from typing import List, Tuple

import torch
import torch.distributed as dist
from omegaconf import DictConfig
from torch.nn import SyncBatchNorm
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.tensorboard import SummaryWriter

from custom_utils.ddp_utils import get_sampler
from custom_utils.utils import Logger, build_model, get_transform
from models import Hagrid

from .utils import set_random_seed

set_random_seed()

def collate_fn(batch: List) -> Tuple:
    return list(zip(*batch))

def get_dataloader(dataset: Dataset, sampler: Sampler = None, **kwargs) -> DataLoader:
    return DataLoader(
        dataset,
        collate_fn=collate_fn,
        shuffle=kwargs["shuffle"] if sampler is None else False,
        sampler=sampler,
        batch_size=kwargs["batch_size"],
        num_workers=kwargs["num_workers"],
        prefetch_factor=kwargs["prefetch_factor"],
    )



def load_train_objects(config: DictConfig, command: str, n_gpu: int):
    model = build_model(config)
    if model.type == "classifier":
        from dataset import ClassificationDataset as GestureDataset
    else:
        raise Exception(f"Model type {model.type} does not exist")
    text_dataset = GestureDataset(config, "text", get_transform(config.text_transforms, model.type))
    if command == "train":
        train_dataset = GestureDataset(config, "train", get_transform(config.text_transforms, model.type))
        if config.dataset.dataset_val and config.dataset.annotations_val:
            val_dataset = GestureDataset(config, "val", get_transform(config.val_transforms, model.type))
        else:
            raise Exception("Cannot train without validation data")
    train_sampler = None
    test_sampler = None
    val_sampler = None
    if n_gpu > 1:
        test_sampler = get_sampler(test_dataset)
        if command == "train":
            train_sampler = get_sampler(train_dataset)
            if val_dataset:
                val_sampler = get_sampler(val_dataset)

    test_dataloader = get_dataloader(test_dataset, test_sampler, **config.test_params)
    if command == "train":
        train_dataloader = get_dataloader(train_dataset, train_sampler, **config.train_params)
        if val_dataset:
            val_dataloader = get_dataloader(val_dataset, val_sampler, **config.val_params)
    else:
        train_dataloader = None
        val_dataloader = None

    return train_dataloader, val_dataloader, test_dataloader, model

def load_train_optimizer(model: Hagrid, config: DictConfig):
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = getattr(torch.optim, config.optimizer.name)(parameters, **config.optimizer.params)
    if config.scheduler.name:
        scheduler = getattr(torch.optim.lr_scheduler, config.scheduler.name)(optimizer, **config.scheduler.params)
    else:
        scheduler = None
    return optimizer, scheduler


