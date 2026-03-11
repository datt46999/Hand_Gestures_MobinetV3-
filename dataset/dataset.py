import json
import logging
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from constants import IMAGES


class HagridDataset(Dataset):
    def __init__(self, conf: DictConfig, dataset_type: str, transform):
        self.conf = conf
        self.labels = {
            label: num for (label, num) in zip(self.conf.dataset.targets, range(len(self.conf.dataset.targets)))
        }

        self.dataset_type = dataset_type

        subset = self.conf.dataset.get("subset", None) if dataset_type == "train" else -1

        self.path_to_json = os.path.expanduser(self.conf.dataset.get(f"annotations_{dataset_type}"))
        self.path_to_dataset = os.path.expanduser(self.conf.dataset.get(f"dataset_{dataset_type}"))
        self.annotations = self.__read_annotations(subset)

        self.transform = transform
    @staticmethod
    def _load_image(image_path: str):
        image = Image.open(image_path).convert("RGB")

        return image
    @staticmethod
    def __get_files_from_dir(pth: str, extns: Tuple) -> List:
        if not os.path.exists(pth):
            logging.warning(f"Dataset directory doesn't exist {pth}")
            return []
        files = [f for f in os.listdir(pth) if f.endswith(extns)]
        return files
    def __read_annotations(self, subset: int = None) -> pd.DataFrame:
        exists_images = set()
        annotations_all = []

        for target in tqdm(self.conf.dataset.targets, desc=f"Prepare {self.dataset_type} dataset"):
            target_tsv = os.path.join(self.path_to_json, f"{target}.json")
            if os.path.exists(target_tsv):
                with open(target_tsv, "r") as file:
                    json_annotation = json.load(file)

                json_annotation = [
                    {**annotation, "name": f"{name}.jpg"} for name, annotation in json_annotation.items()
                ]
                if subset > 1:
                    json_annotation = json_annotation[:subset]

                annotation = pd.DataFrame(json_annotation)
                annotation["target"] = target
                annotations_all.append(annotation)
                exists_images.update(self.__get_files_from_dir(os.path.join(self.path_to_dataset, target), IMAGES))
            else:
                logging.info(f"Database for {target} not found")

        annotations_all = pd.concat(annotations_all, ignore_index=True)
        annotations_all["exists"] = annotations_all["name"].isin(exists_images)
        return annotations_all[annotations_all["exists"]]
    def __getitem__(self, item):

        raise NotImplementedError

    def __len__(self):

        return self.annotations.shape[0]

class ClassificationDataset(HagridDataset):
    def __init__(self, conf: DictConfig, dataset_type: str, transform):

        super().__init__(conf, dataset_type, transform)
        self.annotations = self.annotations[
            ~self.annotations.apply(lambda x: x["labels"] == ["no_gesture"] and x["target"] != "no_gesture", axis=1)
        ]

        self.dataset_type = dataset_type
    def __getitem__(self, index: int) -> Tuple[Image.Image, Dict]:
        row = self.annotations.iloc[[index]].to_dict("records")[0]

        image_pth = os.path.join(self.path_to_dataset, row["target"], row["name"])

        image = self._load_image(image_pth)

        labels = row["labels"]

        if row["target"] == "no_gesture":
            gesture = "no_gesture"
        else:
            for label in labels:
                if label == "no_gesture":
                    continue
                else:
                    gesture = label
                    break
        try:
            label = {"labels": torch.tensor(self.labels[gesture])}
        except Exception:
            raise f"unknown gesture {gesture}"
        image = np.array(image)
        if self.transform is not None:
            image = self.transform(image=image)["image"]

        return image, label