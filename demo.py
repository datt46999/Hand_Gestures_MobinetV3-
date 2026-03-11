import argparse
import logging
import time
from typing import Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from omegaconf import DictConfig, OmegaConf
from torch import Tensor

from constants import targets
from custom_utils.utils import build_model

logging.basicConfig(format="[LINE:%(lineno)d] %(levelname)-8s [%(asctime)s]  %(message)s", level=logging.INFO)

COLOR = (0, 255, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX
class writeLogg:
    def __init__(self, file):
        self.stdout = sys.stdout
        self.file = file

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)
    def flush(self):
        self.stdout.flush()
        self.file.flush()
class Demo:
    @staticmethod
    def process(image, transform):
        transformed_img = transform(image = image)
        return transformed_img["image"]

    @staticmethod
    def get_transform_for_inf(transform_config):
        transforms_list = [getattr(A, key)(**params) for key, params in transform_config.items()]
        transforms_list.append(ToTensorV2())
        return A.Compose(transforms_list)
    @staticmethod
    def run(classifier, transforms):
        t1 = cnt = 0 
        cap = cv2.VideoCapture(0)
        
        while cap.isOpened():
            delta = time.time() - t1
            t1 = time.time()
            ret, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if not ret:
                break
            process_img = Demo.process(frame, transforms)
            with torch.no_grad():
                output = classifier([process_img])
            label = output["labels"].argmax(1)
            cv2.putText(frame, targets[int(label)], (10, 100), cv2.FONT_HERSHEY_SIMPLEX,2, (0,0,255), thickness =3)
            fps = 1/delta
            cv2.putText(frame, f"FPS: {fps :02.1f} Frame: {cnt}", (30, 30), FONT, 1, (255, 0, 255), 2 )
            cnt+=1
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF in [ord('q')]:
                break
        cap.release()
        cv2.destroyAllWindows()

            
        
    # @staticmethod
    # def show_infor_model(model_path):
    #     state_dict = torch.load(model_path, map_location = torch.device("cpu"))
    #     log_file = "infor_model.log"
    #     with open(log_file, 'w') as f:
    #         sys.stdout = writeLogg(f)
    #         print("=====Infor of model======")
    #         print(model_path)
    #         if isinstance(state_dict, dict):
    #             for key, value in state_dict.items():
                  
    #                 print(f"key: {key}")
    #                 print("value: ")
    #                 # print({tuple(val.shape)})
    #                 # print({value.dtype})
    #                 print()
    #                 print(value)            

                   
    #         else:
    #             print("Checkpoint format not recognized")

    #     sys.stdout = sys.__stdout__
    #     print(f"\nSaved model information to {log_file}")
def parse_arguments(params: Optional[Tuple] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Demo full frame classification...")

    parser.add_argument("-p", "--path_to_config", required=True, type=str, help="Path to config")

    known_args, _ = parser.parse_known_args(params)
    return known_args

if __name__ == "__main__":
    args = parse_arguments()
    conf = OmegaConf.load(args.path_to_config)
    model = build_model(conf)
    transform = Demo.get_transform_for_inf(conf.test_transforms)
    # print(conf.test_transforms)
    if conf.model.checkpoint is not None:
        snapshot = torch.load(conf.model.checkpoint, map_location=torch.device("cpu"))
        model.load_state_dict(snapshot["MODEL_STATE"])
    model.eval()
    if model is not None:
        Demo.run(model, transform)