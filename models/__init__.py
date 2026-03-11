from models.model import Hagrid
from .classifiers import VitB16, classifierModel

from torchvision import models
from functools import partial

classifiers_list = {
    "MobileNetV3_large": partial(classifierModel, models.mobilenet_v3_large),
    "VitB16": partial(classifierModel, VitB16),
}