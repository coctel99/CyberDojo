from enum import StrEnum
from importlib import import_module

from .base import PoseModel


class MODELS(StrEnum):
    RTMW = "rtmw"
    SPINEPOSE = "spinepose"


MODEL_CLASSES = {
    MODELS.RTMW: "models.rtmw.model.RTMWModel",
    MODELS.SPINEPOSE: "models.spinepose.model.SpinePoseModel",
}


def create_model(name: str) -> PoseModel:
    module_path, class_name = MODEL_CLASSES[MODELS(name)].rsplit(".", 1)
    return getattr(import_module(module_path), class_name)()
