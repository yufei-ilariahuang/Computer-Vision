import torch
import yaml
from ultralytics import YOLO


def get_device(mode="yolo"):
    if torch.cuda.is_available():
        if mode == "yolo":
            num_gpus = torch.cuda.device_count()
            if num_gpus > 1:
                return list(range(num_gpus))
            else:
                return 0
        elif mode == "torch":
            return "cuda"
    elif torch.backends.mps.is_available():
        if mode in ["yolo", "torch"]:
            return "mps"
    else:
        if mode in ["yolo", "torch"]:
            return "cpu"


def load_model(path_to_model):
    if path_to_model.endswith(".pt"):
        model = YOLO(path_to_model)
    else:
        raise ValueError("Invalid model file format")

    return model, "yolo"


def load_label_map(path_to_label_map, mode="yolo"):
    if mode == "yolo":
        with open(path_to_label_map, "r") as f:
            label_map = yaml.safe_load(f)
        return label_map["names"]
    else:
        raise ValueError("Invalid label map file format")
