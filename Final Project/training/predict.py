# import sys
# sys.path.insert(1, os.path.join(sys.path[0], '..'))
from .utils import get_device


def predict(img_tensor, model, conf, iou, half, augment, model_type, imgsz=640):
    """
    expect img is frame from left camera after undistortion
    """

    if model_type == "yolo":
        device = get_device()
        img_tensor = img_tensor.to(device)
        pred = model.predict(
            img_tensor,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            half=half,
            augment=augment,
            verbose=False,
        )
        return pred[0]
