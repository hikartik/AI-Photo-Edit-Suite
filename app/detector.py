# app/detector.py

import numpy as np
from ultralytics import YOLO

# Load once at startup
MODEL = YOLO("yolov8n-seg.pt")

def get_masks(image: np.ndarray, conf=0.25):
    """
    image: HxWx3 uint8 RGB
    Returns: list of (HxW bool mask, class_id)
    """
    results = MODEL.predict(source=image, conf=conf, verbose=False)
    masks   = results[0].masks.data.cpu().numpy()      # [N,H,W]
    classes = results[0].boxes.cls.cpu().numpy().astype(int)
    # convert masks to bool
    return [(m.astype(bool), int(c)) for m, c in zip(masks, classes)]
