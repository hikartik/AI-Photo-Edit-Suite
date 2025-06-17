# app/utils.py

import numpy as np
import cv2                # needed for resizing
from .detector import MODEL

# Grab class names from YOLO model
CLASS_NAMES = MODEL.names

def parse_prompt(prompt: str):
    text = prompt.lower().strip()
    for w in ("remove", "and", "the", ",", ".", "please"):
        text = text.replace(w, " ")
    text = " ".join(text.split())
    to_remove = set()
    for cls_id, name in CLASS_NAMES.items():
        if name in text:
            to_remove.add(cls_id)
    print(f"[parse_prompt] text={text!r} → to_remove={[CLASS_NAMES[i] for i in to_remove]}")
    return to_remove

def merge_masks(masks_and_labels, to_remove, shape):
    """
    masks_and_labels: list of (mask HxW bool, class_id int)
    to_remove: set of class_id to erase
    shape: (H_orig, W_orig)
    Returns: boolean array HxW where True means pixel belongs to removed classes (hole).
    """
    H, W = shape
    final = np.zeros((H, W), dtype=bool)

    if not masks_and_labels:
        print("[merge_masks] no masks returned")
        return final

    for mask, cls in masks_and_labels:
        if cls not in to_remove:
            continue
        # mask: bool H_pred×W_pred. Resize to original size
        m_uint8 = (mask.astype(np.uint8) * 255)
        m_resized = cv2.resize(m_uint8, (W, H), interpolation=cv2.INTER_NEAREST)
        final |= (m_resized > 0)
    return final
