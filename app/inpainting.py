# app/inpainting.py

import torch
import numpy as np
import cv2
from .gan_inpainter import PConvUNet

def erase_and_inpaint(
    img_np: np.ndarray,
    erase_mask: np.ndarray,
    generator: PConvUNet,
    device=None,
    model_size=(256, 256)
) -> np.ndarray:
    """
    Inpaint an image using a PConvUNet generator with resizing and correct normalization.

    Args:
        img_np:      H0 x W0 x 3 uint8 RGB input image.
        erase_mask:  H0 x W0 boolean or {0,1} mask where True (or 1) = pixels to erase/fill.
        generator:   loaded PConvUNet model (in eval() mode) on appropriate device.
        device:      torch.device or None (inferred from generator if None).
        model_size:  tuple (Hm, Wm) resolution the model was trained on (e.g., (256,256)).

    Returns:
        result_full: H0 x W0 x 3 uint8 RGB inpainted image.
    """
    if device is None:
        device = next(generator.parameters()).device

    # Original size
    H0, W0 = erase_mask.shape[:2]
    Hm, Wm = model_size

    # Ensure erase_mask is boolean
    erase_mask_bool = erase_mask.astype(bool)

    # 1) Resize image and mask to model_size
    #   - image: linear interpolation
    #   - mask: nearest-neighbor to preserve binary
    img_rs = cv2.resize(img_np, (Wm, Hm), interpolation=cv2.INTER_LINEAR)
    mask_rs = cv2.resize(erase_mask_bool.astype(np.uint8), (Wm, Hm),
                         interpolation=cv2.INTER_NEAREST).astype(bool)

    # 2) Zero-out hole region in resized image
    img_rs_work = img_rs.copy()
    img_rs_work[mask_rs] = 0

    # 3) Normalize to [-1,1] if model was trained that way (tanh output):
    #    Adjust here if your training used a different normalization.
    img_norm = img_rs_work.astype(np.float32) / 127.5 - 1.0  # [0,255] -> [-1,1]
    img_t = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).to(device)  # 1×3×Hm×Wm

    # 4) Prepare keep-mask tensor: 1=keep, 0=hole
    keep_mask = (~mask_rs).astype(np.float32)
    mask_t = torch.from_numpy(keep_mask).unsqueeze(0).unsqueeze(0).to(device)  # 1×1×Hm×Wm

    # 5) Run generator
    with torch.no_grad():
        out_t, _ = generator(img_t, mask_t)  # out_t in [-1,1] if tanh
    # Map output from [-1,1] to [0,1]
    out01 = (out_t + 1.0) / 2.0  # 1×3×Hm×Wm
    # Original resized normalized to [0,1]
    img01 = (img_t + 1.0) / 2.0   # 1×3×Hm×Wm

    # 6) Composite: use original outside hole
    comp_rs = out01 * (1 - mask_t) + img01 * mask_t  # 1×3×Hm×Wm

    # 7) Convert composite to uint8 Hm×Wm×3
    comp_np = comp_rs.squeeze(0).permute(1, 2, 0).cpu().numpy()  # [0,1]
    comp_uint8 = (comp_np * 255.0).clip(0, 255).astype(np.uint8)

    # 8) Resize back to original size (W0, H0)
    result_full = cv2.resize(comp_uint8, (W0, H0), interpolation=cv2.INTER_LINEAR)

    return result_full
