# app/inpainting.py

import torch
import numpy as np
import cv2
from .gan_inpainter import PConvUNet

def rough_inpaint_opencv(img_patch: np.ndarray, mask_patch: np.ndarray) -> np.ndarray:
    """
    Quick rough fill using OpenCV Telea. Returns the patch with holes roughly filled.
    img_patch: HxWx3 uint8 RGB with hole region zeroed out or original.
    mask_patch: HxW boolean mask where True=hole.
    """
    # Convert mask to 8-bit: 255 for hole
    inpaint_mask = (mask_patch.astype(np.uint8)) * 255
    # Dilate mask slightly to cover edges
    kernel = np.ones((3,3), np.uint8)
    inpaint_mask = cv2.dilate(inpaint_mask, kernel, iterations=1)
    # OpenCV expects BGR
    bgr = cv2.cvtColor(img_patch, cv2.COLOR_RGB2BGR)
    inpainted_bgr = cv2.inpaint(bgr, inpaint_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    inpainted_rgb = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)
    return inpainted_rgb

def get_mask_bbox(mask: np.ndarray):
    """
    Given a boolean mask HxW, return bounding box [y1,y2,x1,x2] inclusive of all True pixels.
    If no True, return None.
    """
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None
    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()
    return y1, y2, x1, x2

def erase_and_inpaint(
    img_np: np.ndarray,
    erase_mask: np.ndarray,
    generator: PConvUNet,
    device=None,
    model_size=(256, 256),
    margin_ratio=0.3,
    use_rough_prefill=True,
    feather_amount=15
) -> np.ndarray:
    """
    Improved inpainting that crops around holes, optionally does rough OpenCV prefill,
    then uses GAN to refine, and pastes back with feathered blending.

    Args:
        img_np:      H0 x W0 x 3 uint8 RGB full image.
        erase_mask:  H0 x W0 boolean or 0/1 mask where True/1 indicates hole to fill.
        generator:   loaded PConvUNet in eval mode.
        device:      torch.device or None (inferred from generator if None).
        model_size:  (Hm, Wm) resolution the model was trained on, e.g. (256,256).
        margin_ratio: fraction of bbox size to pad around hole for context (e.g. 0.3).
        use_rough_prefill: if True, perform a quick OpenCV Telea inpaint on the crop before GAN.
        feather_amount: pixel radius for alpha-blending around pasted patch to reduce seams.
    Returns:
        result_full: H0 x W0 x 3 uint8 RGB inpainted full image.
    """
    if device is None:
        device = next(generator.parameters()).device

    H0, W0 = erase_mask.shape[:2]
    Hm, Wm = model_size

    # Ensure erase_mask boolean
    erase_mask_bool = erase_mask.astype(bool)

    # If no hole, just return original
    if not erase_mask_bool.any():
        return img_np.copy()

    # 1) Compute bounding box of mask
    bbox = get_mask_bbox(erase_mask_bool)
    if bbox is None:
        return img_np.copy()
    y1, y2, x1, x2 = bbox
    h_box = y2 - y1 + 1
    w_box = x2 - x1 + 1

    # 2) Expand bbox by margin_ratio for context
    pad_h = int(h_box * margin_ratio)
    pad_w = int(w_box * margin_ratio)
    y1e = max(0, y1 - pad_h)
    y2e = min(H0-1, y2 + pad_h)
    x1e = max(0, x1 - pad_w)
    x2e = min(W0-1, x2 + pad_w)

    # Crop region from original image and mask
    img_crop = img_np[y1e:y2e+1, x1e:x2e+1].copy()   # shape Ch x Cw x 3
    mask_crop = erase_mask_bool[y1e:y2e+1, x1e:x2e+1].copy()  # Ch x Cw bool

    # 3) Zero-out holes in the crop
    img_crop_zero = img_crop.copy()
    img_crop_zero[mask_crop] = 0

    # 4) Optional rough prefill to give initial structure
    if use_rough_prefill:
        img_prefill = rough_inpaint_opencv(img_crop, mask_crop)
    else:
        img_prefill = img_crop_zero

    # 5) Resize patch and mask to model_size
    #    For image: linear interpolation; for mask: nearest neighbor
    img_rs = cv2.resize(img_prefill, (Wm, Hm), interpolation=cv2.INTER_LINEAR)
    mask_rs = cv2.resize(mask_crop.astype(np.uint8), (Wm, Hm), interpolation=cv2.INTER_NEAREST).astype(bool)

    # 6) Normalize and tensorize for GAN
    #    Assuming generator was trained with [-1,1] normalization and tanh output
    img_norm = img_rs.astype(np.float32) / 127.5 - 1.0  # [0,255] -> [-1,1]
    img_t = torch.from_numpy(img_norm).permute(2,0,1).unsqueeze(0).to(device)  # 1x3xHm xWm
    # Prepare keep-mask: 1=keep, 0=hole
    keep_mask = (~mask_rs).astype(np.float32)
    mask_t = torch.from_numpy(keep_mask).unsqueeze(0).unsqueeze(0).to(device)  # 1x1xHm xWm

    # 7) Run generator
    with torch.no_grad():
        out_t, _ = generator(img_t, mask_t)  # out_t in [-1,1]
    # Map out to [0,1]
    out01 = (out_t + 1.0) / 2.0   # 1x3xHm xWm
    img01 = (img_t + 1.0) / 2.0   # 1x3xHm xWm
    comp_rs = out01 * (1 - mask_t) + img01 * mask_t  # 1x3xHm xWm

    # 8) Convert composite patch back to uint8 Hm xWm x3
    comp_np = comp_rs.squeeze(0).permute(1,2,0).cpu().numpy()  # [0,1]
    comp_uint8 = (comp_np * 255.0).clip(0,255).astype(np.uint8)

    # 9) Resize inpainted patch back to original crop size
    Ch, Cw = img_crop.shape[:2]
    comp_back = cv2.resize(comp_uint8, (Cw, Ch), interpolation=cv2.INTER_LINEAR)

    # 10) Paste back into full image with feathered blending
    result_full = img_np.copy()

    # Create an alpha mask for blending: 1 inside hole, 0 outside, with feather at boundary
    alpha = np.zeros((Ch, Cw), dtype=np.float32)
    # hole region: mask_crop True -> alpha=1
    alpha[mask_crop] = 1.0
    # distance transform for feather: distance to nearest zero in mask_crop
    # We want to fade alpha from 1 at center of hole to 0 at boundary over feather_amount pixels
    if feather_amount > 0:
        # distance inside hole
        dist_inside = cv2.distanceTransform((mask_crop.astype(np.uint8)*255), cv2.DIST_L2, 5)
        # Clip distances above feather_amount
        dist_inside = np.clip(dist_inside, 0, feather_amount)
        # Normalize: at hole center dist large -> alpha near 1; near boundary dist small -> alpha near 0
        # Actually we want alpha = dist_inside/feather_amount
        alpha = dist_inside.astype(np.float32) / float(feather_amount)
        # But ensure outside hole is zero
        alpha[~mask_crop] = 0.0

    # Convert alpha to 3-channel
    alpha_3 = np.stack([alpha]*3, axis=2)  # Ch x Cw x 3

    # Blend: result = comp_back * alpha + original_crop * (1-alpha)
    orig_crop = img_crop.astype(np.float32)
    comp_crop_f = comp_back.astype(np.float32)
    blended = (comp_crop_f * alpha_3 + orig_crop * (1 - alpha_3)).astype(np.uint8)

    # Place blended patch back
    result_full[y1e:y2e+1, x1e:x2e+1] = blended

    return result_full
