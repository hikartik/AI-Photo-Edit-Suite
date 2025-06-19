import os
import sys
from typing import Tuple

import cv2
import numpy as np
import torch
from skimage.feature import canny

# Add EdgeConnect src to path
EC_SRC_PATH = os.path.join(os.path.dirname(__file__), '..', 'external', 'edge-connect', 'src')
if EC_SRC_PATH not in sys.path:
    sys.path.append(EC_SRC_PATH)

from config import Config  # type: ignore
from models import EdgeModel, InpaintingModel  # type: ignore


def load_edgeconnect(checkpoints_dir: str, device=None) -> Tuple[EdgeModel, InpaintingModel, Config]:
    """Load EdgeConnect edge and inpaint models from a checkpoints directory."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config_path = os.path.join(checkpoints_dir, 'config.yml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.yml not found under {checkpoints_dir}")

    config = Config(config_path)
    config.MODE = 2       # test mode
    config.MODEL = 3      # edge + inpaint
    config.DEVICE = device
    config.PATH = checkpoints_dir

    edge_model = EdgeModel(config).to(device)
    inpaint_model = InpaintingModel(config).to(device)
    edge_model.load()
    inpaint_model.load()
    edge_model.eval()
    inpaint_model.eval()
    return edge_model, inpaint_model, config


def edgeconnect_inpaint(
    image: np.ndarray,
    mask: np.ndarray,
    edge_model: EdgeModel,
    inpaint_model: InpaintingModel,
    config: Config,
    device=None
) -> np.ndarray:
    """Inpaint a single image using a loaded EdgeConnect model."""
    if device is None:
        device = config.DEVICE

    h, w = image.shape[:2]
    size = config.INPUT_SIZE if config.INPUT_SIZE else None

    img = image.astype(np.float32) / 255.0
    m = mask.astype(np.float32)

    if size is not None and (h != size or w != size):
        img_rs = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
        mask_rs = cv2.resize(m, (size, size), interpolation=cv2.INTER_NEAREST)
    else:
        img_rs = img
        mask_rs = m

    gray = cv2.cvtColor((img_rs * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) / 255.0
    edge_init = canny(gray, sigma=config.SIGMA, mask=(1 - mask_rs).astype(bool)).astype(np.float32)

    img_t = torch.from_numpy(img_rs.transpose(2, 0, 1)).unsqueeze(0).to(device)
    gray_t = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0).to(device)
    edge_t = torch.from_numpy(edge_init).unsqueeze(0).unsqueeze(0).to(device)
    mask_t = torch.from_numpy(mask_rs).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_edges = edge_model(gray_t, edge_t, mask_t)
        out = inpaint_model(img_t, pred_edges, mask_t)
        comp = out * mask_t + img_t * (1 - mask_t)

    comp_np = comp.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    if size is not None and (h != size or w != size):
        comp_np = cv2.resize(comp_np, (w, h), interpolation=cv2.INTER_LINEAR)
    comp_np = (comp_np * 255.0).clip(0, 255).astype(np.uint8)
    return comp_np
