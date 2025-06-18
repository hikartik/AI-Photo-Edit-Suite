# app/main.py

import io
import logging
import traceback

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from starlette.responses import StreamingResponse

from .detector import get_masks
from .utils import parse_prompt, merge_masks
from .inpainting import erase_and_inpaint
from .gan_inpainter import load_generator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# If running on CPU, limit threads to avoid oversubscription
torch.set_num_threads(4)

# Load GAN generator at startup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SIZE = (256, 256)  # resolution used in training (height, width)
GEN_WEIGHTS_PATH = "models/inpaint_gan.pth"

try:
    GEN = load_generator(GEN_WEIGHTS_PATH, device=DEVICE)
    logger.info(f"Loaded GAN generator from {GEN_WEIGHTS_PATH} onto {DEVICE}")
except Exception as e:
    logger.error(f"Failed to load generator weights: {e}\n{traceback.format_exc()}")
    GEN = None

@app.get("/healthz", include_in_schema=False)
async def health():
    return {"status":"ok"}

@app.post("/erase/")
async def erase_endpoint(file: UploadFile = File(...), prompt: str = Form(...)):
    if GEN is None:
        raise HTTPException(status_code=500, detail="Generator not loaded.")

    # 1) Read and decode image
    try:
        data = await file.read()
        arr = np.frombuffer(data, np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("cv2.imdecode returned None")
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    except Exception as e:
        logger.error(f"Error reading image: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=400, detail="Invalid image file.")
    H0, W0 = img.shape[:2]
    logger.info(f"Received image with shape: {img.shape}")

    # 2) Build erase mask via YOLO + prompt
    try:
        masks_and_labels = get_masks(img)
        to_remove = parse_prompt(prompt)
        if not to_remove:
            raise ValueError("No valid classes found in prompt.")
        erase_mask = merge_masks(masks_and_labels, to_remove, (H0, W0))
        # Ensure boolean mask
        erase_mask = erase_mask.astype(bool)
        logger.info(f"Erase mask shape: {erase_mask.shape}, unique vals: {np.unique(erase_mask)}")
    except Exception as e:
        logger.error(f"Error creating mask: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=400, detail=f"Mask creation failed: {e}")

    # 3) Inpaint with GAN (with resizing & normalization)
    try:
        result = erase_and_inpaint(img, erase_mask, GEN, device=DEVICE, model_size=MODEL_SIZE)
        # result: H0 x W0 x 3 uint8 RGB
    except Exception as e:
        logger.error(f"Inpainting failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Inpainting failed.")

    # 4) Encode result as PNG and return
    try:
        out_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        success, buf = cv2.imencode(".png", out_bgr)
        if not success:
            raise ValueError("cv2.imencode failed")
    except Exception as e:
        logger.error(f"Error encoding output image: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Failed to encode output image.")
    return StreamingResponse(io.BytesIO(buf.tobytes()), media_type="image/png")
