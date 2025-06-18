# Dockerfile for CPU-based FastAPI + YOLOv8-seg + PConv U-Net GAN inference

# 1. Base image
FROM python:3.9-slim

# 2. Install system dependencies needed by OpenCV etc.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 3. Set working directory
WORKDIR /app

# 4. Copy and install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# 5. Copy application code
COPY app/ ./app/

# 6. Copy model weights
COPY models/ ./models/

# 7. Copy YOLO weights if needed for offline use
COPY yolov8n-seg.pt ./app/

# 8. (Optional) Environment variables for weight paths
ENV INPAINT_GEN_PATH=/app/models/inpaint_gan.pth
ENV YOLO_SEG_PATH=/app/app/yolov8n-seg.pt

# 9. Expose port
EXPOSE 8000

# 10. Launch FastAPI with Uvicorn
CMD ["uvicorn", "app.main:app", "--port", "8000"]
