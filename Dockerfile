FROM python:3.9-slim

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
# Copy weights
COPY models/ ./models/
# Copy YOLOv8-seg.pt if you keep locally; else you can let ultralytics download
COPY yolov8n-seg.pt ./app/  # or adjust path in detector.py accordingly

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
