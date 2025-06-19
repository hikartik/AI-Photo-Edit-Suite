# AI‑Photo‑Edit Service

Erase **any object class** from an image with a single API call.

*Powered by ****YOLOv8‑Seg**** masks + a custom‑trained ****GAN‑U‑Net inpainting network with Local Context convolutions**** and wrapped in FastAPI.*

---

## Why we built it 🚀

Manual object removal is time‑consuming for photographers, e‑commerce catalog teams, and content creators. We wanted a **drop‑in microservice** that does the heavy lifting automatically, scales in the cloud, and is trivial to integrate into any pipeline or product.

## Key Features

- **Pixel‑accurate masks** from `yolov8n‑seg.pt` (tiny, fast, CUDA‑ready).
- **Seamless fill** by our own GAN‑based U‑Net (implemented & trained from scratch for production inference).
- **FastAPI** + **Uvicorn** server with async I/O.
- **One‑click Docker** deployment – run locally or on ECS/Fargate, K8s, Fly.io, Render, etc.
- Clean, minimal REST interface (single `/erase` endpoint).

## Quick Start (Docker)

```bash
# 1. Pull the pre‑built image
$ docker pull hiikartik/ai-photo-edit-service:latest

# 2. Run it (CPU)
$ docker run -p 8000:8000 hiikartik/ai-photo-edit-service:latest

# 2b. Run with GPU (CUDA 11.8)
$ docker run --gpus all -e CUDA_VISIBLE_DEVICES=0 -p 8000:8000 \
           hiikartik/ai-photo-edit-service:latest

# 3. Test
$ curl -X POST "http://localhost:8000/erase" \
       -F "image=@/path/to/input.jpg" \
       -F "prompt=person" \
       --output output.jpg
```

The service is now live on [http://localhost:8000](http://localhost:8000). Visit `/docs` for the Swagger UI.

### Environment Variables

| Variable          | Default   | Description                                           |
| ----------------- | --------- | ----------------------------------------------------- |
| `MODEL_DIR`       | `/models` | Folder containing `yolov8n-seg.pt` and `gan_unet.pth` |
| `ALLOWED_ORIGINS` | `*`       | CORS origins (comma‑separated)                        |
| `MAX_RESOLUTION`  | `2048`    | Safeguard against huge uploads                        |

---

## API Reference

### `POST /erase`

**Form‑Data Fields**

| Field    | Type            | Required | Example   |
| -------- | --------------- | -------- | --------- |
| `image`  | File (JPEG/PNG) | ✔︎       | `cat.jpg` |
| `prompt` | String          | ✔︎       | `cat`     |

**Response** `image/jpeg` – edited image bytes. Errors follow RFC 7807 problem+json.

---

## Running from Source

```bash
# Clone
$ git clone https://github.com/hikartik/AI-Photo-Edit-Suite.git
$ cd ai-photo-edit-service

# Install deps (Torch 2.3, Ultralytics v8, FastAPI, etc.)
$ python -m venv .venv && source .venv/bin/activate
$ pip install -r requirements.txt

# Launch
$ uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Project Structure

```
.
├── app/               # FastAPI source
│   ├── main.py        # entrypoint
│   ├── yolo_detector.py
│   ├── inpainting.py  # GAN‑U‑Net
│   ├── mask_generator.py
│   ├── utils.py
│   └── models/        # Pydantic DTOs
├── docker/            # Dockerfile & compose
└── kaggle_notebooks/  # training notebooks & weights
```

---

## Performance

| Image Size | Mean Latency (CPU) | Mean Latency (GPU) |
| ---------- | ------------------ | ------------------ |
| 512×512    | 0.9 s              | 120 ms             |
| 1024×1024  | 2.4 s              | 310 ms             |

*Benchmarked on AMD Ryzen 7 4800H (CPU) and RTX 3060‑Laptop GPU.*

---

## Use‑Cases 💡

- E‑commerce background cleanup (remove mannequins, stands, clutter).
- Photo editors / mobile apps (one‑tap object eraser).
- Social media platforms (auto‑moderation & content redaction).
- Dataset preprocessing for computer vision research.

---

## License

MIT © 2025 Kartik Kumar

