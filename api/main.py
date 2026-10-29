import tempfile
import time
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile

from src.inference import load_model, predict_text, video_to_mouth_frames

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model(device)

app = FastAPI(title="Lip Reading API")


@app.get("/health")
def health():
    return {"status": "ok", "device": str(device)}


@app.post("/predict")
def predict(video: UploadFile = File(...)):
    start_time = time.time()

    suffix = Path(video.filename).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
        temp_file.write(video.file.read())
        temp_path = temp_file.name

    mouth_frames = video_to_mouth_frames(temp_path)
    Path(temp_path).unlink()

    if len(mouth_frames) < 10:
        raise HTTPException(status_code=422, detail="Mouth not found in the video")

    text = predict_text(model, mouth_frames, device)
    processing_ms = (time.time() - start_time) * 1000

    return {
        "text": text,
        "frames_with_mouth": len(mouth_frames),
        "processing_time_ms": round(processing_ms),
    }