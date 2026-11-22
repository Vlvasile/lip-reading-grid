import os
from pathlib import Path

import cv2
import mediapipe as mp
import mlflow
import mlflow.pytorch
import numpy as np
import torch

from src.evaluation import greedy_decode
from src.mouth import crop_mouth

PROJECT_DIR = Path(__file__).resolve().parent.parent
MLFLOW_DB = PROJECT_DIR / "mlflow.db"
MODEL_URI = os.environ.get("MODEL_URI", "models:/lipreading-grid@champion")
NUM_FRAMES = 75


def load_model(device):
    if MODEL_URI.startswith("models:/"):
        mlflow.set_tracking_uri("sqlite:///" + MLFLOW_DB.as_posix())

    model = mlflow.pytorch.load_model(MODEL_URI, map_location=device)
    model.eval()
    model.gru.flatten_parameters()
    return model


def resample_frames(frames, count):
    positions = np.linspace(0, len(frames) - 1, count)
    result = []
    for position in positions:
        result.append(frames[int(round(position))])
    return result


def video_to_mouth_frames(video_path):
    capture = cv2.VideoCapture(str(video_path))
    face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

    mouth_frames = []
    while True:
        success, frame = capture.read()
        if not success:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        if results.multi_face_landmarks is None:
            continue

        landmarks = results.multi_face_landmarks[0].landmark
        mouth = crop_mouth(rgb_frame, landmarks)
        if mouth is None:
            continue

        gray_mouth = cv2.cvtColor(mouth, cv2.COLOR_RGB2GRAY)
        mouth_frames.append(gray_mouth)

    capture.release()
    face_mesh.close()
    return mouth_frames


def predict_text(model, mouth_frames, device):
    frames = resample_frames(mouth_frames, NUM_FRAMES)

    video = np.array(frames).astype(np.float32)
    video = (video - video.mean()) / (video.std() + 1e-6)
    video = video[np.newaxis, np.newaxis]
    video_tensor = torch.tensor(video).to(device)

    with torch.no_grad():
        outputs = model(video_tensor)

    return greedy_decode(outputs)[0]