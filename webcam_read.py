import time

import cv2
import mediapipe as mp
import numpy as np
import torch

from src.mouth import crop_mouth, get_mouth_box
from src.model import LipReadingModel
from src.evaluation import greedy_decode

NUM_FRAMES = 75
RECORD_SECONDS = 3.0
CAMERA_INDEX = 0
CHECKPOINT_PATH = "models/run_A_no_augmentation.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LipReadingModel().to(device)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
model.load_state_dict(checkpoint["model"])
model.eval()

def blur_background(frame, landmarks):
    height = frame.shape[0]
    width = frame.shape[1]

    xs = []
    ys = []
    for point in landmarks:
        xs.append(int(point.x * width))
        ys.append(int(point.y * height))

    left = max(min(xs) - 40, 0)
    right = min(max(xs) + 40, width)
    top = max(min(ys) - 60, 0)
    bottom = min(max(ys) + 40, height)

    blurred = cv2.GaussianBlur(frame, (55, 55), 0)
    blurred[top:bottom, left:right] = frame[top:bottom, left:right]
    return blurred

def resample_frames(frames, count):
    positions = np.linspace(0, len(frames) - 1, count)
    result = []
    for position in positions:
        result.append(frames[int(round(position))])
    return result


def predict(mouth_frames):
    video = np.array(mouth_frames).astype(np.float32)
    video = (video - video.mean()) / (video.std() + 1e-6)
    video = video[np.newaxis, np.newaxis]
    video_tensor = torch.tensor(video).to(device)

    with torch.no_grad():
        outputs = model(video_tensor)

    return greedy_decode(outputs)[0]


camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

recording = False
record_start = 0
mouth_frames = []
result_text = "Speak"
latency_text = ""

while True:
    success, frame = camera.read()
    if not success:
        print("Camera not available")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks is not None:
        landmarks = results.multi_face_landmarks[0].landmark
        frame = blur_background(frame, landmarks)

        height = frame.shape[0]
        width = frame.shape[1]
        left, top, right, bottom = get_mouth_box(landmarks, width, height)
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        if recording:
            mouth = crop_mouth(rgb_frame, landmarks)
            if mouth is not None:
                gray_mouth = cv2.cvtColor(mouth, cv2.COLOR_RGB2GRAY)
                mouth_frames.append(gray_mouth)

    if recording:
        elapsed = time.time() - record_start

        if elapsed >= RECORD_SECONDS:
            recording = False
            print("Captured frames:", len(mouth_frames), "in", round(elapsed, 2), "seconds")

            if len(mouth_frames) < 10:
                result_text = "Mouth not found, try again"
            else:
                video_frames = resample_frames(mouth_frames, NUM_FRAMES)

                start_time = time.time()
                result_text = predict(video_frames)
                latency_ms = (time.time() - start_time) * 1000

                latency_text = "Prediction time: " + str(round(latency_ms)) + " ms"
                print("Prediction:", result_text, "|", latency_text)

    cv2.putText(frame, result_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, latency_text, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    if recording:
        seconds_left = RECORD_SECONDS - (time.time() - record_start)
        progress = "REC " + str(round(seconds_left, 1)) + " s"
        cv2.putText(frame, progress, (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Lip reading", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord(" "):
        recording = True
        record_start = time.time()
        mouth_frames = []
        result_text = "Speak"

camera.release()
face_mesh.close()
cv2.destroyAllWindows()