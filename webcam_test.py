import cv2
import mediapipe as mp

from src.mouth import crop_mouth, get_mouth_box

camera = cv2.VideoCapture(0)

face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
)

while True:
    success, frame = camera.read()

    if not success:
        print("Camera not available")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks is not None:
        landmarks = results.multi_face_landmarks[0].landmark

        height = frame.shape[0]
        width = frame.shape[1]
        left, top, right, bottom = get_mouth_box(landmarks, width, height)
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        mouth = crop_mouth(rgb_frame, landmarks)

        if mouth is not None:
            gray_mouth = cv2.cvtColor(mouth, cv2.COLOR_RGB2GRAY)
            big_mouth = cv2.resize(gray_mouth, (320, 160), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("Mouth (model input)", big_mouth)

    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
face_mesh.close()
cv2.destroyAllWindows()