from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200


def test_predict_rejects_non_video():
    fake_file = {"video": ("not_a_video.mpg", b"this is not a video", "video/mpeg")}

    response = client.post("/predict", files=fake_file)

    assert response.status_code == 422