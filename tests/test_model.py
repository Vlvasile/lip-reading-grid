import torch

from src.inference import resample_frames
from src.model import LipReadingModel, NUM_CLASSES


def test_model_output_shape():
    model = LipReadingModel()
    video = torch.randn(1, 1, 75, 40, 80)

    output = model(video)

    assert output.shape == (1, 75, NUM_CLASSES)


def test_resample_more_frames():
    frames = list(range(90))
    result = resample_frames(frames, 75)
    assert len(result) == 75


def test_resample_fewer_frames():
    frames = list(range(50))
    result = resample_frames(frames, 75)
    assert len(result) == 75