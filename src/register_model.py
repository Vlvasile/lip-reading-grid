import argparse
from pathlib import Path


import numpy as np
import mlflow
import mlflow.pytorch
import torch
from mlflow.tracking import MlflowClient

from src.model import LipReadingModel

PROJECT_DIR = Path(__file__).resolve().parent.parent
MLFLOW_DB = PROJECT_DIR / "mlflow.db"
MODEL_NAME = "lipreading-grid"


def register(checkpoint_path, alias):
    model = LipReadingModel()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.eval()

    mlflow.set_tracking_uri("sqlite:///" + MLFLOW_DB.as_posix())
    mlflow.set_experiment("lipnet-grid")

    with mlflow.start_run(run_name="register-" + Path(checkpoint_path).stem):
        mlflow.log_param("checkpoint", str(checkpoint_path))
        mlflow.log_param("best_epoch", checkpoint["epoch"])

        input_example = np.zeros((1, 1, 75, 40, 80), dtype=np.float32)

        model_info = mlflow.pytorch.log_model(
            model,
            name="model",
            registered_model_name=MODEL_NAME,
            input_example=input_example,
            serialization_format="pickle",
        )

    version = model_info.registered_model_version

    client = MlflowClient()
    client.set_registered_model_alias(MODEL_NAME, alias, version)

    print("Registered", MODEL_NAME, "version", version, "with alias", alias)


def parse_args():
    parser = argparse.ArgumentParser(description="Register a trained checkpoint in MLflow")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--alias", type=str, default="champion")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    register(args.checkpoint, args.alias)