import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")

MODEL_URI = "models:/lipreading-grid@champion"
OUTPUT_DIR = "model_export"

path = mlflow.artifacts.download_artifacts(artifact_uri=MODEL_URI, dst_path=OUTPUT_DIR)
print("Model saved to", path)