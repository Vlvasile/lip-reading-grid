import argparse
from pathlib import Path

import mlflow
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import GridDataset, collate_batch
from src.model import LipReadingModel
from src.evaluation import ctc_loss, evaluate, compute_metrics

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed" / "grid"
MODELS_DIR = PROJECT_DIR / "models"
MLFLOW_DB = PROJECT_DIR / "mlflow.db"

def make_loaders(batch_size):
    labels = pd.read_csv(DATA_DIR / "labels_split.csv")

    train_dataset = GridDataset(labels[labels["split"] == "train"], DATA_DIR)
    val_dataset = GridDataset(labels[labels["split"] == "val"], DATA_DIR)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    return train_loader, val_loader

def compute_loss(model, videos, padded_labels, label_lengths, device):
    videos = videos.to(device)
    padded_labels = padded_labels.to(device)
    label_lengths = label_lengths.to(device)

    outputs = model(videos)
    log_probs = outputs.permute(1, 0, 2)
    input_lengths = torch.full((videos.shape[0],), log_probs.shape[0], dtype=torch.long, device=device)

    return ctc_loss(log_probs, padded_labels, input_lengths, label_lengths)


def train_one_epoch(model, loader, optimizer, device, epoch):
    model.train()
    total_loss = 0

    for videos, padded_labels, label_lengths in tqdm(loader, desc="Epoch " + str(epoch)):
        optimizer.zero_grad()
        loss = compute_loss(model, videos, padded_labels, label_lengths, device)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def train(epochs, learning_rate, batch_size, patience, run_name, resume_from):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_loader, val_loader = make_loaders(batch_size)

    model = LipReadingModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    start_epoch = 1

    if resume_from is not None:
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        print("Resumed from", resume_from, "at epoch", checkpoint["epoch"])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    MODELS_DIR.mkdir(exist_ok=True)
    checkpoint_path = MODELS_DIR / (run_name + ".pt")
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    mlflow.set_tracking_uri("sqlite:///" + MLFLOW_DB.as_posix())
    mlflow.set_experiment("lipnet-grid")

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "epochs": epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "patience": patience,
            "resume_from": str(resume_from),
        })

        for epoch in range(start_epoch, start_epoch + epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)

            val_loss, predictions, targets = evaluate(model, val_loader, device)
            wer, cer, word_accuracy = compute_metrics(predictions, targets)

            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_wer": wer,
                "val_cer": cer,
                "val_word_accuracy": word_accuracy,
                "learning_rate": current_lr,
            }, step=epoch)

            print("Epoch", epoch, "val loss:", round(val_loss, 3), "WER:", round(wer, 3),
                  "word acc:", round(word_accuracy, 3))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                checkpoint = {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                torch.save(checkpoint, checkpoint_path)
                print("   saved best checkpoint")
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print("Early stopping")
                break

        mlflow.log_metric("best_val_loss", best_val_loss)

    print("Best checkpoint:", checkpoint_path)
    return checkpoint_path

def parse_args():
    parser = argparse.ArgumentParser(description="Train the lip reading model on GRID")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--run-name", type=str, default="script_run")
    parser.add_argument("--resume-from", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        patience=args.patience,
        run_name=args.run_name,
        resume_from=args.resume_from,
    )