import jiwer
import torch
import torch.nn as nn

from src.data import CHARACTERS, numbers_to_text

ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)


def greedy_decode(outputs):
    best_numbers = outputs.argmax(dim=2)
    texts = []
    for sequence in best_numbers:
        chars = []
        previous = None
        for number in sequence.tolist():
            if number != previous and number != 0:
                chars.append(CHARACTERS[number])
            previous = number
        texts.append("".join(chars))
    return texts


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    predictions = []
    targets = []
    with torch.no_grad():
        for videos, padded_labels, label_lengths in loader:
            videos = videos.to(device)
            padded_labels = padded_labels.to(device)
            label_lengths = label_lengths.to(device)
            outputs = model(videos)
            log_probs = outputs.permute(1, 0, 2)
            input_lengths = torch.full((videos.shape[0],), log_probs.shape[0], dtype=torch.long, device=device)
            loss = ctc_loss(log_probs, padded_labels, input_lengths, label_lengths)
            total_loss += loss.item()
            predictions.extend(greedy_decode(outputs))
            for label, length in zip(padded_labels, label_lengths):
                targets.append(numbers_to_text(label[:length].tolist()))
    model.train()
    return total_loss / len(loader), predictions, targets


def compute_metrics(predictions, targets):
    wer = jiwer.wer(targets, predictions)
    cer = jiwer.cer(targets, predictions)
    correct_words = 0
    total_words = 0
    for prediction, target in zip(predictions, targets):
        predicted_words = prediction.split()
        target_words = target.split()
        for position in range(len(target_words)):
            total_words += 1
            if position < len(predicted_words):
                if predicted_words[position] == target_words[position]:
                    correct_words += 1
    return wer, cer, correct_words / total_words