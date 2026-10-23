import string

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

BLANK = "_"
CHARACTERS = [BLANK] + list(string.ascii_lowercase) + [" "]

char_to_index = {}
for index, char in enumerate(CHARACTERS):
    char_to_index[char] = index


def text_to_numbers(text):
    numbers = []
    for char in text:
        numbers.append(char_to_index[char])
    return numbers


def numbers_to_text(numbers):
    chars = []
    for number in numbers:
        chars.append(CHARACTERS[number])
    return "".join(chars)


class GridDataset(Dataset):
    def __init__(self, table, data_dir):
        self.table = table.reset_index(drop=True)
        self.data_dir = data_dir

    def __len__(self):
        return len(self.table)

    def __getitem__(self, index):
        row = self.table.iloc[index]
        video_path = self.data_dir / row["speaker"] / (row["name"] + ".npy")
        video = np.load(video_path).astype(np.float32)
        video = (video - video.mean()) / (video.std() + 1e-6)
        video = video[np.newaxis]
        label = text_to_numbers(row["sentence"])
        return torch.tensor(video), torch.tensor(label, dtype=torch.long)


def collate_batch(batch):
    videos = []
    label_list = []
    label_lengths = []
    for video, label in batch:
        videos.append(video)
        label_list.append(label)
        label_lengths.append(len(label))
    videos = torch.stack(videos)
    padded_labels = pad_sequence(label_list, batch_first=True, padding_value=0)
    label_lengths = torch.tensor(label_lengths, dtype=torch.long)
    return videos, padded_labels, label_lengths