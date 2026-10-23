import torch.nn as nn

from src.data import CHARACTERS

NUM_CLASSES = len(CHARACTERS)


class LipReadingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(3, 5, 5), padding=(1, 2, 2)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Dropout3d(0.3),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=(3, 5, 5), padding=(1, 2, 2)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Dropout3d(0.3),
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv3d(64, 96, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Dropout3d(0.3),
        )
        self.gru = nn.GRU(
            input_size=96 * 5 * 10,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )
        self.classifier = nn.Linear(256 * 2, NUM_CLASSES)

    def forward(self, videos):
        x = self.conv_block1(videos)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x, _ = self.gru(x)
        x = self.classifier(x)
        return x.log_softmax(dim=2)