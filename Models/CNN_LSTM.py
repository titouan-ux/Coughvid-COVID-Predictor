import torch
import torch.nn as nn


class CNN_LSTM(nn.Module):
    def __init__(self, input_dim, lstm_hidden_size=128, lstm_layers=2, num_classes=1):
        super().__init__()

        cnn_input_channels, input_height, input_width = input_dim
        # CNN layers
        self.layer1 = nn.Sequential(
            nn.Conv2d(cnn_input_channels, 16, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.2),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.2),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.2),
        )

        self.cnn = nn.Sequential(self.layer1, self.layer2, self.layer3)

        # Infer feature size from dummy input to define LSTM input dimension
        with torch.no_grad():
            dummy = torch.zeros(1, cnn_input_channels, input_height, input_width)
            feat = self.cnn(dummy)
            _, C, H, W = feat.shape
            self.lstm_input_size = H * C
            self.seq_len = W  # sequence length for LSTM

        # LSTM
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(lstm_hidden_size * 2),
            nn.Dropout(0.2),
            nn.Linear(lstm_hidden_size * 2, 100),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(100, num_classes),
        )

    def forward(self, x):
        feat = self.cnn(x)
        N, C, H, W = feat.shape
        feat = feat.permute(0, 3, 2, 1)  # N x W x H x C
        lstm_in = feat.reshape(N, W, H * C)
        lstm_out, _ = self.lstm(lstm_in)
        last_t = lstm_out[:, -1, :]
        return self.classifier(last_t)
