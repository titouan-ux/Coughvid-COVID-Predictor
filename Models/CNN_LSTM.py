import torch
import torch.nn as nn

class CNN_LSTM(nn.Module):
    def __init__(self, cnn_input_channels,
                hidden_size_1=16,
                hidden_size_2=32,
                hidden_size_3=64,
                hidden_size_4=128,
                lstm_hidden_size=128,
                lstm_layers=2,
                num_classes=1):
        super().__init__()

        # CNN
        self.layer1 = nn.Sequential(
            nn.Conv2d(cnn_input_channels, hidden_size_1, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(hidden_size_1), nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3))

        self.layer2 = nn.Sequential(
            nn.Conv2d(hidden_size_1, hidden_size_2, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(hidden_size_2), nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3))

        self.layer3 = nn.Sequential(
            nn.Conv2d(hidden_size_2, hidden_size_3, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(hidden_size_3), nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3))
        self.layer4 = nn.Sequential(
            nn.Conv2d(hidden_size_3, hidden_size_4, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(hidden_size_4), nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3))

        self.cnn = nn.Sequential(self.layer1, self.layer2, self.layer3, self.layer4)

        # LSTM
        self._lstm_built = False
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_layers = lstm_layers

        # Classifier
        self.classifier = nn.Sequential(
            nn.Tanh(),
            nn.BatchNorm1d(lstm_hidden_size * 2),
            nn.Dropout(0.2),
            nn.Linear(lstm_hidden_size * 2 , 100),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(100, num_classes)
        )

    # Lazily initializes the LSTM once the input feature dimension (H * C) is known from the CNN output
    def _build_lstm(self, feat_dim):
        self.lstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=self.lstm_hidden_size,
            num_layers=self.lstm_layers,
            batch_first=True,
            bidirectional=True,
        ).to(next(self.parameters()).device)
        self._lstm_built = True

    # Forward pass through CNN and LSTM with reshpaing
    def forward(self, x): 
        feat = self.cnn(x)
        N, C, H, W = feat.shape
        feat = feat.permute(0, 3, 2, 1)
        lstm_in = feat.reshape(N, W, H * C)

        if not self._lstm_built:
            self._build_lstm(H * C)

        lstm_out, _ = self.lstm(lstm_in)
        last_t = lstm_out[:, -1, :]
        return self.classifier(last_t)
