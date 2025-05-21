import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    def __init__(self, embed_dim):
        super(AttentionLayer, self).__init__()
        self.query_layer = nn.Linear(embed_dim, embed_dim)
        self.key_layer = nn.Linear(embed_dim, embed_dim)
        self.value_layer = nn.Linear(embed_dim, embed_dim)
        self.scale_factor = embed_dim**0.5  # Square root of embed dimension for scaling

    def forward(self, x):
        Q = self.query_layer(x)
        K = self.key_layer(x)
        V = self.value_layer(x)

        # Compute the scaled dot-product attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale_factor
        attention_weights = F.softmax(scores, dim=-1)

        # Output weighted sum of values
        attention_output = torch.matmul(attention_weights, V)

        return attention_output, attention_weights


class CNN_LSTM_attention(nn.Module):
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

        # Attention Layer
        self.attention_layer = AttentionLayer(lstm_hidden_size * 2)

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
        attn_out, attn_weights = self.attention_layer(
            lstm_out
        )  # shape: [batch, seq_len, embed_dim]
        attn_out = attn_out.sum(
            dim=1
        )  # aggregate over time or use other pooling if needed
        return self.classifier(attn_out)
