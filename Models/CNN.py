import torch
import torch.nn as nn


class ConvNet(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_size_1=16,
        hidden_size_2=32,
        hidden_size_3=64,
        num_classes=1
    ):
        super(ConvNet, self).__init__()

        cnn_input_channels, input_height, input_width = input_dim

        self.layer1 = nn.Sequential(
            nn.Conv2d(
                cnn_input_channels, hidden_size_1, kernel_size=2, stride=1, padding=1
            ), 
            nn.BatchNorm2d(hidden_size_1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(hidden_size_1, hidden_size_2, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(hidden_size_2),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(hidden_size_2, hidden_size_3, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(hidden_size_3),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.2),
        )

        self.cnn = nn.Sequential(self.layer1, self.layer2, self.layer3)

        # Use a test input to calculate the flatten size for the linear layer 
        with torch.no_grad():
            test_input = torch.zeros(1, *input_dim)
            out = self.layer1(test_input)
            out = self.layer2(out)
            out = self.layer3(out)
            flatten_dim = out.view(1, -1).shape[1]

        self.fc = nn.Linear(flatten_dim, num_classes)

    def forward(self, x):
        out =  self.cnn(x)
        out = out.reshape(out.size(0), -1)  # flatten everything except the batch
        out = self.fc(out)
        return out
