import torch
import torch.nn as nn


class ConvNet(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_size_1=16,
        hidden_size_2=32,
        hidden_size_3=64,
        hidden_size_4=128,
    ):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(
                input_dim[0], hidden_size_1, kernel_size=2, stride=1, padding=1
            ),  # Convolutional layer
            nn.BatchNorm2d(hidden_size_1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(hidden_size_1, hidden_size_2, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(hidden_size_2),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(hidden_size_2, hidden_size_3, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(hidden_size_3),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3),
        )
        # TO MODIFY BASED ON THE INPUT SIZE
        # Use a dummy input to calculate the flatten size
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_dim)
            out = self.layer1(dummy_input)
            out = self.layer2(out)
            out = self.layer3(out)
            flatten_dim = out.view(1, -1).shape[1]

        self.fc = nn.Linear(flatten_dim, 1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)  # flatten everything except the batch
        out = self.fc(out)
        return out
