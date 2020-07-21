import torch
import torch.nn as nn
import torch.nn.functional as F


class PNN(nn.Module):
    def __init__(self, input_length=1600, window_size=5):
        super(PNN, self).__init__()
        self.input_length = input_length
        self.window_size = window_size
        self.embed = nn.Embedding(257, 8, padding_idx=0)

        self.conv_1 = nn.Conv1d(8, 128, window_size, stride=window_size)
        self.conv_2 = nn.Conv1d(8, 128, window_size, stride=window_size)

        self.pooling = nn.MaxPool1d(int(input_length / window_size))

        self.fc_1 = nn.Linear(128, 128)
        self.fc_2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.embed(x)
        x = torch.transpose(x, -1, -2)
        cnn_value = self.conv_1(x)
        gating_weight = torch.sigmoid(self.conv_2(x))

        x = cnn_value * gating_weight
        x = nn.MaxPool1d(self.input_length // self.window_size)(x)
        x = F.relu(x.view(-1, 128))
        x = F.relu(self.fc_1(x))
        x = self.fc_2(x)

        return torch.sigmoid(x)


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce

    def forward(self, inputs, targets):
        loss = F.binary_cross_entropy(inputs, targets)
        pt = torch.exp(-loss)
        F_loss = self.alpha * (1 - pt)**self.gamma * loss
        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss