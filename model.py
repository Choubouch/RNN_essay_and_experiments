
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

class LSTMv1(nn.Module):
    def __init__(self, nLayer, nHidden, nInput):
        super(LSTMv1, self).__init__()
        self.lstm = nn.LSTM(nInput, nHidden, nLayer)
        self.fc = nn.Linear(nHidden, nInput)

    def forward(self, x):
        h, _ = self.lstm(x)
        return self.fc(h)