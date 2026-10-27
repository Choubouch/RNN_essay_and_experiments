import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

class LSTMv1(nn.Module):
    def __init__(self, nLayer, nHidden, nInput, dropout):
        super(LSTMv1, self).__init__()
        self.lstm = nn.LSTM(nInput, nHidden, nLayer, dropout=dropout)
        self.fc = nn.Linear(nHidden, nInput)

    def forward(self, x):
        h, _ = self.lstm(x)
        return self.fc(h)

# Implémentation d'un LSTM sans tricher
class LSTMcell(nn.Module):
    #reprise de la notation utilisée dans la doc de Pytorch
    def __init__(self, inputSize, hiddenSize):
        super(LSTMcell, self).__init__()
        h = hiddenSize
        d = inputSize
        self.Wii = nn.Parameter(torch.randn(h, d))
        self.Whi = nn.Parameter(torch.randn(h, h))
        
        self.Wif = nn.Parameter(torch.randn(h, d))
        self.Whf = nn.Parameter(torch.randn(h, h))
        
        self.Wig = nn.Parameter(torch.randn(h, d))
        self.Whg = nn.Parameter(torch.randn(h, h))

        self.Wio = nn.Parameter(torch.randn(h, d))
        self.Who = nn.Parameter(torch.randn(h, h))

        self.bi = nn.Parameter(torch.zeros(h))
        self.bf = nn.Parameter(torch.zeros(h))
        self.bg = nn.Parameter(torch.zeros(h))
        self.bo = nn.Parameter(torch.zeros(h))

    def forward(self, x, h_old, c_old):
        it = torch.sigmoid(self.Wii@x + self.Whi@h_old + self.bi)
        ft = torch.sigmoid(self.Wif@x + self.Whf@h_old + self.bf)
        gt = torch.tanh(self.Wig@x + self.Whg@h_old + self.bg)
        ot = torch.sigmoid(self.Wio@x + self.Who@h_old + self.bo)
        ct = ft*c_old + it*gt
        ht = ot*torch.tanh(ct)
        return ht, ct

class LSTMLayer(nn.Module):
    def __init__(self, inputSize, hiddenSize):
        super(LSTMLayer, self).__init__()
        self.hiddenSize = hiddenSize
        self.cell = LSTMcell(inputSize, hiddenSize)

    def forward(self, x):
        seq_len = x.shape[0]
        h = torch.zeros(self.hiddenSize)
        c = torch.zeros(self.hiddenSize)
        outputs = []
        for t in range(seq_len):
            x_t = x[t]
            h, c = self.cell(x_t, h, c)
            outputs.append(h)
        outputs = torch.stack(outputs)
        return outputs, h, c

class LSTMv2(nn.Module):
    def __init__(self, nLayer, nHidden, nInput):
        super(LSTMv2, self).__init__()
        self.hiddenSize = nHidden
        self.numLayer = nLayer
        self.layers = nn.ModuleList()
        for i in range(nLayer):
            layerInputSize = nHidden
            if i == 0 :
                layerInputSize = nInput
            self.layers.append(LSTMLayer(layerInputSize, nHidden))
        self.fc = nn.Linear(nHidden, nInput)

    def forward(self, x):
        for layer in self.layers:
            out, h, c = layer(x)
            x = out
        logits = self.fc(x)
        return logits