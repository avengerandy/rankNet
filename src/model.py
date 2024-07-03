import torch
from torch import nn

device = None

def getDevice():
    global device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return device

class RankNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(12, 16)
        self.tanh = nn.Tanh()
        self.linear2 = nn.Linear(16, 16)
        self.linear3 = nn.Linear(16, 1)

    def forward(self, x):
        out = self.linear1(x)
        out = self.tanh(out)
        out = self.linear2(out)
        out = self.tanh(out)
        out = self.linear3(out)
        return out

def train(dataloader, model, lossFunction, optimizer, l1Weight = 0):
    model.train()
    currentCount = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(getDevice()), y.to(getDevice())
        optimizer.zero_grad()

        X1 = X[:, 0]
        X2 = X[:, 1]
        score1 = model(X1)
        score2 = model(X2)
        out = score1 - score2
        loss = lossFunction(out, y)

        l1Parameters = []
        for parameter in model.parameters():
            l1Parameters.append(parameter.view(-1))
        l1Loss = l1Weight * torch.abs(torch.cat(l1Parameters)).sum()
        loss = loss + l1Loss

        loss.backward()
        optimizer.step()

        currentCount = currentCount + (out > 0).sum().item()

    datasetSize = len(dataloader.dataset)
    currentRate = currentCount / datasetSize
    return currentRate

def test(dataloader, model):
    model.eval()
    currentCount = 0
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(getDevice()), y.to(getDevice())
            X1 = X[:, 0]
            X2 = X[:, 1]
            score1 = model(X1)
            score2 = model(X2)
            out = score1 - score2
            currentCount = currentCount + (out > 0).sum().item()

    datasetSize = len(dataloader.dataset)
    currentRate = currentCount / datasetSize
    return currentRate
