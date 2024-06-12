import json
import torch
from torch import optim
from torch.utils.data import DataLoader
from dataset import *
from model import *

with open('./dataset/asmrAllItemDict.json', 'r') as infile:
    itemDict = json.load(infile)
with open('./dataset/asmrAllRankItem.json', 'r') as infile:
    rankItem = json.load(infile)

postivePairsDataset, minMaxScaler = getNormalizedDataset(rankItem, itemDict)
dataloader = DataLoader(postivePairsDataset, batch_size = 64, shuffle = True)

model = RankNet().to(getDevice())
lossFunction = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001, weight_decay = 0.001)
l1Weight = 0.001
currentRate1 = train(dataloader, model, lossFunction, optimizer, l1Weight)
currentRate2 = train(dataloader, model, lossFunction, optimizer, l1Weight)
currentRate3 = train(dataloader, model, lossFunction, optimizer, l1Weight)
print(f'trainCurrentRate: {currentRate1:.2f}, {currentRate2:.2f}, {currentRate3:.2f}')

with open('./dataset/gameAllItemDict.json', 'r') as infile:
    itemDict = json.load(infile)
with open('./dataset/gameAllRankItem.json', 'r') as infile:
    rankItem = json.load(infile)

postivePairsDataset, minMaxScaler = getNormalizedDataset(rankItem, itemDict)
dataloader = DataLoader(postivePairsDataset, batch_size = 64, shuffle = False)
currentRate = test(dataloader, model)
print(f'testCurrentRate: {currentRate:.2f}')
