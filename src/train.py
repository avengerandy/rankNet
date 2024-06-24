import json
import pickle
import torch
from torch import optim
from torch.utils.data import DataLoader
from dataset import *
from model import *

BATCH_SIZE = 64
EPOCH = 20

def trainTestWithSameRankingPage(printResult = True, saveModel = False):
    with open('./dataset/asmrAllItemDict.json', 'r') as infile:
        itemDict = json.load(infile)
    with open('./dataset/asmrAllRankItem.json', 'r') as infile:
        rankItem = json.load(infile)

    rankItem, testRankItem = orderedTrainTestSplit(rankItem, 0.1)
    postivePairsDataset, minMaxScaler = getNormalizedDataset(rankItem, itemDict)
    dataloader = DataLoader(postivePairsDataset, batch_size = BATCH_SIZE, shuffle = True)

    model = RankNet().to(getDevice())
    lossFunction = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.01, weight_decay = 0)
    epoch = EPOCH
    for i in range(epoch):
        trainCurrentRate = train(dataloader, model, lossFunction, optimizer)
        if (printResult):
            print(f'epoch{i} trainCurrentRate: {trainCurrentRate:.2f}')

    postivePairsDataset, _ = getNormalizedDataset(testRankItem, itemDict, minMaxScaler)
    dataloader = DataLoader(postivePairsDataset, batch_size = BATCH_SIZE, shuffle = False)
    testCurrentRate = test(dataloader, model)
    if (printResult):
        print(f'testCurrentRate: {testCurrentRate:.2f}')
    if (saveModel):
        torch.save(model.state_dict(), './checkpoint/sameRankingPageStateDict')
        with open('./checkpoint/sameRankingPageMinMaxScaler.pkl', 'wb') as file:
            pickle.dump(minMaxScaler, file) 
    return trainCurrentRate, testCurrentRate

def trainTestWithOtherRankingPage(learningRate = 0.001, l1Weight = 0.001, l2Weight = 0.001, shareScaler = False, printResult = True, saveModel = False):
    with open('./dataset/asmrAllItemDict.json', 'r') as infile:
        itemDict = json.load(infile)
    with open('./dataset/asmrAllRankItem.json', 'r') as infile:
        rankItem = json.load(infile)

    postivePairsDataset, minMaxScaler = getNormalizedDataset(rankItem, itemDict)
    dataloader = DataLoader(postivePairsDataset, batch_size = BATCH_SIZE, shuffle = True)

    model = RankNet().to(getDevice())
    lossFunction = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr = learningRate, weight_decay = l2Weight)
    epoch = EPOCH
    for i in range(epoch):
        trainCurrentRate = train(dataloader, model, lossFunction, optimizer, l1Weight)
        if (printResult):
            print(f'epoch{i} trainCurrentRate: {trainCurrentRate:.2f}')

    with open('./dataset/gameAllItemDict.json', 'r') as infile:
        itemDict = json.load(infile)
    with open('./dataset/gameAllRankItem.json', 'r') as infile:
        rankItem = json.load(infile)

    if (shareScaler):
        postivePairsDataset, minMaxScaler = getNormalizedDataset(rankItem, itemDict, minMaxScaler)
    else:
        postivePairsDataset, minMaxScaler = getNormalizedDataset(rankItem, itemDict)
    dataloader = DataLoader(postivePairsDataset, batch_size = BATCH_SIZE, shuffle = False)
    testCurrentRate = test(dataloader, model)
    if (printResult):
        print(f'testCurrentRate: {testCurrentRate:.2f}')
    if (saveModel):
        torch.save(model.state_dict(), './checkpoint/otherRankingPageStateDict')
        with open('./checkpoint/otherRankingPageMinMaxScaler.pkl', 'wb') as file:
            pickle.dump(minMaxScaler, file)

    return trainCurrentRate, testCurrentRate

if __name__ == '__main__':
    print('trainTestWithSameRankingPage')
    trainTestWithSameRankingPage()
    print('=============================')
    print('trainTestWithOtherRankingPage (original)')
    trainTestWithOtherRankingPage(learningRate = 0.01, l1Weight = 0, l2Weight = 0, shareScaler = True)
    print('=============================')
    print('trainTestWithOtherRankingPage (modified)')
    trainTestWithOtherRankingPage()
