import datetime
import itertools
import random
import math
import numpy
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

def orderedTrainTestSplit(rankItem, testRatio):
    trainData = rankItem
    totalSize = len(trainData)
    testDateSize = math.ceil(totalSize * testRatio)
    testDateIndex = list(range(totalSize))
    random.shuffle(testDateIndex)
    testDateIndex = testDateIndex[:testDateSize]
    testDateIndex.sort()

    testDate = []
    for index in testDateIndex:
        testDate.append(trainData[index])
    for item in testDate:
        trainData.remove(item)

    return trainData, testDate

def getNormalizedDataset(rankItem, itemDict, minMaxScaler = None):
    itemDict = {key: preprocessing(value) for key, value in itemDict.items()}
    item2Index, itemFeature = decompositionItemDict(itemDict)
    itemFeature, minMaxScaler = normalize(itemFeature, minMaxScaler)
    postivePairsDataset = PostivePairsDataset(rankItem, item2Index, itemFeature)
    return postivePairsDataset, minMaxScaler

def preprocessing(itemObject):
    inputFeature = []

    if ('dl_count' not in itemObject or itemObject['dl_count'] is None):
        inputFeature.append(0)
    else:
        inputFeature.append(int(itemObject['dl_count']))

    if ('wishlist_count' not in itemObject or itemObject['wishlist_count'] is None):
        inputFeature.append(0)
    else:
        inputFeature.append(int(itemObject['wishlist_count']))

    if ('rate_average_2dp' not in itemObject or itemObject['rate_average_2dp'] is None):
        inputFeature.extend([0.0, 0, 0, 0, 0, 0, 0])
    else:
        inputFeature.append(float(itemObject['rate_average_2dp']))
        inputFeature.append(int(itemObject['rate_count']))
        inputFeature.append(int(itemObject['rate_count_detail'][0]['count']))
        inputFeature.append(int(itemObject['rate_count_detail'][1]['count']))
        inputFeature.append(int(itemObject['rate_count_detail'][2]['count']))
        inputFeature.append(int(itemObject['rate_count_detail'][3]['count']))
        inputFeature.append(int(itemObject['rate_count_detail'][4]['count']))

    if ('review_count' not in itemObject or itemObject['review_count'] is None):
        inputFeature.append(0)
    else:
        inputFeature.append(int(itemObject['review_count']))

    inputFeature.append(int(itemObject['price']))
    inputFeature.append(int(datetime.datetime.strptime(itemObject['regist_date'], "%Y-%m-%d %H:%M:%S").timestamp()))

    return inputFeature

def decompositionItemDict(itemDict):
    itemFeature = []
    item2Index = {}
    counter = 0
    for key, value in itemDict.items():
        itemFeature.append(value)
        item2Index[key] = counter
        counter = counter + 1
    return item2Index, itemFeature

def normalize(itemFeature, minMaxScaler = None):
    if minMaxScaler is None:
        minMaxScaler = MinMaxScaler()
        minMaxScaler.fit(itemFeature)
    else:
        minMaxScaler = minMaxScaler
    itemFeature = minMaxScaler.transform(itemFeature)
    return itemFeature, minMaxScaler

class PostivePairsDataset(Dataset):
    def __init__(self, rankItem, item2Index, itemFeature):
        self.item2Index = item2Index
        self.itemFeature = numpy.array(itemFeature)
        self.postivePairs = order2postivePairs(rankItem)

    def __len__(self):
        return len(self.postivePairs)

    def __getitem__(self, idx):
        postivePair = torch.tensor(numpy.stack((
            self.itemFeature[self.item2Index[self.postivePairs[idx][0]]],
            self.itemFeature[self.item2Index[self.postivePairs[idx][1]]]
        )), dtype=torch.float32)
        postiveLabel = torch.tensor([1], dtype=torch.float32)
        return postivePair, postiveLabel

def order2postivePairs(order):
    pairs = [list(combination) for combination in itertools.combinations(order, 2)]
    return pairs
