import json
import numpy
from sklearn.model_selection import train_test_split
from scipy import stats
from dataset import *

with open('./dataset/asmrAllItemDict.json', 'r') as infile:
    itemDict = json.load(infile)

asmrFeaturesList = [preprocessing(value) for key, value in itemDict.items()]
asmrFeaturesList = numpy.array(asmrFeaturesList)

trainFeaturesList, testFeaturesList = train_test_split(asmrFeaturesList, test_size = 0.1, shuffle = False)
trainFeaturesList = numpy.array(trainFeaturesList)
testFeaturesList = numpy.array(testFeaturesList)

featureSize = len(trainFeaturesList[0])
for featureIndex in range(featureSize):
    print(stats.kruskal(trainFeaturesList[:, featureIndex], testFeaturesList[:, featureIndex]))

print('================================================================')

with open('./dataset/gameAllItemDict.json', 'r') as infile:
    itemDict = json.load(infile)

gameFeaturesList = [preprocessing(value) for key, value in itemDict.items()]
gameFeaturesList = numpy.array(gameFeaturesList)

featureSize = len(trainFeaturesList[0])
for featureIndex in range(featureSize):
    print(stats.kruskal(asmrFeaturesList[:, featureIndex], gameFeaturesList[:, featureIndex]))
