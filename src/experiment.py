import numpy
from scipy import stats
from train import *

repeatTimes = 20

originalCurrentRateRecords = []
modifiedCurrentRateRecords = []
for i in range(repeatTimes):
    trainCurrentRate, testCurrentRate = trainTestWithOtherRankingPage(
        learningRate = 0.01,
        l1Weight = 0,
        l2Weight = 0,
        shareScaler = True,
        printResult = False
    )
    originalCurrentRateRecords.append(testCurrentRate)

    trainCurrentRate, testCurrentRate = trainTestWithOtherRankingPage(
        learningRate = 0.001,
        l1Weight = 0.001,
        l2Weight = 0.001,
        shareScaler = False,
        printResult = False
    )
    modifiedCurrentRateRecords.append(testCurrentRate)

print('originalCurrentRateRecords')
originalMean = numpy.mean(originalCurrentRateRecords)
originalStd = numpy.std(originalCurrentRateRecords)
print(f'originalMean: {originalMean:.2f}')
print(f'originalStd: {originalStd:.2f}')

print('==========================')

print('modifiedCurrentRateRecords')
modifiedMean = numpy.mean(modifiedCurrentRateRecords)
modifiedStd = numpy.std(modifiedCurrentRateRecords)
print(f'modifiedMean: {modifiedMean:.2f}')
print(f'modifiedStd: {modifiedStd:.2f}')

print('==========================')

print('hypothesisTesting')
print(stats.ttest_ind_from_stats( 
    mean1 = originalMean,
    std1 = originalStd,
    nobs1 = repeatTimes,   
    mean2 = modifiedMean,
    std2 = modifiedStd,
    nobs2 = repeatTimes
))
print(stats.kruskal(originalCurrentRateRecords, modifiedCurrentRateRecords))
