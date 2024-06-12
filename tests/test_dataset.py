import unittest
import unittest.mock as mock
import torch
from src.dataset import *

class TestDataset(unittest.TestCase):

    @mock.patch('random.shuffle')
    def test_get(self, mockShuffle):
        rankItem = ['1', '2', '3', '4', '5']
        testRatio = 0.4
        actualTrainData, actualTestDate = orderedTrainTestSplit(rankItem, testRatio)

        expectedTrainData = ['3', '4', '5']
        expectedTestDate = ['1', '2']
        self.assertEqual(actualTrainData, expectedTrainData)
        self.assertEqual(actualTestDate, expectedTestDate)

    @mock.patch('src.dataset.PostivePairsDataset')
    @mock.patch('src.dataset.normalize')
    @mock.patch('src.dataset.decompositionItemDict')
    @mock.patch('src.dataset.preprocessing')
    def test_getNormalizedDataset(self, mockPreprocessing, mockDecompositionItemDict, mockNormalize, MockPostivePairsDataset):
        rankItem = ['id01', 'id02']
        itemDict = {
            'id01Key': 'id01Value',
            'id02Key': 'id02Value'
        }
        minMaxScaler = 'minMaxScaler'

        mockPreprocessing.side_effect = [
            'preprocessedId01Value',
            'preprocessedId02Value'
        ]
        mockDecompositionItemDict.return_value = ('item2Index', 'itemFeature')
        mockNormalize.return_value = ('normalizedItemFeature', 'normalizedMinMaxScaler')
        MockPostivePairsDataset.return_value = 'postivePairsDataset'
        actualPostivePairsDataset, actualMinMaxScaler = getNormalizedDataset(rankItem, itemDict, minMaxScaler)

        calls = [mock.call('id01Value'), mock.call('id02Value')]
        mockPreprocessing.assert_has_calls(calls)
        self.assertEqual(mockPreprocessing.call_count, 2)
        mockDecompositionItemDict.assert_called_once_with({
            'id01Key': 'preprocessedId01Value',
            'id02Key': 'preprocessedId02Value'
        })
        mockNormalize.assert_called_once_with('itemFeature', 'minMaxScaler')
        MockPostivePairsDataset.assert_called_once_with(rankItem, 'item2Index', 'normalizedItemFeature')
        self.assertEqual(actualPostivePairsDataset, 'postivePairsDataset')
        self.assertEqual(actualMinMaxScaler, 'normalizedMinMaxScaler')

    def test_preprocessing(self):
        itemDict = {
            'id01': {
                'dl_count': '1',
                'wishlist_count': '1',
                'rate_average_2dp': '1',
                'rate_count': '1',
                'rate_count_detail': [
                    {'count': '1'},
                    {'count': '1'},
                    {'count': '1'},
                    {'count': '1'},
                    {'count': '1'}
                ],
                'review_count': '1',
                'price': '1',
                'regist_date': '2024-01-01 00:00:00'
            },
            'id02': {
                'dl_count': None,
                'wishlist_count': None,
                'rate_average_2dp': None,
                'review_count': None,
                'price': '2',
                'regist_date': '2024-02-01 00:00:00'
            },
            'id03': {
                'price': '3',
                'regist_date': '2024-03-01 00:00:00'
            }
        }
        itemDict = {key: preprocessing(value) for key, value in itemDict.items()}

        actualInputFeature = itemDict['id01']
        expectedInputFeature = [1, 1, 1.0, 1, 1, 1, 1, 1, 1, 1, 1, 1704038400]
        self.assertEqual(actualInputFeature, expectedInputFeature)

        actualInputFeature = itemDict['id02']
        expectedInputFeature = [0, 0, 0.0, 0, 0, 0, 0, 0, 0, 0, 2, 1706716800]
        self.assertEqual(actualInputFeature, expectedInputFeature)

        actualInputFeature = itemDict['id03']
        expectedInputFeature = [0, 0, 0.0, 0, 0, 0, 0, 0, 0, 0, 3, 1709222400]
        self.assertEqual(actualInputFeature, expectedInputFeature)

    def test_decompositionItemDict(self):
        itemDict = {
            'id02': [2, 2, 2],
            'id03': [3, 3, 3],
            'id01': [1, 1, 1]
        }
        item2Index, itemFeature = decompositionItemDict(itemDict)
        self.assertEqual(itemFeature[item2Index['id01']], itemDict['id01'])
        self.assertEqual(itemFeature[item2Index['id02']], itemDict['id02'])
        self.assertEqual(itemFeature[item2Index['id03']], itemDict['id03'])

    @mock.patch('src.dataset.MinMaxScaler')
    def test_normalize(self, MockMinMaxScaler):
        expectedItemFeature = 'itemFeature'
        actualItemFeature = 'itemFeature'
        expectedTransformReturnValue = 'transformReturnValue'
        mockMinMaxScaler = mock.Mock()
        mockMinMaxScaler.transform.return_value = expectedTransformReturnValue
        MockMinMaxScaler.return_value = mockMinMaxScaler
        actualNormalizeItemFeature, minMaxScaler = normalize(actualItemFeature)
        mockMinMaxScaler.fit.assert_called_once_with(expectedItemFeature)
        mockMinMaxScaler.transform.assert_called_once_with(expectedItemFeature)
        self.assertEqual(actualNormalizeItemFeature, expectedTransformReturnValue)
        self.assertEqual(minMaxScaler, mockMinMaxScaler)

        expectedItemFeature = 'itemFeature'
        actualItemFeature = 'itemFeature'
        expectedTransformReturnValue = 'transformReturnValue'
        mockMinMaxScaler = mock.Mock()
        mockMinMaxScaler.transform.return_value = expectedTransformReturnValue
        MockMinMaxScaler.return_value = mockMinMaxScaler
        actualNormalizeItemFeature, minMaxScaler = normalize(actualItemFeature, mockMinMaxScaler)
        mockMinMaxScaler.fit.assert_not_called()
        mockMinMaxScaler.transform.assert_called_once_with(expectedItemFeature)
        self.assertEqual(actualNormalizeItemFeature, expectedTransformReturnValue)
        self.assertEqual(minMaxScaler, mockMinMaxScaler)

    def test_PostivePairsDataset(self):
        data = PostivePairsDataset(
            ['id01', 'id02', 'id03'],
            {
                'id01': 2,
                'id02': 1,
                'id03': 0
            },
            [
                [3, 3, 3],
                [2, 2, 2],
                [1, 1, 1]
            ]
        )

        expectedInput = torch.tensor([[1, 1, 1], [2, 2, 2]], dtype=torch.float32)
        expectedLabel = torch.tensor([1], dtype=torch.float32)
        actualInput, actualLabel = data[0]
        torch.testing.assert_close(actualInput, expectedInput)
        torch.testing.assert_close(actualLabel, expectedLabel)

        expectedInput = torch.tensor([[1, 1, 1], [3, 3, 3]], dtype=torch.float32)
        actualInput, actualLabel = data[1]
        torch.testing.assert_close(actualInput, expectedInput)
        torch.testing.assert_close(actualLabel, expectedLabel)

        expectedInput = torch.tensor([[2, 2, 2], [3, 3, 3]], dtype=torch.float32)
        actualInput, actualLabel = data[2]
        torch.testing.assert_close(actualInput, expectedInput)
        torch.testing.assert_close(actualLabel, expectedLabel)

    def test_order2postivePairs(self):
        order = [1, 2, 3]
        expectedPairs = [
            [1, 2],
            [1, 3],
            [2, 3]
        ]
        actualPairs = order2postivePairs(order)
        self.assertEqual(actualPairs, expectedPairs)

        order = ['C', 'B', 'A']
        expectedPairs = [
            ['C', 'B'],
            ['C', 'A'],
            ['B', 'A']
        ]
        actualPairs = order2postivePairs(order)
        self.assertEqual(actualPairs, expectedPairs)
