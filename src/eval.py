import json
import pickle
from sklearn.metrics import ndcg_score
import torch
from dataset import *
from model import *

# load checkpoint
with open('./checkpoint/mo_otherRankingPageMinMaxScaler.pkl', 'rb') as file:
    minMaxScaler = pickle.load(file)

model = RankNet().to(getDevice())
model.load_state_dict(torch.load('./checkpoint/mo_otherRankingPageStateDict'))
model.eval()

# load data
with open('./dataset/gameAllItemDict.json', 'r') as infile:
    itemDict = json.load(infile)
with open('./dataset/gameAllRankItem.json', 'r') as infile:
    rankItem = json.load(infile)

# preprocessing
processedItemDict = {key: preprocessing(value) for key, value in itemDict.items()}
item2Index, itemFeature = decompositionItemDict(processedItemDict)
itemFeature, _ = normalize(itemFeature, minMaxScaler)

# eval
with torch.no_grad():
    score = model(torch.tensor(itemFeature, dtype=torch.float32).to(getDevice()))
    score = score.cpu()

# ndcg
totalSize = len(rankItem)
yTrue = list([i for i in range(totalSize, 0, -1)])
yScore = []
for item in rankItem:
    yScore.append(score[item2Index[item]][0])
print(ndcg_score([yTrue], [yScore]))

# demo page
demoHtml = "<html><head><style>img {width: '180px';} div {width: 45%; display: inline-block; padding: 10px;}</style></head><body><div>"

for i in rankItem:
    demoHtml = demoHtml + f"<img width = '180px' src='https:{itemDict[i]['work_image']}'/>"
demoHtml = demoHtml + "</div><div>"

def rankfunction(item):
    return score[item2Index[item]][0]
rankItem.sort(key = rankfunction, reverse=True)

for i in rankItem:
    demoHtml = demoHtml + f"<img width = '180px' src='https:{itemDict[i]['work_image']}'/>"

demoHtml = demoHtml + "</div></body></html>"
print(demoHtml)
