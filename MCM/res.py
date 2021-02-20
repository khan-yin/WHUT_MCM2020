
import glob
import json
import math
from torch.utils.data import Dataset,DataLoader
from data import HornetsDataset
import transformdata
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold, StratifiedKFold,KFold
import numpy as np
import torch
import random
from torch.utils.data.sampler import WeightedRandomSampler
from myloss import MyCrossEntropyLoss
import myloss
import timm
import torchvision
import torch.nn as nn
import pandas as pd
from model import HornetsClassifier
from tqdm import tqdm
import torch.utils.data.sampler
import glob
from sklearn.model_selection import train_test_split
from sklearn import datasets
test_path = glob.glob('./origin/all/*.jpg')
print(len(test_path))
test_path.sort()
models = glob.glob('./model/*.pth')

loss_fn = MyCrossEntropyLoss()

model = HornetsClassifier('tf_efficientnet_b4_ns', 2, pretrained=True).cuda()
# model.load_state_dict(torch.load('./model/50/tf_efficientnet_b4_ns_fold_0_3.pth'))
model.load_state_dict(torch.load('D:/python/MCM/modelgood/tf_efficientnet_b4_ns_fold_0_2.pth'))


model.eval()

test_dataset = HornetsDataset(
    test_path,
    img_label=None,
    transform=transformdata.get_test_transforms()
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False
)

# pbar = tqdm(enumerate(val_loader), total=len(val_loader))
df = pd.DataFrame(columns=['filename','predict'])
total = []
possibility = []
for step, imgs in enumerate(test_dataloader):
    x = imgs.cuda()
    y_pred = model(x)
    p = torch.sigmoid(y_pred)
    p= p[:,1]+0.2
    _, pred = torch.max(y_pred.data, 1)
    pred = pred.reshape(pred.shape[0], 1)
    p  = p.data.cpu().numpy().reshape(1,-1).flatten().tolist()

    pred = pred.cpu().numpy().reshape(1,-1).flatten()
    total.append(pred.tolist())
    possibility.append(p)

#把列表转为字符串
b = str(total)
pa = str(possibility)
print(possibility)
#替换掉'['和']'
b = b.replace('[','')
b = b.replace(']','')
pa = pa.replace('[','')
pa = pa.replace(']','')
possibility = list(eval(pa))
print(possibility)
print(len(possibility))
#最后转化成列表
print(total)
total = list(eval(b))
print(len(total))

# print(total)
# print(len(total),len(test_path))




c={'filename': test_path,'predict': total,'possibility': possibility}
data=pd.DataFrame(c)
print(data)
data.to_csv('prediction_new.csv',index=False)