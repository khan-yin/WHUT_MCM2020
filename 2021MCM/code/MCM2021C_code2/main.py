import glob
import json
from torch.utils.data import Dataset,DataLoader
from data import HornetsDataset
from myloss import MyCrossEntropyLoss
import transformdata
from sklearn.model_selection import GroupKFold, StratifiedKFold,KFold
import numpy as np
import matplotlib.pyplot as plt
import torch
import random
import os
import pandas as pd
import torch.nn as nn
from model import HornetsClassifier
from tqdm import tqdm
from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn import datasets
torch.cuda.empty_cache()
mse_list = []
signal = '50'
fold_num = 10
iterate = 1000


train_file_cnt = {
    '40' : 144,
    '45' : 135,
    '50' : 128,
    '55' : 116,
    '60' : 22
}
batch_size = 10

CFG = {
    'fold_num': fold_num,
    'img_size': 64,
    'lr': 1e-4,
    'weight_decay':1e-6,
    'epochs': iterate//(fold_num*((fold_num-1)*train_file_cnt[signal]//(batch_size*fold_num))),
    'batch_size':batch_size,
    'model_arch': 'tf_efficientnet_b4_ns'
}
print(CFG['epochs'])

# K则，并分层抽样
data_path = glob.glob('./newdata/'+signal+'/train/*.jpg')
data_path.sort()
# print(data_path)
data_json = json.load(open('./train_data_'+signal+'.json'))
data_label = [data_json[x] for x in data_json]



def prepare_dataloader(trn_idx, val_idx):
    # print(data_path[trn_idx[0]:trn_idx[-1]])
    train_data_path = [ data_path[i]  for i in trn_idx]
    val_data_path = [ data_path[i]  for i in val_idx]
    random.shuffle(train_data_path)
    random.shuffle(val_data_path)

    # print(train_data_path)

    # print(train_data_path)
    # print(val_data_path)
    # Compute samples weight (each sample should get its own weight)
    # train_weight = torch.tensor([data_label[trn_idx[0]:trn_idx[-1]].count(0)/len(data_label[trn_idx[0]:trn_idx[-1]]),data_label[trn_idx[0]:trn_idx[-1]].count(1)/len(data_label[trn_idx[0]:trn_idx[-1]])])
    #
    # train_weight = 1./train_weight
    # print(train_weight)
    # train_weight =torch.tensor([0.2,0.40])
    # val_weight = torch.tensor([data_label[val_idx[0]:val_idx[-1]].count(0) / len(data_label[val_idx[0]:val_idx[-1]]),
    #                             data_label[val_idx[0]:val_idx[-1]].count(1) / len(
    #                                 data_label[val_idx[0]:val_idx[-1]])])

    # val_weight = torch.tensor([0.2, 0.40])
    # val_weight = 1./val_weight
    # print(val_weight)

    train_dataset = HornetsDataset(
        img_path=train_data_path,
        signal=signal,
        img_label=data_label[trn_idx[0]:trn_idx[-1]],
        transform=transformdata.get_train_transforms(),
    )


    # class_sample_counts = [len(data_path), 10]
    # weights = 1. / torch.tensor(class_sample_counts, dtype=torch.float)
    # train_targets = train_dataset.get_classes_for_all_imgs()
    # samples_weights = weights[train_targets]
    # sampler = WeightedRandomSampler(weights=samples_weights, num_samples=len(samples_weights), replacement=True)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=CFG['batch_size'],
        # shuffle = False if sampler is not None else True,
        shuffle=True,
        drop_last=True,
    )

    valid_dataset = HornetsDataset(
        img_path=val_data_path,
        signal=signal,
        img_label=data_label[val_idx[0]:val_idx[-1]],
        transform=transformdata.get_train_transforms(),
    )

    val_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=CFG['batch_size'],
        shuffle=True,
    )
    return train_loader, val_loader


def train_one_epoch(model, loss_fn, optimizer, train_loader, ds_len):
    model.train()
    # print('training')
    running_loss = 0.0
    running_corrects = 0
    right = 0
    # print("train_loader", len(train_loader))


    # pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    positive_acc_list=[]
    for step, (imgs, image_labels) in enumerate(train_loader):
        x, y = imgs.cuda(), image_labels.cuda()
        # print(y.shape)
        y=y.view(y.shape[0], -1)
        # print(y.shape)
        # print(y.squeeze(1).long())
        y_pred = model(x)
        # print(y_pred.shape)
        _, pred = torch.max(y_pred.data, 1)
        pred = pred.reshape(pred.shape[0], 1)
        optimizer.zero_grad()
        loss = loss_fn(y_pred, y.squeeze(1).long())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        loss_list.append(loss.item())
        # print("k")
        # print(pred == y.data)
        # print(torch.sum(pred == y.data))
        positive_in_one_batch = torch.sum((y.data==1))
        if positive_in_one_batch >0:
            right = torch.sum((pred == y.data) & (y.data==1))
            # print(positive_in_one_batch)
            # print(float(right))
            # print(float(right)/positive_in_one_batch.item())
            positive_acc_list.append(float(right)/positive_in_one_batch.item())

        running_corrects += torch.sum(pred == y.data)
        # print(pred.shape)
        # print(y.data.shape)

        # print(pred == y.data)
        # if step%5==0 and step>0:
        #     print("step {}, Train Loss:{:.4f}, Train ACC:{:.2f}%".format(step,loss.item(),
        #                   torch.sum(pred == y.data)*100 / (CFG['batch_size'] )))

    epoch_loss = running_loss
    epoch_acc = float(running_corrects)*100 / (CFG['batch_size'] * len(train_loader))
    positive_acc = 100*sum(positive_acc_list)/len(positive_acc_list)
    # print("epoch: Train: Loss:{:.4f} Acc:{:.4f}% positive_acc:{:.4f}%".format(epoch_loss,epoch_acc,positive_acc))



def valid_one_epoch(model_name,model,loss_fn,val_loader,ds_len):
    model_info = {}
    # print(model_path)
    # model_name = model_path.split('\\')[1]
    # print(model_name)
    model.eval()
    model.cuda()

    running_loss = 0.0
    running_corrects = 0
    right = 0
    TP = 0
    FN = 0
    FP = 0
    TN = 0
    total_positive = 0
    total_negative = 0

    for step, (imgs, image_labels) in enumerate(val_loader):

        x, y = imgs.cuda(), image_labels.cuda()
        y = y.view(y.shape[0], -1)
        y_pred = model(x)
        _, pred = torch.max(y_pred.data, 1)
        pred = pred.reshape(pred.shape[0], 1)
        loss = loss_fn(y_pred, y.squeeze(1).long())
        running_loss += loss.item()
        # print(loss.item())
        # print(torch.sum(pred == y.data))
        positive_in_one_batch = torch.sum((y.data == 1))
        total_negative += torch.sum((y.data == 1))
        if positive_in_one_batch > 0:
            TP += torch.sum((pred == y.data) & (y.data == 1))
            FN += torch.sum((pred != y.data) & (y.data == 1))
            total_positive += positive_in_one_batch
        TN += torch.sum((pred == y.data) & (y.data == 0))
        FP += torch.sum((y.data == 0) & (pred != y.data))
        # print(TP,FN)

        running_corrects += torch.sum(pred == y.data)
        # print(len(val_loader))
    # print(running_loss)
    # print(CFG['batch_size'])
    # print(len(val_loader))
    # print(running_loss/(CFG['batch_size']*len(val_loader)))
    mse_list.append(running_loss/(CFG['batch_size']*len(val_loader)))
    epoch_acc = float(running_corrects) / (CFG['batch_size'] * len(val_loader))
    recall_acc = TP / (TP + FN) if (TP + FN) !=0 else torch.tensor([0]).item()
    total_acc = (TP + TN) / (TP + TN + FP + FN)
    Precision_acc = TP / (TP + FP)
    F1 = (2 * TP) / (2 * TP + FP + FN)
    # print(recall_acc)
    # print(Precision_acc)
    # print(total_acc)
    print(
        "{} test total_total_acc:{:.4f} recall_acc:{:.4f} Precision_acc:{:.4f} F1:{:.4f} TP:{:.4f} TN:{:.4f} FP:{:.4f} FN:{:.4f}".format(
            model_name, total_acc, recall_acc, Precision_acc, F1, TP, TN, FP, FN))
    model_info['name'] = model_name
    model_info['totalacc'] = epoch_acc
    model_info['recall'] = recall_acc if recall_acc == 0 else recall_acc.item()
    model_info['Precision'] = Precision_acc.item()
    model_info['F1'] = F1.item()
    model_info['TP'] = TP if TP == 0 else TP.item()
    model_info['TN'] = TN if TN == 0 else TN.item()
    model_info['FN'] = FN if FN == 0 else FN.item()
    model_info['FP'] = FP if FP == 0 else FP.item()
    dict_list.append(model_info)

    # model.eval()
    #
    # running_loss = 0.0
    # running_corrects = 0
    # right = 0
    # positive_acc_list = []
    # # pbar = tqdm(enumerate(val_loader), total=len(val_loader))
    # for step, (imgs, image_labels) in enumerate(val_loader):
    #     x, y = imgs.cuda(), image_labels.cuda()
    #     y=y.view(y.shape[0],-1)
    #     y_pred = model(x)
    #
    #     _, pred = torch.max(y_pred.data, 1)
    #     pred = pred.reshape(pred.shape[0], 1)
    #     loss = loss_fn(y_pred, y.squeeze(1).long())
    #     running_loss += loss.item()
    #
    #     positive_in_one_batch = torch.sum((y.data == 1))
    #     if positive_in_one_batch > 0:
    #         right = torch.sum((pred == y.data) & (y.data == 1))
    #         positive_acc_list.append(float(right) / positive_in_one_batch.item())
    #
    #     running_corrects += torch.sum(pred == y.data)
    #     # print(len(val_loader))
    # epoch_loss = running_loss
    # epoch_acc = float(running_corrects)*100 / (CFG['batch_size'] * len(val_loader))
    # positive_acc = 100*sum(positive_acc_list)/len(positive_acc_list)
    # print("epoch: test: Loss:{:.4f} Acc:{:.4f}% positive_acc:{:.4f}%".format(epoch_loss, epoch_acc,positive_acc))
    #




folds = StratifiedKFold(n_splits=CFG['fold_num']).split(np.arange(len(data_path)),data_label)

device = "cuda:0"
model = HornetsClassifier(CFG['model_arch'],n_class=2,pretrained=True).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
loss_fn = nn.CrossEntropyLoss()#MyCrossEntropyLoss()
loss_list = []
dict_list = []


for fold, (trn_idx, val_idx) in enumerate(folds):
    # print(len(val_idx))
    # print("len",len(trn_idx))
    for epoch in range(CFG['epochs']):
        print("ep:",epoch)
        modelname = '{}_fold_{}_{}.pth'.format(CFG['model_arch'], fold, epoch)
        print(epoch)
        train_loader, val_loader = prepare_dataloader(trn_idx, val_idx)
        train_one_epoch(model, loss_fn, optimizer, train_loader, len(trn_idx))
        with torch.no_grad():
            valid_one_epoch(modelname,model, loss_fn, val_loader,len(val_idx))
        filepath = os.path.join('./model/'+signal, modelname)  # 最终参数模型
        torch.save(model.state_dict(), filepath)

            # del model, optimizer, train_loader, val_loader
            # torch.cuda.empty_cache()
            # torch.save(model.cnn_model.state_dict(),'{}/cnn_model_fold_{}_{}'.format(CFG['model_path'], fold, CFG['tag']))

# print(dict_list)
print("mean_loss",sum(mse_list)/len(mse_list))
pd.DataFrame(dict_list).to_csv('mode_info' + signal + '.csv', index=False)



lis_x = [ i for i in range(len(loss_list))]

fig = plt.figure(figsize = (7,5))    #figsize是图片的大小`
ax1 = fig.add_subplot(1, 1, 1) # ax1是子图的名字`
plt.plot(lis_x,loss_list,label=u'loss value')
plt.legend()
#显示图例
plt.xlabel(u'iters')
plt.ylabel(u'loss')
plt.title('training loss in efficientnet')


# def np_move_avg(a,n,mode="valid"):
#     return(np.convolve(a, np.ones((n,))/n, mode=mode))
# plt.plot(np_move_avg(np.array(loss_list), 20, mode="same"));
# plt.plot(np_move_avg(np.array(loss_list), 20, mode="full"));
# plt.plot(np_move_avg(np.array(loss_list), 20, mode="valid"),label=u'loss value');
axes = plt.axes()
axes.set_xlim([0, len(loss_list)])
axes.set_ylim([0, 1])
plt.show()

with open('./loss'+signal+'.txt','w') as f:
    for line in loss_list:
        f.write(str(line) + '\n')