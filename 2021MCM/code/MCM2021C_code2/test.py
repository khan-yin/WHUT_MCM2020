from myloss import MyCrossEntropyLoss
from model import HornetsClassifier
import torch.utils.data.sampler
import torch
import pandas as pd
import numpy as np
import glob
import json
from PIL import Image
import torch.nn
from torch.utils.data import Dataset,DataLoader
import transformdata
from tqdm import tqdm

class OriginHornetsDataset(Dataset):
    def __init__(self, img_path,signal='60', img_label=None, transform=None):
        self.img_path = img_path
        self.img_label = img_label
        self.classes_for_all_imgs = []
        self.signal = signal
        if self.img_label is not None:
            train_json = json.load(open('./train_data_'+signal+'.json'))
            for path in img_path:
                name = path.split('\\')[1]
                class_id = 0
                if train_json[name] == 0:
                    class_id = 0
                elif train_json[name] == 1:
                    class_id = 1
                self.classes_for_all_imgs.append(class_id)
        # print(type(img_path))
        # print(type(img_label))
        # print(img_path)
        # print(img_label)
        # print(len(img_path)==len(img_label))
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

    def __getitem__(self, index):
        img = Image.open(self.img_path[index]).convert('RGB')
        name = self.img_path[index].split('\\')[1]
        train_json = json.load(open('./train_data_'+self.signal+'.json'))
        # print(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.img_label is not None:
            label = np.array(train_json[name], dtype=np.int)
            # print(label)
            # print(img,torch.from_numpy(label))
            return img, torch.from_numpy(label)
        else:
            return img

    def __len__(self):
        return len(self.img_path)

    def get_classes_for_all_imgs(self):
        return self.classes_for_all_imgs

signal ='50'
test_path = glob.glob('./newdata/'+signal+'/test/*.jpg')
test_path.sort()
data_path = glob.glob('./newdata/'+signal+'/train/*.jpg')
data_path.sort()

models = glob.glob('./model/'+signal+'/*.pth')
model = HornetsClassifier('tf_efficientnet_b4_ns', 2, pretrained=True)
loss_fn = MyCrossEntropyLoss()

CFG = {
    'fold_num': 10,
    'img_size': 64,
    'lr': 1e-4,
    'weight_decay':1e-6,
    'epochs': 5,
    'batch_size':16,
    'model_arch': 'tf_efficientnet_b4_ns'
}

all_train_dataset = OriginHornetsDataset(
    img_path=data_path,
    signal=signal,
    img_label=True,
    transform=transformdata.get_train_transforms()
)

all_train_dataloader = DataLoader(
    all_train_dataset,
    batch_size=16,
    shuffle=True
)

test_dataset = OriginHornetsDataset(
    test_path,
    img_label=None,
    transform=transformdata.get_test_transforms()
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False
)


# TP(True Positive)：将正类预测为正类数，真实为1，预测也为1
# FN(False Negative)：将正类预测为负类数，真实为1，预测为0
# FP(False Positive)：将负类预测为正类数， 真实为0，预测为1
# TN(True Negative)：将负类预测为负类数，真实为0，预测也为0
dict_list = []
par = tqdm(models,total=len(models))
for model_path in par:
    model_info={}
    # print(model_path)
    model_name = model_path.split('\\')[1]
    # print(model_name)
    model.cuda()
    model.load_state_dict(torch.load(model_path))
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    running_loss = 0.0
    running_corrects = 0
    right = 0
    TP = 0
    FN = 0
    FP = 0
    TN = 0
    total_positive = 0
    total_negative = 0
    with torch.no_grad():
        for step, (imgs, image_labels) in enumerate(all_train_dataloader):
            x, y = imgs.cuda(), image_labels.cuda()
            y=y.view(y.shape[0],-1)
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
            FP += torch.sum((y.data == 0) & (pred != y.data) )
            # print(TP,FN)

            running_corrects += torch.sum(pred == y.data)
            # print(len(val_loader))
        epoch_loss = running_loss
        epoch_acc = float(running_corrects) / (CFG['batch_size'] * len(all_train_dataloader))
        recall_acc = TP/(TP + FN)
        total_acc = (TP+TN) / (TP+TN+FP+FN)
        Precision_acc = TP /(TP+FP)
        F1 = (2*TP)/(2*TP+FP+FN)
        # print(Precision_acc)
        # print(total_acc)
        print("{} test total_total_acc:{:.4f} recall_acc:{:.4f} Precision_acc:{:.4f} F1:{:.4f} TP:{:.4f} TN:{:.4f} FP:{:.4f} FN:{:.4f}".format(model_name,total_acc,recall_acc,Precision_acc,F1,TP,TN,FP,FN))
        model_info['name'] = model_name
        model_info['totalacc'] = epoch_acc
        model_info['recall'] = recall_acc.item()
        model_info['Precision'] = Precision_acc.item()
        model_info['F1'] = F1.item()
        model_info['TP'] = TP.item()
        model_info['TN'] = TN.item()
        model_info['FN'] = FN.item()
        model_info['FP'] = FP.item()
        dict_list.append(model_info)

print(dict_list)
pd.DataFrame(dict_list).to_csv('test_info'+signal+'.csv',index=False)














