import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import glob
import json
from PIL import Image
import torch.nn
from torch.utils.data import Dataset,DataLoader
import torch.nn.functional as F
import transformdata

class HornetsDataset(Dataset):
    def __init__(self, img_path,signal='60', img_label=None, transform=None):
        self.img_path = img_path
        self.img_label = img_label
        self.classes_for_all_imgs = []
        self.signal =signal

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


if __name__ == '__main__':
    train_path = glob.glob('./data/train/*.jpg')
    train_path.sort()
    # print(train_path)
    train_json=json.load(open('./train_data.json'))
    train_label = [train_json[x] for x in train_json]
    # print(train_label)

    train_dataset = HornetsDataset(
        img_path=train_path,
        img_label=train_label,
        transform=transformdata.get_train_transforms(),
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=120,
        shuffle=True
    )
    for step,(imgs,label) in enumerate(train_loader):
        print(imgs.shape,label.shape)
        break




