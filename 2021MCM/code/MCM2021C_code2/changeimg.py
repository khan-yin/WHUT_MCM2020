#!/usr/bin/env python
# coding: utf-40

# In[1]:


from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize
)


# In[2]:


from albumentations.pytorch import ToTensorV2


# In[79]:


import glob
data_path = glob.glob('./data/Positive/*.jpg')
data_path.sort()


# In[42]:


from PIL import Image
img = Image.open(data_path[1]).convert('RGB')


# In[44]:


import cv2
import matplotlib.pyplot as plt


# In[49]:


target = './data/Positive/'
img_name = '00.jpg'
img = cv2.imread('./data/Positive/0.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

image1 = Compose([
        # 对比度受限直方图均衡
            #（Contrast Limited Adaptive Histogram Equalization）
        CLAHE(),
        # 随机旋转 90°
        RandomRotate90(),
        # 转置
        Transpose(),
        # 随机仿射变换
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50, rotate_limit=45, p=.75),
        # 模糊
        Blur(blur_limit=3),
        # 光学畸变
        OpticalDistortion(),
        # 网格畸变
        GridDistortion(),
        Transpose(p=0.5),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
    # 随机改变图片的 HUE、饱和度和值
        HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
        RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
    ], p=1.0)(image=img)['image']
# plt.figure(figsize=(10, 10))
# plt.imshow(image1)
image2 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
cv2.imwrite( target+img_name, image2);


# In[101]:


def change_img(img_path):
    img = Image.open(img_path).convert('RGB')
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cnt=20
    for c in range(cnt):
        image1 = Compose([
                # 对比度受限直方图均衡
                    #（Contrast Limited Adaptive Histogram Equalization）
                CLAHE(),
                # 随机旋转 90°
                RandomRotate90(),
                # 转置
                Transpose(),
                # 随机仿射变换
                ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50, rotate_limit=45, p=.75),
                # 模糊
                Blur(blur_limit=3),
                # 光学畸变
                OpticalDistortion(),
                # 网格畸变
                GridDistortion(),
                # 随机改变图片的 HUE、饱和度和值
                HueSaturationValue()
            ], p=1.0)(image=img)['image']
#         plt.figure(figsize=(10, 10))
#         plt.imshow(image1)
        image2 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

        filepath = img_path.split('\\')
#         print(filepath[0])
        name = filepath[1].split('.')[0]
        houzhui = filepath[1].split('.')[1]
        print(filepath[0]+'/'+name+'-'+str(c)+'.'+houzhui)
        cv2.imwrite( filepath[0]+'/'+name+'-'+str(c)+'.'+houzhui, image2);
    


# In[102]:


'./data/Positive\\0.jpg'.split('\\')


# In[103]:


for i in data_path:
    change_img(i)


# In[105]:


import os
import shutil
cnt=0
target='./data/train/'
isExists=os.path.exists(target)
if not isExists:
        os.makedirs(target) 
train_data_json = {}
for root, dirs, files in os.walk("./data/Positive/", topdown=False):
    for name in files:
        shutil.copy(os.path.join(root, name), target+str(cnt)+'.jpg')
        train_data_json[str(cnt)+'.jpg']=1
        cnt+=1

for root, dirs, files in os.walk("./data/Negative/", topdown=False):
    for name in files:
        shutil.copy(os.path.join(root, name), target+str(cnt)+'.jpg')
        train_data_json[str(cnt)+'.jpg']=0
        cnt+=1  


# In[106]:


len(train_data_json)


# In[107]:


import json
with open('train_data.json', 'w') as fp:
    json.dump(train_data_json, fp)


# In[ ]:




