import os
import shutil
import numpy as np
import pandas as pd
import json

img_id_path = './data/2021MCM_ProblemC_ Images_by_GlobalID.xlsx'
img_dataset_path = './data/2021MCMProblemC_DataSet.xlsx'
source_dir='./data/2021MCM_ProblemC_Files/'

def copy_img_to_right_dataset(classname,csv):
    target='./data/'
    if classname == 'Positive ID':
        target+='Positive/'
    elif classname == 'Negative ID':
        target+='Negative/'
    else:
        target+=classname+'//'
    isExists=os.path.exists(target)
    if not isExists:
        # 如果不存在则创建目录
        os.makedirs(target)
    right_img = csv[csv['Lab Status']==classname]
    right_img['FileName'].apply(lambda x: shutil.copy(source_dir+x, target))

def get_class_dir():
    img_id_csv = pd.read_excel(img_id_path)
    img_dataset = pd.read_excel(img_dataset_path)
    img_id_csv = img_id_csv[(img_id_csv['FileType'] == 'image/jpg') | (img_id_csv['FileType'] == 'image/png')]
    img_id_csv = img_id_csv[['FileName', 'GlobalID', 'FileType']]

    img_ds_idindex = img_dataset.set_index('GlobalID')
    img_id_idindex = img_id_csv.set_index('GlobalID')
    hasimg_csv = img_id_idindex.join(img_ds_idindex, how='left').reset_index()
    num_classes = hasimg_csv['Lab Status'].unique()
    hasimg_csv=hasimg_csv.drop_duplicates('GlobalID')

    for classname in num_classes:
        copy_img_to_right_dataset(classname, hasimg_csv)

def ckeck_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def get_train_test_dir():
    cnt=0
    target='./data/train/'
    ckeck_dir(target)
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
    with open('train_data.json', 'w') as fp:
        json.dump(train_data_json, fp)

    cnt = 0
    target = './data/test/'
    ckeck_dir(target)
    for root, dirs, files in os.walk("./data/Unverified/", topdown=False):
        for name in files:
            shutil.copy(os.path.join(root, name), target + str(cnt) + '.jpg')
            cnt += 1

if __name__ == '__main__':
    get_class_dir()
    get_train_test_dir()
