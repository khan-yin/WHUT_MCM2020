# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 11:42:58 2016

@author: utkarsh
"""

import numpy as np
import cv2
import sys
import os
from Enhancement.src.image_enhance import image_enhance

DATA_DIR = '../../../data'
IMG_SRC_DIR = '../../../data/img'


def enhance_dir(path, savepath):
    for imgfile in os.listdir(path):
        print(imgfile)
        img = cv2.imread(os.path.join(path, imgfile))
        if (len(img.shape) > 2):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        enhanced_img = image_enhance(img)
        cv2.imwrite(os.path.join(savepath, imgfile), (255 * enhanced_img))


def test():
    print('loading sample image')
    # img_name = '16.jpg'
    # img = cv2.imread('../images/' + img_name)
    img_name = '12.tif'
    img = cv2.imread(IMG_SRC_DIR + img_name)
    if (len(img.shape) > 2):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    enhanced_img = image_enhance(img)
    print('saving the image')
    cv2.imwrite('../../Feature_Extraction/enhanced/' + img_name, (255 * enhanced_img))


if __name__ == '__main__':
    enhance_dir(IMG_SRC_DIR, '../../Feature_Extraction/enhanced/')
    #test()

