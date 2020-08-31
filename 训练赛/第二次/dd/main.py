# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 18:13:36 2018

@author: Utkarsh
"""


import cv2
import numpy as np
import skimage.morphology as sm
from skimage import filters
import skimage
import os
import pandas as pd
from getTerminationBifurcation import getTerminationBifurcation
from removeSpuriousMinutiae import removeSpuriousMinutiae
from CommonFunctions import *
from extractMinutiaeFeatures import extractMinutiaeFeatures



def normalizedArr(data, newMinValue=-1, newMaxValue=1):
    minVal = np.amin(data)
    maxVal = np.amax(data)
    normalized = (data-minVal)*newMaxValue/(maxVal-minVal) + newMinValue
    return normalized

def getGradient(img):
    SCALE = 1/2
    SOBEL_KSIZE = 5
    GUASSIAN_KSIZE = (7, 7)
    img = cv2.resize(img, (0, 0), fx=SCALE, fy=SCALE)
    gradX = cv2.Sobel(img/255., cv2.CV_16S, 1, 0, ksize=SOBEL_KSIZE)
    gradY = cv2.Sobel(img/255., cv2.CV_16S, 0, 1, ksize=SOBEL_KSIZE)

    # gradX = normalizedArr(gradX, -1,1)
    # gradY = normalizedArr(gradY, -1,1)

    dxdy = 2 * gradX*gradY + 1e-3
    dx2_dy2 = gradX**2 - gradY**2
    theta = np.arctan(dx2_dy2/dxdy)

    phix = np.cos(2*theta)
    phiy = np.sin(2*theta)

    phix = cv2.GaussianBlur(phix, GUASSIAN_KSIZE, 15)
    phiy = cv2.GaussianBlur(phiy, GUASSIAN_KSIZE, 15)

    THRESHOLD = 200

    O = .5*np.arctan(2*(phiy/phix))
    imgO = normalizedArr(O, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(PROCESS_DIR, "0imgO.png"), imgO)

    imgOEdge = normalizedArr(filters.sobel(imgO), 0, 255)
    cv2.imwrite(os.path.join(PROCESS_DIR, "1imgOEdge.png"), imgOEdge)

    imgOEdge = sm.dilation(imgOEdge, sm.disk(2))
    imgOEdge = sm.closing(imgOEdge, sm.disk(4))
    cv2.imwrite(os.path.join(PROCESS_DIR, "2imgOEdge_closing.png"), imgOEdge)

    imgOEdge = (imgOEdge > THRESHOLD)*255
    cv2.imwrite(os.path.join(PROCESS_DIR, "3imgOEdge_01.png"), imgOEdge)
    # plt_show(imgOEdge>THRESHOLD)
    imgOSkel = sm.skeletonize(imgOEdge > THRESHOLD)*255
    cv2.imwrite(os.path.join(PROCESS_DIR, "4imgOSkel.png"), imgOSkel)

    return gradX, gradY


def test(img_id):
    img_name = "{:02d}.tif".format(img_id)
    centerdf = pd.read_excel(os.path.join(DATA_DIR, "new_center.xlsx")).astype(int)
    center = tuple(centerdf.loc[img_id-1])
    print("center :{}".format(center))
    img = cv2.imread('../enhanced/'+img_name, 0)
    # 获取二值化图像
    img_01 = np.uint8(img > 128)
    # skeletonize细化图像，细化用于减少二值图像中的每个连通分量，到一个单像素宽的骨架。
    skel = sm.skeletonize(img_01)
    skel = np.uint8(skel) * 255
    # 未经细化的二值图像
    mask = img_01 * 255
    mask = sm.dilation(mask, sm.disk(1))
    cv2.imwrite(os.path.join(RESULT_DIR, "00mask.png"), mask)

    foregroundArea = sm.closing(mask, sm.disk(15))
    _, _, angle = External_ellipse(foregroundArea, mask)
    edge_filter = sm.erosion(foregroundArea, sm.disk(15))
    edge_filter = sm.erosion(edge_filter, sm.disk(15))

    cv2.imwrite(os.path.join(RESULT_DIR, "01edge_filter.png"), edge_filter)
    gradientX, gradientY = getGradient(mask)

    # 原始端点和分叉点
    (minutiaeTerm, minutiaeBif) = getTerminationBifurcation(skel, mask)

    Features = extractMinutiaeFeatures(skel, minutiaeTerm, minutiaeBif, center)
    Features = removeSpuriousMinutiae(Features, edge_filter, 10)
    # print("Features:\n {}".format(Features))
    ShowResults(skel, Features, edge_filter, center, img_name+".png", False)


if __name__ == '__main__':
    test(2)

