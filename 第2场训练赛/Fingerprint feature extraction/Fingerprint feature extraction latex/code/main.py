# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 18:13:36 2018

@author: Utkarsh
"""


import cv2
import numpy as np
import pandas as pd
import skimage.morphology as sm
from skimage import filters
import skimage
import os
import warnings
warnings.simplefilter("ignore")

from Feature_Extraction.src.getTerminationBifurcation import getTerminationBifurcation
from Feature_Extraction.src.removeSpuriousMinutiae import removeSpuriousMinutiae
from Feature_Extraction.src.getGradient import getGradient
from Feature_Extraction.src.CommonFunctions import *
from Feature_Extraction.src.extractMinutiaeFeatures import extractMinutiaeFeatures
from Feature_Extraction.src.matchMinutiae import getSimilarity


def getFeatures(img_id, featureNum=20):
    try:
        Features = pd.read_excel(os.path.join(FEATURE_DIR, "{:02d}.xlsx".format(img_id)))
        return Features[:featureNum]
    except FileNotFoundError:
        pass
    img_name = "{:02d}.tif".format(img_id)
    centerdf = pd.read_excel(os.path.join(DATA_DIR, "new_center.xlsx")).astype(int)
    center = tuple(centerdf.loc[img_id-1])
    # print("center :{}".format(center))
    img = cv2.imread('../rotated/'+img_name, 0)
    # 获取二值化图像
    img_01 = np.uint8(img > 128)
    # skeletonize细化图像，细化用于减少二值图像中的每个连通分量，到一个单像素宽的骨架。
    skel = sm.skeletonize(img_01)
    skel = np.uint8(skel) * 255
    # 未经细化的二值图像
    mask = img_01 * 255
    mask = sm.dilation(mask, sm.disk(1))
    cv2.imwrite(os.path.join(RESULT_DIR, "{:02d}mask.png".format(img_id)), mask)

    foregroundArea = sm.closing(mask, sm.disk(15))
    edge_filter = sm.erosion(foregroundArea, sm.disk(15))

    cv2.imwrite(os.path.join(RESULT_DIR, "{:02d}edge_filter.png".format(img_id)), edge_filter)
    # gradientX, gradientY = getGradient(mask)

    # 原始端点和分叉点
    (minutiaeTerm, minutiaeBif) = getTerminationBifurcation(skel, mask)

    Features = extractMinutiaeFeatures(skel, minutiaeTerm, minutiaeBif, center)
    Features = removeSpuriousMinutiae(Features, edge_filter, 5)#[:NUM_POINTS]
    # print(len(Features))
    Features.to_excel(os.path.join(FEATURE_DIR, "{:02d}.xlsx".format(img_id)))
    # print("Features:\n {}".format(Features))
    ShowResults(skel, Features, edge_filter, center, img_id, False)
    return Features[:featureNum]

def print_all_Features_len():
    for i in range(1, NUM_PIC+1):
        Features = getFeatures(i, 200)
        print(len(Features))

def print_all_Bytes():
    for i in range(1, NUM_PIC+1):
        Features = getFeatures(i, 20)
        prettyBytesStr = getPrettyFeaturesBytes(Features)
        print(prettyBytesStr)
        print()

def get_match_result(featureNum=20):
    similarityMat = np.zeros((NUM_PIC, NUM_PIC))
    for i in range(NUM_PIC):
        for j in range(NUM_PIC):
            Features1 = getFeatures(i+1, featureNum)
            Features2 = getFeatures(j+1, featureNum)
            total_potential = getSimilarity(Features1, Features2)
            similarityMat[i, j] = total_potential
    plot_mat(similarityMat, name="similarityMat{}".format(featureNum))
    pd.DataFrame(similarityMat).to_excel(os.path.join(FEATURE_DIR, "similarityMat{}.xlsx".format(featureNum)))
    return similarityMat

def getMatError(mat1, mat2):
    errMat = np.abs(mat1-mat2)
    avgErr = np.mean(errMat)
    maxErr = np.max(errMat)
    return avgErr, maxErr

def compare_match_result(featureNumLst=[10, 15, 20]):
    lstLen = len(featureNumLst)
    avgErrMat = np.zeros((lstLen, lstLen))
    maxErrMat = np.zeros((lstLen, lstLen))
    similarityMatLit = []
    for featureNum in featureNumLst:
        similarityMatLit.append(get_match_result(featureNum))
    for i in range(lstLen):
        for j in range(lstLen):
            avgErr, maxErr = getMatError(similarityMatLit[i], similarityMatLit[j])
            avgErrMat[i, j] = avgErr
            maxErrMat[i, j] = maxErr
    pd.DataFrame(avgErrMat).to_excel(os.path.join(FEATURE_DIR, "avgErrMat.xlsx"))
    pd.DataFrame(maxErrMat).to_excel(os.path.join(FEATURE_DIR, "maxErrMat.xlsx"))

if __name__ == '__main__':
    # print_all_Features_len()
    print_all_Bytes()
    compare_match_result()

