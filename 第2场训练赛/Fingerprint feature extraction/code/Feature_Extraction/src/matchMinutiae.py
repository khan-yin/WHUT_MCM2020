
import numpy as np
import pandas as pd

from Feature_Extraction.src.hungarian import Hungarian
from sklearn.preprocessing import MinMaxScaler

from Feature_Extraction.src.CommonFunctions import *

def getSimilarity(Features1, Features2):
    featureNum = len(Features1)
    Features1Arr = np.array(Features1[["deltay", "deltax", "angle"]])
    class1Arr = np.array(Features1["class"])
    Features2Arr = np.array(Features2[["deltay", "deltax", "angle"]])
    class2Arr = np.array(Features2["class"])

    mm = MinMaxScaler((-1,1))
    # 将3个特征"deltay", "deltax", "angle"标准化到区间[0, 1]
    Features1Arr = mm.fit_transform(Features1Arr)
    Features2Arr = mm.fit_transform(Features2Arr)

    # 每两个特征点的匹配损失
    profitMatrix = np.zeros((featureNum, featureNum))
    for i in range(featureNum):
        locProfitArr = np.exp(-np.sum((Features2Arr-Features1Arr[i])**2, axis=1))
        locProfitArr *= (class1Arr[i] == class2Arr)
        neighborProfitArr = np.zeros_like(locProfitArr)
        for j in range(featureNum):
            dist1Arr = np.sum((Features1Arr-Features1Arr[i])**2, axis=1)
            neighbor1Id = np.argsort(dist1Arr)[1]
            dist2Arr = np.sum((Features2Arr-Features2Arr[i])**2, axis=1)
            neighbor2Id = np.argsort(dist2Arr)[1]
            if class1Arr[neighbor1Id] != class2Arr[neighbor2Id]:
                continue
            neighborProfitArr[j] += np.exp(
                -np.sum((Features1Arr[neighbor1Id]-Features1Arr[neighbor2Id])**2)
            )
        profitMatrix[i] = locProfitArr + neighborProfitArr
    # print(profitMatrix)
    hungarian = Hungarian()
    hungarian.calculate(profitMatrix, is_profit_matrix=True)
    result = hungarian.get_results()
    total_potential = hungarian.get_total_potential()/(2*featureNum)
    print(result)
    print("total_profit: {}".format(total_potential))
    return total_potential


