# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 18:44:22 2018

@author: Utkarsh
"""

import cv2
import numpy as np
import pandas as pd
import skimage.morphology
import skimage

from Feature_Extraction.src.CommonFunctions import *

def removeSpuriousMinutiae(Features, edge_filter, thresh):
    '''
    细节点去伪
    '''
    numPoints = len(Features)
    D = np.zeros((numPoints, numPoints))
    Features["valid"] = 1

    for i, minutiaei in Features.iterrows():
        (rowi, coli) = int(minutiaei["row"]),  int(minutiaei["col"])
        if edge_filter[rowi, coli]==0:
            Features["valid"][i] = 0
    Features = Features.loc[Features["valid"] == 1].reset_index(drop=True)

    for i, minutiaei in Features.iterrows():
        (rowi, coli) = int(minutiaei["row"]),  int(minutiaei["col"])
        for j in range(i):
            minutiaej = Features.loc[j]
            (rowj, colj) = int(minutiaej["row"]),  int(minutiaej["col"])
            if np.sqrt((rowi-rowj)**2 - (coli-colj)**2) < thresh:
                Features["valid"][i] = 0
                Features["valid"][j] = 0
    Features = Features.loc[Features["valid"] == 1]
    Features = Features.sort_values(by="distance").reset_index(drop=True)
    return Features
