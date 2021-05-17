import numpy as np
import cv2
import skimage
import math
import pandas as pd

from Feature_Extraction.src.CommonFunctions import *

def computeAngle(block, minutiaeType):
    (blkRows, blkCols) = np.shape(block)
    CenterX, CenterY = (blkRows-1)/2, (blkCols-1)/2
    if(minutiaeType.lower() == 'termination'):
        sumVal = 0
        for i in range(blkRows):
            for j in range(blkCols):
                if((i == 0 or i == blkRows-1 or j == 0 or j == blkCols-1) and block[i][j] != 0):
                    angle = math.atan2((i-CenterY),(j-CenterX))
                    sumVal += 1
        if(sumVal != 1):
            return None
        return(angle)
    elif(minutiaeType.lower() == 'bifurcation'):
        (blkRows, blkCols) = np.shape(block)
        CenterX, CenterY = (blkRows - 1) / 2, (blkCols - 1) / 2
        angle = []
        sumVal = 0
        for i in range(blkRows):
            for j in range(blkCols):
                if ((i == 0 or i == blkRows - 1 or j == 0 or j == blkCols - 1) and block[i][j] != 0):
                    angle.append(math.atan2(i - CenterY, j - CenterX))
                    sumVal += 1
        if(sumVal != 3):
            return None
        minid = np.argmin([np.abs(angle[2]-angle[1]), np.abs(angle[0]-angle[2]), np.abs(angle[1]-angle[0])])
        return angle[minid]


def extractMinutiaeFeatures(skel, minutiaeTerm, minutiaeBif, center):
    minutiaeTerm = skimage.measure.label(minutiaeTerm, connectivity=2)
    RP = skimage.measure.regionprops(minutiaeTerm)
    WindowSize = 3

    FeaturesTerm = pd.DataFrame(columns=["row", "col", "angle", "class", "distance", "deltay", "deltax"])
    for i, Term in enumerate(RP):
        (row, col) = np.int16(np.round(Term['Centroid']))
        block = skel[row-WindowSize:row+WindowSize+1, col-WindowSize:col+WindowSize+1]
        angle = computeAngle(block, 'termination')
        deltay = row-center[1]
        deltax = col-center[0]
        distance = (deltax)**2 + (deltay)**2
        if angle is not None:
            FeaturesTerm.loc[i] = np.array([row, col, angle, 0, distance, deltay, deltax])

    minutiaeBif = skimage.measure.label(minutiaeBif, connectivity=2)
    RP = skimage.measure.regionprops(minutiaeBif)
    FeaturesBif = pd.DataFrame(columns=["row", "col", "angle", "class", "distance", "deltay", "deltax"])
    for i, Bif in enumerate(RP):
        (row, col) = np.int16(np.round(Bif['Centroid']))
        block = skel[row-WindowSize:row+WindowSize+1, col-WindowSize:col+WindowSize+1]
        angle = computeAngle(block, 'Bifurcation')
        deltay = row-center[1]
        deltax = col-center[0]
        distance = (deltax)**2 + (deltay)**2
        if angle is not None:
            FeaturesBif.loc[i] = np.array([row, col, angle, 1, distance, deltay, deltax])
    Features = FeaturesTerm.append(FeaturesBif)
    return Features.reset_index(drop=True)
