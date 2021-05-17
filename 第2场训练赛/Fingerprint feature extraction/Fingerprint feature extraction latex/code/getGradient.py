
import cv2
import numpy as np
import pandas as pd
import skimage.morphology as sm
from skimage import filters
import skimage
import os
import warnings

from Feature_Extraction.src.CommonFunctions import *
from Feature_Extraction.src.getTerminationBifurcation import getTerminationBifurcation


def getGradient(img):
    SCALE = 1/2
    SOBEL_KSIZE = 5
    GUASSIAN_KSIZE = (7, 7)
    img = cv2.resize(img, (0, 0), fx=SCALE, fy=SCALE)
    cv2.imwrite(os.path.join(PROCESS_DIR, "0img.png"), img)
    gradX = cv2.Sobel(img/255., cv2.CV_16S, 1, 0, ksize=SOBEL_KSIZE)
    gradY = cv2.Sobel(img/255., cv2.CV_16S, 0, 1, ksize=SOBEL_KSIZE)

    # gradX = normalizedArr(gradX, -1,1)
    # gradY = normalizedArr(gradY, -1,1)

    dxdy = 2 * gradX*gradY + 1e-3
    dx2_dy2 = gradX**2 - gradY**2
    theta = np.arctan(dx2_dy2/dxdy)
    cv2.imwrite(os.path.join(PROCESS_DIR, "1theta.png"), normalizedArr(theta, 0, 255).astype(np.uint8))

    phix = np.cos(2*theta)
    phiy = np.sin(2*theta)

    O = .5*np.arctan(2*(phiy/phix))
    cv2.imwrite(os.path.join(PROCESS_DIR, "2imgO_before_Gaussian.png"),
                normalizedArr(O, 0, 255).astype(np.uint8))

    phix = cv2.GaussianBlur(phix, GUASSIAN_KSIZE, 15)
    phiy = cv2.GaussianBlur(phiy, GUASSIAN_KSIZE, 15)

    O = .5*np.arctan(2*(phiy/phix))
    imgO = normalizedArr(O, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(PROCESS_DIR, "3imgO_after_Gaussian.png"), imgO)

    imgOEdge = normalizedArr(filters.sobel(imgO), 0, 255)
    cv2.imwrite(os.path.join(PROCESS_DIR, "4imgOEdge.png"), imgOEdge)

    imgOEdge = sm.dilation(imgOEdge, sm.disk(2))
    imgOEdge = sm.closing(imgOEdge, sm.disk(4))
    cv2.imwrite(os.path.join(PROCESS_DIR, "5imgOEdge_closing.png"), imgOEdge)

    THRESHOLD = 200

    imgOEdge = (imgOEdge > THRESHOLD)*255
    cv2.imwrite(os.path.join(PROCESS_DIR, "6imgOEdge_01.png"), imgOEdge)
    # plt_show(imgOEdge>THRESHOLD)
    imgOSkel = sm.skeletonize(imgOEdge > THRESHOLD)*255
    cv2.imwrite(os.path.join(PROCESS_DIR, "7imgOSkel.png"), imgOSkel)

    return gradX, gradY

