
import cv2
import numpy as np
import skimage.morphology as sm
import skimage
import os

from Feature_Extraction.src.CommonFunctions import plt_show, External_ellipse

def enhance_dir(path, savepath):
    for imgfile in os.listdir(path):
        print(imgfile)
        img = cv2.imread(os.path.join(path, imgfile), 0)
        # 获取二值化图像
        img_01 = np.uint8(img > 128)
        # 未经细化的二值图像
        mask = img_01 * 255
        foregroundArea = skimage.morphology.closing(mask, sm.disk(20))
        foregroundArea = skimage.morphology.erosion(foregroundArea, sm.disk(5))
        # plt_show(foregroundArea)
        center, axes, angle = External_ellipse(foregroundArea, mask)
        if angle > 90:
            angle -= 180
        print(angle)
        matRotate = cv2.getRotationMatrix2D(center, angle, 1.)
        rotated = cv2.warpAffine(img, matRotate, (mask.shape[1],mask.shape[0]))
        cv2.imwrite(os.path.join(savepath, imgfile), rotated)


if __name__ == '__main__':
    enhance_dir("../enhanced", "../rotated")
