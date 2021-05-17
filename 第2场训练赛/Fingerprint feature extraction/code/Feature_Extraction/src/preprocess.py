
import cv2
import numpy as np
import skimage.morphology as sm
import skimage
import os

from Feature_Extraction.src.CommonFunctions import *

def enhance_dir(path, savepath):
    for imgfile in os.listdir(path):
        img_id = int(imgfile.rstrip(".tif"))
        print(imgfile)
        img = cv2.imread(os.path.join(path, imgfile), 0)
        # 获取二值化图像
        img_01 = np.uint8(img > 128)
        # 未经细化的二值图像
        mask = img_01 * 255
        edge_filter = skimage.morphology.closing(mask, sm.disk(20))
        edge_filter = skimage.morphology.erosion(edge_filter, sm.disk(5))
        # plt_show(foregroundArea)
        ellipse = External_ellipse(edge_filter, mask)
        center, axes, angle = ellipse

        if angle > 90:
            angle -= 180
        # print(angle)
        matRotate = cv2.getRotationMatrix2D(center, angle, 1.)
        rotated = cv2.warpAffine(img, matRotate, (mask.shape[1],mask.shape[0]))
        cv2.imwrite(os.path.join(savepath, imgfile), rotated)

        maskBGR = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        rotatedBGR = cv2.cvtColor(rotated, cv2.COLOR_GRAY2BGR)
        cv2.ellipse(maskBGR, ellipse, color=(0, 0, 255), thickness=2)
        cat = np.concatenate((maskBGR, rotatedBGR), axis=1)
        cv2.imwrite(os.path.join(RESULT_DIR, "{:02d}rotate.png".format(img_id)), cat)




if __name__ == '__main__':
    enhance_dir("../enhanced", "../rotated")
