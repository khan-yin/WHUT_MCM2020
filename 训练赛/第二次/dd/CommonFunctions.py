import numpy as np
import skimage
import os
import cv2
import matplotlib.pyplot as plt
import skimage.morphology as sm
import skimage
import imutils


PROCESS_DIR = "../process"
RESULT_DIR = "../result"
DATA_DIR = '../../../data'
IMG_SRC_DIR = '../../../data/img'

def plt_show(img,name=None, ax=None):
    if ax == None:
        fig=plt.figure()
        ax = fig.add_subplot(111)
    if len(img.shape) == 3:
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        #opencv读取的格式默认是BGR
    else:
        ax.imshow(img, cmap='gray')
    ax.axis("off")
    ax.set_title(name)
    plt.show()

def External_ellipse(edge_filter, mask, show=False):
    contours = cv2.findContours(edge_filter, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = contours[0] if imutils.is_cv2() else contours[1]  # 用imutils来判断是opencv是2还是2+
    cnt = cnts[0]
    """ellipse为三元组(center, axes, angle)
    center：椭圆中心点坐标
    axes：椭圆尺寸（即长短轴）
    angle：旋转角度（顺时针方向）
    """
    ellipse = cv2.fitEllipse(cnt)
    # print(ellipse)
    maskBGR = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    cv2.ellipse(maskBGR, ellipse, color=(255, 255, 0), thickness=2)
    if show:
        plt_show(maskBGR)
    return ellipse


def ShowResults(skel, Features, edge_filter, center, img_name, show=False):
    (rows, cols) = skel.shape
    DispImg = np.zeros((rows, cols, 3), np.uint8)
    DispImg[:, :, 0] = skel + .03 * edge_filter
    DispImg[:, :, 1] = skel
    DispImg[:, :, 2] = skel + .03 * edge_filter

    (rr, cc) = skimage.draw.circle(center[1], center[0], 5)
    skimage.draw.set_color(DispImg, (rr, cc), (0, 215, 255))

    for i, minutiae in Features.loc[Features["class"] == 0].iterrows():
        (row, col) = int(minutiae["row"]),  int(minutiae["col"])
        (rr, cc) = skimage.draw.circle_perimeter(row, col, 3)
        skimage.draw.set_color(DispImg, (rr, cc), (255, 0, 0))

    for i, minutiae in Features.loc[Features["class"] == 1].iterrows():
        (row, col) = int(minutiae["row"]),  int(minutiae["col"])
        (rr, cc) = skimage.draw.circle_perimeter(row, col, 3)
        skimage.draw.set_color(DispImg, (rr, cc), (0, 0, 255))
    # for idx, i in enumerate(RP):
    #     (row, col) = np.int16(np.round(i['Centroid']))
    #     (rr, cc) = skimage.draw.circle_perimeter(row, col, 3)
    #     skimage.draw.set_color(DispImg, (rr, cc), (0, 0, 255))

    cv2.imwrite(os.path.join(RESULT_DIR, img_name), DispImg)
    if show:
        plt_show(DispImg, name=img_name)
