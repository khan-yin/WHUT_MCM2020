import numpy as np
import pandas as pd
import skimage
import os
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import skimage
import imutils


PROCESS_DIR = "../process"
RESULT_DIR = "../result"
FEATURE_DIR = "../featuresData"
DATA_DIR = '../../../data'
IMG_SRC_DIR = '../../../data/img'

NUM_PIC = 16
NUM_POINTS = 10

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


def plot_mat(mat, save=True, name="mat"):
    plt.figure()
    plt.imshow(mat, cmap=plt.cm.hot)
    plt.colorbar()
    if save:
        pdf = PdfPages("../{}.pdf".format(name))
        pdf.savefig()
        pdf.close()
    else:
        plt.show()
    plt.close()

def External_ellipse(edge_filter, mask):
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

    return ellipse


def ShowResults(skel, Features, edge_filter, center, img_id, show=False):
    # Features["angle"] /= (180/np.pi)
    arrowsLen = 12
    arrowsWidth = 2
    radius = 4
    (rows, cols) = skel.shape
    DispImg = np.zeros((rows, cols, 3), np.uint8)
    DispImg[:, :, 0] = skel + .03 * edge_filter
    DispImg[:, :, 1] = skel
    DispImg[:, :, 2] = skel + .03 * edge_filter

    (rr, cc) = skimage.draw.circle(center[1], center[0], 7)
    skimage.draw.set_color(DispImg, (rr, cc), (0, 215, 255))

    for i, minutiae in Features.loc[Features["class"] == 0].iterrows():
        c = (255, 0, 0)
        (row, col) = int(minutiae["row"]),  int(minutiae["col"])
        (row_, col_) = int(minutiae["row"]+arrowsLen*np.sin(minutiae["angle"])), \
                       int(minutiae["col"]+arrowsLen*np.cos(minutiae["angle"]))
        (rr, cc) = skimage.draw.circle(row, col, radius)
        skimage.draw.set_color(DispImg, (rr, cc), c)
        cv2.line(DispImg, (col, row), (col_, row_), c, arrowsWidth)

    for i, minutiae in Features.loc[Features["class"] == 1].iterrows():
        c = (0, 0, 255)
        (row, col) = int(minutiae["row"]),  int(minutiae["col"])
        (row_, col_) = int(minutiae["row"]+arrowsLen*np.sin(minutiae["angle"])), \
                       int(minutiae["col"]+arrowsLen*np.cos(minutiae["angle"]))
        (rr, cc) = skimage.draw.circle(row, col, radius)
        skimage.draw.set_color(DispImg, (rr, cc), c)
        cv2.line(DispImg, (col, row), (col_, row_), c, arrowsWidth)

    img_name = "{:02d}result.png".format(img_id)
    cv2.imwrite(os.path.join(RESULT_DIR, img_name), DispImg)
    if show:
        plt_show(DispImg, name=img_name)


def normalizedArr(data, newMinValue=0, newMaxValue=1):
    minVal = np.amin(data)
    maxVal = np.amax(data)
    normalized = (data-minVal)*newMaxValue/(maxVal-minVal) + newMinValue
    return normalized

def getPrettyBytesStr(bytes):
    return ''.join(['0x%02x ' % b for b in bytes])

def fromPrettyBytesStr(prettyBytesStr):
    pass

def getPrettyFeaturesBytes(Features: pd.DataFrame):
    locArr = np.array(Features[["deltay", "deltax", "angle"]])

    Features[["deltay", "deltax", "angle", "class"]].astype(int).to_excel("../Features.xlsx")
    locArr = locArr.astype(np.int16)
    locBytes = locArr.tostring()

    # print(locBytes)
    # print(len(locBytes))
    # print(type(locBytes))
    # print(np.fromstring(locBytes, dtype=np.int16).reshape((-1, 3)))

    classArr = np.array(Features["class"]).astype(np.uint8)
    classBytes = classArr.tostring()
    # print(classBytes)
    # print(len(classBytes))
    # print(type(classBytes))
    # print(np.fromstring(classBytes, dtype=np.uint8))

    prettyBytesStr = getPrettyBytesStr(locBytes+classBytes)
    return prettyBytesStr


