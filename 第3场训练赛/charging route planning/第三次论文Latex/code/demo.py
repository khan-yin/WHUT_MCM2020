import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
# 支持保存pdf矢量图
import pandas as pd


def plot_mat(mat, save=True, name="mat"):
    plt.figure()
    plt.imshow(mat, cmap=plt.cm.hot)
    plt.colorbar()
    if save:
        pdf = PdfPages("img//{}.pdf".format(name))
        pdf.savefig()
        pdf.close()
    else:
        plt.show()
    plt.close()


def plot_map(station, save=True, pointSize=25, name="map"):
    # print(station["classID"])
    plt.figure()
    plt.scatter(station["X"], station["Y"], s=pointSize-10*station["classID"], c=station["classID"], cmap="Set1")
    plt.xlabel("X (km)")
    plt.ylabel("Y (km)")
    if save:
        pdf = PdfPages("img//{}.pdf".format(name))
        pdf.savefig()
        pdf.close()
    else:
        plt.show()
    plt.close()

