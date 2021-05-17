import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import numpy as np
import os

def plot_cluster_map(point, name, save=True, pointSize=50):
    plt.figure()
    plt.scatter(point["X"], point["Y"], s=pointSize*point["requirement"], c=point["cluster"])
    plt.scatter([10.], [10.], s=[50], c="r")
    plt.xlabel("X (km)")
    plt.ylabel("Y (km)")
    if save:
        pdf = PdfPages("img//cluster_map_"+name+".pdf")
        pdf.savefig()
        pdf.close()
    else:
        plt.show()
    plt.close()


def plot_mat(mat, save=True, name="mat"):
    plt.figure()
    plt.imshow(mat, cmap=plt.cm.hot,
               # vmin=0, vmax=1
               )
    plt.colorbar()
    if save:
        pdf = PdfPages("img//"+name+".pdf")
        pdf.savefig()
        pdf.close()
    else:
        plt.show()
    plt.close()

def plot_TS_from_excel(file="ansmer1.xlsx", pointSize=5):
    # path = [0, 2, 1, 9, 8, 12, 10, 13, 16, 27, 15, 14, 11, 6, 7, 20, 17, 19, 18, 25, 26, 29, 21, 23, 24, 28, 22, 4, 3, 5, 0]
    path = list(pd.read_excel(os.path.join("data", file))["节点"])
    point = pd.read_excel("data//data.xlsx", sheet_name=0)
    df = point[["latitude", "longitude"]]
    best_cities = np.zeros((len(path), 2))
    for i, city_id in enumerate(path):
        best_cities[i] = np.array(df.loc[path[i]])

    plt.figure()
    plt.scatter(point["latitude"], point["longitude"], s=pointSize * point["TDP"], c="y")
    plt.plot(best_cities[:, 0], best_cities[:, 1], dashes=[6, 2])
    plt.scatter(point["latitude"][0], point["longitude"][0], s=[pointSize]*10, c="r")
    plt.xlabel("latitude")
    plt.ylabel("longitude")
    plt.tight_layout()
    pdf = PdfPages("img//path_TS"+file+".pdf")
    pdf.savefig()
    pdf.close()
    plt.close()


# 单旅行社数据可视化展示
def display_TS(point, name, ants_info, best_path, best_cities, pointSize=5):
    plt.figure()
    plt.plot(ants_info, 'g.')
    plt.plot(best_path, 'r-', label='history_best')
    plt.xlabel('Iteration')
    plt.ylabel('length')
    plt.legend()
    plt.savefig('img//optimize_process.png', dpi=500)

    plt.figure()
    plt.scatter(point["latitude"], point["longitude"], s=pointSize * point["TDP"], c="y")
    plt.plot(best_cities[:, 1], best_cities[:, 0], dashes=[6, 2])
    plt.scatter(point["latitude"][0], point["longitude"][0], s=[pointSize]*10, c="r")
    plt.xlabel("latitude")
    plt.ylabel("longitude")
    plt.tight_layout()
    pdf = PdfPages("img//path_TS"+name+".pdf")
    pdf.savefig()
    pdf.close()
    plt.close()


# 多旅行社数据可视化展示
def display_mutiTS(point, name, ACO_model_ls, pointSize=50):
    plt.figure()
    plt.scatter(point["X"], point["Y"], s=pointSize*point["requirement"], c=point["cluster"])
    plt.scatter([10.], [10.], s=[pointSize], c="r")
    for cluster, ACO_model in enumerate(ACO_model_ls):
        plt.plot(ACO_model.best_cities[:, 0], ACO_model.best_cities[:, 1], dashes=[6, 2])

    plt.xlabel("X (km)")
    plt.ylabel("Y (km)")
    pdf = PdfPages("img//path_mutiTS"+name+".pdf")
    pdf.savefig()
    pdf.close()
    plt.close()

if __name__ == '__main__':
    plot_TS_from_excel()
