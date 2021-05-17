import pandas as pd
import numpy as np
import plot
import pickle

DATA_PATH = "data//data.xlsx"
DISTANCE01_MAT_PATH = "data//distanceMat01.txt"
DISTANCE12_MAT_PATH = "data//distanceMat12.txt"


class Data:
    def __init__(self):
        self.dataPath = DATA_PATH
        self.stationNum = [0, 0, 0]
        self.station012 = self.read_data()#[:120]
        self.station012[["id", "classID"]].astype(int)
        self.station01 = self.station012[self.station012["classID"] != 2].reset_index(drop=True)

        # try:
        #     self.distanceMat01 = np.loadtxt(DISTANCE01_MAT_PATH)
        # except IOError:
        self.distanceMat01 = self.get_distanceMat(self.station01)
        np.savetxt(DISTANCE01_MAT_PATH, self.distanceMat01)
        plot.plot_mat(self.distanceMat01, name="distanceMat01")

        # try:
        #     self.distanceMat12 = np.loadtxt(DISTANCE12_MAT_PATH)
        # except IOError:
        self.distanceMat12 = self.get_distanceMat(self.station012)
        self.distanceMat12[:, 0] = np.inf
        self.distanceMat12[0, :] = np.inf
        np.savetxt(DISTANCE12_MAT_PATH, self.distanceMat12)
        plot.plot_mat(self.distanceMat12, name="distanceMat12")

    def read_data(self):
        point = pd.read_excel(self.dataPath, sheet_name=0)
        point["classID"] = 0
        for i, c in enumerate(point["class"]):
            if c == "A":
                point.at[i, "classID"] = 0
                self.stationNum[0] += 1
                # 不要使用链式赋值: point["classID"][i]=0
                # ;SettingWithCopyWarning
            elif c.startswith("V"):
                point.at[i, "classID"] = 1
                self.stationNum[1] += 1
            elif c.startswith("P"):
                point.at[i, "classID"] = 2
                self.stationNum[2] += 1
        # print(point)
        plot.plot_map(point)
        return point

    def get_distanceMat(self, station):
        """
        计算供水站之间的距离矩阵
        :return:
        """
        distanceMat = np.zeros((len(station), len(station)))
        for i in range(distanceMat.shape[0]):
            for j in range(distanceMat.shape[1]):
                distanceMat[i, j] = np.sqrt(
                    (station["X"][i]-station["X"][j])**2 + (station["Y"][i]-station["Y"][j])**2
                )
        return distanceMat


if __name__ == '__main__':
    d = Data()

