

from matplotlib import pylab
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import pandas as pd

import seaborn as sns
import numpy as np
from tqdm import tqdm

class HeatEquation:
    def __init__(self, timespan=1000):
        self.d = [0.37, 26.24e-3, 0.0527, 0.0068, 0.06]# 热导率W/(m·K)
        self.c = [3720, 1.003, 5463.2, 4803.8, 2400]
        self.rho = [1020, 1.9, 300, 208, 552.3]
        self.thickness = np.array([2, 2, .3, .7, .4])
        self.Tin = 37
        self.Tout = -40
        self.dxs = [2e-1, 2e-1,3e-2,7e-2, 4e-2]
        self.dt = 1
        self.LEN = []
        self.LEN.append(int(self.thickness[0]/self.dxs[0]))
        for i in range(1, 5):
            self.LEN.append(self.LEN[-1]+self.thickness[i]/self.dxs[i])
        self.LEN = np.array(self.LEN).astype(int)
        self.timestep = int(timespan/self.dt)
        self.T = np.zeros((self.timestep+1, self.LEN[-1]+1),dtype=np.float64)
        self.T[0] = self.Tin
        # self.T[0, 15]=36.9999884416089
        # self.T[0, 16] = 37.0000092547697

    def solve(self):
        for n in (range(self.timestep)):
            self.step(n, 3, 3)
            if any(np.isnan(self.T[n + 1])):
                print("has nan on step {}".format(n))
                break
        return n

    def step(self, n, h1, h2):
        T = self.T
        Tin = self.Tin
        Tout = self.Tout
        dxs = self.dxs
        dt = self.dt
        c = self.c
        lam = np.array(self.d)*1e3
        rho = self.rho

        j = 0
        bc = (h1*(Tout-T[n, 0]) - lam[j] * (T[n, 0]-T[n, 1])/dxs[j]) * dt / (0.5 * dxs[j] * rho[j] * c[j]) + T[n, 0]
        T[n+1, 0] = bc
        for i in range(1, self.LEN[j]):
            T[n+1, i] = lam[j]*(T[n, i+1]-2*T[n, i]+T[n, i-1])*dt / dxs[j] / (dxs[j]*rho[j]*c[j]) + T[n, i]
        i = self.LEN[j]
        T[n+1, self.LEN[j]] = (lam[j+1]*(T[n, i+1]-T[n, i])/dxs[j] + lam[j]*(T[n, i-1]-T[n, i])/dxs[j+1])*\
                              dt/(0.5*(dxs[j]*rho[j]*c[j]+dxs[j+1]*rho[j+1]*c[j+1]))+T[n,i]

        for j in [1,2,3]:
            for i in range(self.LEN[j-1]+1, self.LEN[j]):
                T[n+1, i] = lam[j]*(T[n, i+1]-2*T[n, i]+T[n, i-1])*dt / dxs[j] / (dxs[j]*rho[j]*c[j]) + T[n, i]
            i = self.LEN[j]
            T[n+1, self.LEN[j]] = (lam[j+1]*(T[n, i+1]-T[n, i])/dxs[j] + lam[j]*(T[n, i-1]-T[n, i])/dxs[j+1])*\
                                  dt/(0.5*(dxs[j]*rho[j]*c[j]+dxs[j+1]*rho[j+1]*c[j+1]))+T[n,i]

        j = 4
        for i in range(self.LEN[j-1]+1, self.LEN[j]):
            T[n+1, i] = lam[j]*(T[n, i+1]-2*T[n, i]+T[n, i-1])*dt / dxs[j] / (dxs[j]*rho[j]*c[j]) + T[n, i]
        bc = (-h2*(T[n, -1]-Tin) + lam[j] * (T[n, -2]-T[n, -1])/dxs[j]) * dt / (0.5 * dxs[j] * rho[j] * c[j]) + T[n, -1]
        if n == 240:
            pass
        T[n+1, self.LEN[j]] = bc



if __name__ == '__main__':
    he = HeatEquation()
    n = he.solve()
    print(he.T[:n, -1])
    pd.DataFrame(he.T).to_excel("T.xlsx")
