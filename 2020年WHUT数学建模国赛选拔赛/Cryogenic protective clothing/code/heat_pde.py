from common import *
import plot

import numpy as np
from tqdm import tqdm


class HeatEquation():
    def __init__(self, timespan=1000, hout=6, hin=70/3.5,
                 th_air=0.7 * 1e-3, tk_skin=0.4 * 1e-3, tk_out=0.3 * 1e-3, tk_f=0.4 * 1e-3, tk_cloth=0.7 * 1e-3,
                 dt=0.02, dsc_scale=1):

        self.d = [0.0527, 0.06, 0.068, 0.023, 0.003, ]  # 热导率W/(m·K)
        self.c = [5463, 2400, 4803.8, 1000, 3600, ]  # 比热容
        self.rho = [300, 552.3, 208, 129.3, 1000, ]  # 密度

        self.thickness = np.array([tk_out, tk_f, tk_cloth, th_air, tk_skin])
        self.dxs = self.thickness * 2e-1
        self.hout, self.hin = hout, hin

        self.Tin, self.Tout = 37, -40
        self.dt = dt
        self.timestep = int(timespan/self.dt)
        self.timestep_per_sec = int(1/self.dt)
        self.xstep_per_layer = int(1/2e-1)
        self.dxs = self.thickness*2e-1

        self.LEN = []
        self.LEN.append(int(self.thickness[0]/self.dxs[0]))
        for i in range(1, 5):
            self.LEN.append(self.LEN[-1]+self.thickness[i]/self.dxs[i])
        self.LEN = np.array(self.LEN).astype(int)
        # print(self.LEN)
        self.T = np.zeros((self.timestep+1, self.LEN[-1]+1), dtype=np.float64)
        self.T[0, :self.LEN[-2]] = 26
        self.T[0, self.LEN[-2]:] = self.Tin

        self.price = 10 +\
            self.thickness[0] * self.rho[0] * 300 +\
            self.thickness[2] * self.rho[2] * 1000

        self.dsc_scale = dsc_scale
        self.data = pd.read_excel("../data/data.xlsx")
        self.data = np.array(self.data)

    def DSC(self, T):
        if T < 14.5 or T > 26.018:
            return 0
        Tid = (self.data[:, 0] < T).sum() - 1
        return self.data[Tid, 1] * 1e3 * self.dsc_scale

    def solve(self, name='T'):
        for n in tqdm(range(self.timestep)):
            self.step(n, self.hout, self.hin)
            # if np.isnan(self.T[n+1]).sum()>0:
            #     raise ValueError("has nan in solve")
        self.Tps = self.T[::self.timestep_per_sec]
        self.to_cache("../cache/{}.xlsx".format(name))

    def step(self, n, h1, h2):
        scale=0.01
        T = self.T
        Tin = self.Tin
        Tout = self.Tout
        dxs = self.dxs
        dt = self.dt
        c = self.c
        lam = np.array(self.d)
        rho = self.rho

        j = 0
        bc = (h1*(Tout-T[n, 0]) + lam[j] * (T[n, 1]-T[n, 0])/dxs[j]) * dt / (0.5 * dxs[j] * rho[j] * c[j]) + T[n, 0]
        T[n+1, 0] = bc
        for i in range(1, self.LEN[j]):
            delta = lam[j]*(T[n, i+1]-2*T[n, i]+T[n, i-1])*dt / (dxs[j]**2 * rho[j] * c[j])
            if np.abs(delta) < scale * T[n, i]:
                T[n+1, i] = delta + T[n, i]
                if T[n+1, i] >= Tin and T[n, i] >= Tin:
                    T[n+1, i:] = Tin
                    T[n+1, i] = T[n, i]
                    return
            elif T[n, i] <= T[n, i - 1]:
                T[n, i] = T[n, i - 1]
                T[n+1, i] = T[n, i]
            else:
                T[n+1, i] = T[n, i]

        i = self.LEN[j]
        T[n+1, self.LEN[j]] = (lam[j+1]*(T[n, i+1]-T[n, i])/dxs[j] + lam[j]*(T[n, i-1]-T[n, i])/dxs[j+1])*\
                              dt/(0.5*(dxs[j]*rho[j]*c[j]+dxs[j+1]*rho[j+1]*c[j+1]))+T[n,i]

        for j in [1, 2, 3]:
            for i in range(self.LEN[j-1]+1, self.LEN[j]):
                if j != 1:
                    delta = lam[j] * (T[n, i + 1] - 2 * T[n, i] + T[n, i - 1]) * dt / (dxs[j]**2 * rho[j] * c[j])
                else:
                    delta = ((lam[j] * (T[n, i + 1] - 2 * T[n, i] + T[n, i - 1])/ (dxs[j])
                              - self.DSC(T[n, i])/c[j])) * dt / (rho[j] * c[j])/ (dxs[j])

                if np.abs(delta) < scale * T[n, i]:
                    T[n + 1, i] = delta + T[n, i]
                    if T[n+1, i] >= Tin and T[n, i] >= Tin:
                        T[n+1, i:] = Tin
                        T[n+1, i] = T[n, i]
                        return
                elif T[n, i] < T[n, i - 1]:
                    T[n, i] = T[n, i - 1]
                    T[n+1, i] = T[n, i]
                else:
                    T[n + 1, i] = T[n, i]
            i = self.LEN[j]
            T[n+1, self.LEN[j]] = (lam[j+1]*(T[n, i+1]-T[n, i])/dxs[j] + lam[j]*(T[n, i-1]-T[n, i])/dxs[j+1])*\
                                  dt/(0.5*(dxs[j]*rho[j]*c[j]+dxs[j+1]*rho[j+1]*c[j+1]))+T[n,i]

        j = 4
        for i in range(self.LEN[j-1]+1, self.LEN[j]):
            delta = lam[j]*(T[n, i+1]-2*T[n, i]+T[n, i-1])*dt  / (dxs[j]**2 * rho[j] * c[j])
            if np.abs(delta) < scale * T[n, i]:
                T[n+1, i] = delta + T[n, i]
                if T[n+1, i] >= Tin and T[n, i] >= Tin:
                    T[n+1, i:] = Tin
                    T[n+1, i] = T[n, i]
                    return
            elif T[n, i] < T[n, i - 1]:
                T[n, i] = T[n, i - 1]
                T[n+1, i] = T[n, i]
            else:
                T[n+1, i] = T[n, i]
        bc = (h2*(Tin-T[n, -1]) + lam[j] * (T[n, -2]-T[n, -1])/dxs[j]) * dt / (0.5 * dxs[j] * rho[j] * c[j]) + T[n, -1]

        T[n+1, self.LEN[j]] = bc
        T[n+1] = (T[n+1] > 37)*37. + (T[n+1] <= 37)*T[n+1]

    def load_cache(self, file):
        return np.array(pd.read_excel(file))[:, 1:]
    def to_cache(self, file):
        return pd.DataFrame(self.Tps).to_excel(file)

    def time_before(self, T):
        return (self.T[:, self.LEN[-2]] > T).sum() * self.dt

    def plot_T_surface(self, use_cache=None, name=''):
        if use_cache is not None:
            self.Tps = self.load_cache(use_cache)
        plot.plot_T_surface(self.Tps, name=name)

    def plot_T_slice(self, emp=None, delta=200, use_cache=None, name=''):
        if use_cache is not None:
            self.Tps = self.load_cache(use_cache)
        plot.plot_T_slice(self.Tps, emp, deltaT=delta, name=name)

    def plot_x_slice(self, emp=None, use_cache=None, name=''):
        if use_cache is not None:
            self.Tps = self.load_cache(use_cache)
        plot.plot_x_slice(self.Tps, self.LEN, emp, name=name)

    def get_price(self, tk_out, tk_cloth):
        price = 10 +\
            tk_out * self.rho[0] * 300 +\
            tk_cloth * self.rho[2] * 1000

    def get_delta_tk(self):
        i = 0
        tk_out0 = self.thickness[0]
        tk_cloth0 = self.thickness[2]

        he = HeatEquation(600)
        he.solve("tk_out={}, tk_cloth={:.4f}".format(tk_out0, tk_cloth0))
        result = he.time_before(15)
        results = [result]
        tk_outs = [tk_out0]
        tk_cloths = [tk_cloth0]
        msg = "i = {}, tk_out={}, tk_cloth={:.4f}, time={}".format(i, tk_out0, tk_cloth0, result)
        print(msg)
        # he.plot_T_slice(name=msg)
        i = 1
        while True:
            tk_out = tk_out0+i*0.0003
            tk_cloth = (1.5 * self.price - 10 - tk_out*self.rho[0]*300) / (self.rho[2] * 1000)
            if tk_cloth <= tk_cloth0:
                break
            tk_outs.append(tk_out)
            tk_cloths.append(tk_cloth)
            he = HeatEquation(300, tk_out=tk_out, tk_cloth=tk_cloth)
            he.solve("tk_out={}, tk_cloth={:.4f}".format(tk_out, tk_cloth))
            result = he.time_before(15)
            results.append(result)
            msg = "i = {}, tk_out={}, tk_cloth={:.4f}, time={}".format(i, tk_out, tk_cloth, result)
            print(msg)
            # he.plot_T_slice(name=msg)
            i += 1
        results = 557 - np.array(results) +300
        results[0] = 557.04
        print(tk_outs, tk_cloths)
        print(results)
        plot.plot_prob3(tk_outs[1:], results[1:], tk_cloths[1:])


