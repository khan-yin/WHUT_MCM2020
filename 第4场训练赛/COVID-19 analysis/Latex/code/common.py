
import pandas as pd
import numpy as np
import matplotlib
matplotlib.rcParams["font.sans-serif"] = ["SimHei"]
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from matplotlib import style
style.use('seaborn-paper')

WINDOW = 7
DAYS = 150

bestParametersDict = {
    "Channel Islands": {'I0': 2, 'Q0': 3, 'S0': 26380.717689599554, 'betaTimesS0': 0.7347729066021397, 'delta': 0.010045853476576706, 'epsilon': 7.840097082522992e-05, 'eta': 0.003383864399097301, 'gamma0': 0.016848020203597488, 'gamma1': 0.16628438395216183, 'hStart': 58, 'k': 0.6920892202047502, 'lamb': 0.17208981925301345, 'mEnd': 87, 'mStart': 84, 'sigma': 0.07720585767550427, 'start': 55, 'theta': 0.12248114734032356},
    "Anhui":           {'I0': 13, 'Q0': 4, 'S0': 12338.977128653125, 'betaTimesS0': 0.45666217056059893, 'delta': 0.010045504375222879, 'epsilon': 0.04633446975602694, 'eta': 0.000545652721259959, 'gamma0': 0.018348160210604064, 'gamma1': 0.15860752405127965, 'hStart': 12, 'k': 0.6969110600952818, 'lamb': 0.2012066716037268, 'mEnd': 34, 'mStart': 19, 'sigma': 0.2932882661742646, 'start': 2, 'theta': 0.1626536380573757}
}


def save_para_excel():
    df = pd.DataFrame(bestParametersDict["Channel Islands"], index=["海峡群岛(英国)"])
    df = df.append(pd.DataFrame(bestParametersDict["Anhui"], index=["安徽省"]))
    df.T.to_excel("../cache/para.xlsx")
    print(df.T)


if __name__ == '__main__':
    save_para_excel()
