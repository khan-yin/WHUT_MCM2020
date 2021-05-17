
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.rcParams["font.sans-serif"] = ["SimHei"]
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from matplotlib import style
style.use('seaborn-paper')

warnings.filterwarnings("ignore")

WINDOW = 7
DAYS = 150

bestParametersDict = {
    # "0.35": {'colsample_bytree': 0.7477972829697044, 'gamma': 0.18755854462817376, 'learning_rate': 0.33147590418072514, 'max_depth': 7, 'min_child_weight': 3, 'n_estimators': 56, 'over_sample': 0, 'regression': 0},
    # "0.346": {'colsample_bytree': 0.829197705656825, 'gamma': 0.12982749219299972, 'learning_rate': 0.27200281953968064, 'max_depth': 8, 'min_child_weight': 1, 'n_estimators': 54, 'over_sample': 0, 'regression': 1},
    "0.316": {'colsample_bytree': 0.60262321330833, 'gamma': 0.20462377337987994, 'learning_rate': 0.3980428446052504, 'max_depth': 13, 'min_child_weight': 1, 'n_estimators': 56,
              'over_sample': 0, 'regression': 0
              },
    # "0.104":{'colsample_bytree': 0.601809606772758, 'gamma': 0.16651628749041883, 'learning_rate': 0.3517747514399629, 'max_depth': 10, 'min_child_weight': 1, 'n_estimators': 61, 'over_sample': 0, 'regression': 0},

}


def save_para_excel():
    df = pd.DataFrame(bestParametersDict["0.316"], index=["XGBoost超参数"])
    df.T.to_excel("../cache/para.xlsx")
    print(df.T)


if __name__ == '__main__':
    save_para_excel()
