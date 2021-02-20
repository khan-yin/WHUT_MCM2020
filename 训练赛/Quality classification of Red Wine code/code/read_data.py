
import numpy as np
import pandas as pd

from imblearn.over_sampling import ADASYN, SMOTE

def read_data(over_sample=False):
    df = pd.read_csv("./data/final.csv").replace("N", np.nan)
    test_features = pd.read_excel("./data/final2.xlsx").replace("N", np.nan)
    test_features = test_features[['Latitude','Longitude','lenth','note emotion','lab emotion','color','bee lenth','CNNpossibility','Season']]
    tgt = df["Lab Status"]
    # df = pd.read_excel("../data/final.csv", sheet_name=0).replace("N", np.nan)
    # test_features = pd.read_excel("../data/final.csv", sheet_name=1).replace("N", np.nan)
    # tgt = df["quality"]-3
    df.drop(["Lab Status"], axis=1, inplace=True)
    # 缺失值的计算
    # df.apply(lambda col: col.fillna(col.median()), axis=0)
    df.fillna(df.mean(), inplace=True)
    df = (df-df.mean())/(df.std())
    assert not np.isnan(df).any().any()
    cols = []
    for i in range(len(df.columns)):
        cols.append(df.columns[i].replace(" ", "_"))
    df.columns = cols
    test_features.columns = cols
    test_features = (test_features-test_features.mean())/(test_features.std())
    if over_sample:
        smote = SMOTE(k_neighbors=4)
        df_os, tgt_os = smote.fit_sample(df, tgt)
        df_os = pd.DataFrame(df_os, columns=df.columns)
        # tgt_os = pd.DataFrame(tgt_os, columns=tgt.columns)
        return df_os, tgt_os, test_features
    else:
        print(df.shape)
        return df, tgt, test_features


def read_winequality(over_sample=False):
    df = pd.read_excel("../data/wine quality-white.xls")
    print(df.head())
    print(df.describe())



if __name__ == '__main__':
    # df, tgt, test_features = read_data()
    # print(df)
    # print(tgt)
    read_winequality()
