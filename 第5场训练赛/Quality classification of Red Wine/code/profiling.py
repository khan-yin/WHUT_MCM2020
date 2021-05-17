# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 11:58:48 2019

@author: Hawee
"""
import numpy as np
import pandas as pd
import pandas_profiling
import pickle
import time

fn = "profile.pkl"

def profile(datafile="../input/data.xlsx"):
    df = pd.read_excel(datafile).replace("N", np.nan)
    # df.fillna(-1, inplace=True)
    df.astype(np.float)
    df["quality"].astype(np.int)
    df.head()
    df.describe()
    df.info()

    start = time.time()
    profile = df.profile_report(title="data")
    profile.to_file("profile.html")
    print("profile finished, duration: {}".format(time.time()-start))

    with open(fn, 'wb') as f: # open file with write-mode
        picklestring = pickle.dump(profile, f) # serialize and save object


if __name__ == '__main__':
    profile()

