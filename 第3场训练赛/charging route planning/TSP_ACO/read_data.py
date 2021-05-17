import pandas as pd
import numpy as np


def read_data():
    point = pd.read_excel("data//data.xlsx", sheet_name=0)
    return point

def summarize_data():
    point = read_data()
    print(point)

if __name__ == '__main__':
    summarize_data()
