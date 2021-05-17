import pandas as pd
import numpy as np
import plot


def read_data():
    point = pd.read_excel("data//data.xls", sheet_name=0)
    plot.plot_map(point)
    return point

def summarize_data():
    point = read_data()
    print("Total requirement: {}".format(point["requirement"].sum()))
    print(point.sort_values(by="requirement"))

if __name__ == '__main__':
    summarize_data()
