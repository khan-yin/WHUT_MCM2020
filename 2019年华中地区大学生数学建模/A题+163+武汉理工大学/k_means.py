import numpy as np
import pandas as pd
from math import cos,sin,asin,sqrt,radians

# 根据经纬度计算距离
def __distance(lon1, lat1, lon2, lat2):  # 经度1，纬度1，经度2，纬度2 （十进制度数）
    """
    根据经纬度计算距离
    :param lon1: 点1经度
    :param lat1: 点1纬度
    :param lon2: 点2经度
    :param lat2: 点2纬度
    :return:distance
    """
    # 将十进制度数转化为弧度
    lon1, lat1, lon2, lat2 = map(radians, [float(lon1), float(lat1), float(lon2), float(lat2)])

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371.137  # 地球平均半径，单位为公里
    return float('%.2f' % (c * r))


# # 导入excel表举例
df = pd.read_excel('bbb.xls')
df = df[df['强度']==6]
lons = df['经度']/10
lats = df['纬度']/10
#目标距离 （风速55，强度6）
lon2 = 139.9
lat2 = 13.6
for i in range(len(df)):
    x = __distance(lons.values[i],lats.values[i],lon2,lat2)
    if(x<400):
        print("距离：",x," 日期：",df['日期'].values[i])