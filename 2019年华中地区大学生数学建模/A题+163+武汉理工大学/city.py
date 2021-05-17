import requests
import pandas as pd
import json
import csv


# # 导入excel表举例
df = pd.read_excel('aaa.xls')
df0 = df[df['强度']==0]
df1 = df[df['强度']==1]
df2 = df[df['强度']==2]
df3 = df[df['强度']==3]
df4 = df[df['强度']==4]
df5 = df[df['强度']==5]
df6 = df[df['强度']==6]
df9 = df[df['强度']==9]

v01 = df0['经度'].values/10
v02 = df0['纬度'].values/10
v11 = df1['经度'].values/10
v12 = df1['纬度'].values/10
v21 = df2['经度'].values/10
v22 = df2['纬度'].values/10
v31 = df3['经度'].values/10
v32 = df3['纬度'].values/10
v41 = df4['经度'].values/10
v42 = df4['纬度'].values/10
v51 = df5['经度'].values/10
v52 = df5['纬度'].values/10
v61 = df6['经度'].values/10
v62 = df6['纬度'].values/10
v91 = df9['经度'].values/10
v92 = df9['纬度'].values/10

print(v01)
jinweidu0 = []
jinweidu1 = []
jinweidu2 = []
jinweidu3 = []
jinweidu4 = []
jinweidu5 = []
jinweidu6 = []
jinweidu9 = []
for i in range(len(df0)):
    address = str(v01[i])+','+str(v02[i])
    jinweidu0.append(address)
for i in range(len(df1)):
    address = str(v11[i])+','+str(v12[i])
    jinweidu1.append(address)
for i in range(len(df2)):
    address = str(v21[i])+','+str(v22[i])
    jinweidu2.append(address)
for i in range(len(df3)):
    address = str(v31[i])+','+str(v32[i])
    jinweidu3.append(address)
for i in range(len(df4)):
    address = str(v41[i])+','+str(v42[i])
    jinweidu4.append(address)
for i in range(len(df5)):
    address = str(v51[i])+','+str(v52[i])
    jinweidu5.append(address)
for i in range(len(df6)):
    address = str(v61[i])+','+str(v62[i])
    jinweidu6.append(address)
for i in range(len(df9)):
    address = str(v91[i])+','+str(v92[i])
    jinweidu9.append(address)


# 字典中的key值即为csv中列名
dataframe = pd.DataFrame({'jinweidu0': jinweidu0,})
# 将DataFrame存储为csv,index表示是否显示行名，default=True
dataframe.to_csv("jinweidu0.csv", index=False, sep=',')
# 字典中的key值即为csv中列名
dataframe = pd.DataFrame({'jinweidu1': jinweidu1,})
# 将DataFrame存储为csv,index表示是否显示行名，default=True
dataframe.to_csv("jinweidu1.csv", index=False, sep=',')
# 字典中的key值即为csv中列名
dataframe = pd.DataFrame({'jinweidu2': jinweidu2,})
# 将DataFrame存储为csv,index表示是否显示行名，default=True
dataframe.to_csv("jinweidu2.csv", index=False, sep=',')
dataframe = pd.DataFrame({'jinweidu3': jinweidu3,})
# 将DataFrame存储为csv,index表示是否显示行名，default=True
dataframe.to_csv("jinweidu3.csv", index=False, sep=',')
dataframe = pd.DataFrame({'jinweidu4': jinweidu4,})
# 将DataFrame存储为csv,index表示是否显示行名，default=True
dataframe.to_csv("jinweidu4.csv", index=False, sep=',')
dataframe = pd.DataFrame({'jinweidu5': jinweidu5,})
# 将DataFrame存储为csv,index表示是否显示行名，default=True
dataframe.to_csv("jinweidu5.csv", index=False, sep=',')
dataframe = pd.DataFrame({'jinweidu6': jinweidu6,})
# 将DataFrame存储为csv,index表示是否显示行名，default=True
dataframe.to_csv("jinweidu6.csv", index=False, sep=',')
dataframe = pd.DataFrame({'jinweidu9': jinweidu9,})
# 将DataFrame存储为csv,index表示是否显示行名，default=True
dataframe.to_csv("jinweidu9.csv", index=False, sep=',')