# -*- coding: utf-8 -*-

"""have a nice day.

@author: Khan
@contact:  
@time: 2020/8/24 13:29
@file: de.py
@desc:  
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

plt.rcParams['font.sans-serif'] = ['SimHei'] #用来显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
# sns.set()
xx=['密度','酒精度','残糖','总二氧化硫','游离二氧化硫','柠檬酸','非挥发性酸','挥发性酸','酸碱度','硫酸盐','氯化物']
x = ['金融','农业','制造业','新能源']
y = [164, 86, 126, 58]
yy=[123.3,118.3,108.6,106.5,101.8,99.9,99.8,96.5,91.2,88,83]
yy=np.array(yy)
s=yy.sum()
yy=yy/s
print(yy)
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
# sns.barplot(xx,yy)

plt.bar(xx,yy)
plt.xticks(rotation=40)
plt.savefig('res2.png')
plt.show()