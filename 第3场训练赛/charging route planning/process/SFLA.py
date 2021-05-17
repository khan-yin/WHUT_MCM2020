# -*- coding: utf-8 -*-

"""have a nice day.

@author: Khan
@contact:  
@time: 2020/8/14 9:59
@file: SFLA.py
@desc:  
"""

import matplotlib.pyplot as plt
import numpy as np
import random
import operator
import pandas as pd


# 适应度函数
def F(plan):
    sum = []
    for i in range(d):
        sum.append(0)
    for i in range(d):
        if (plan[i] < 0 | plan[i] > nodes.__len__()):
            return 100
        sum[plan[i]] += round(tasks[i] / nodes[plan[i]], 3)
    sum.sort(reverse=True)
    return sum[0]


# 初始化
tasks = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # 任务长度
nodes = [1, 2, 3]  # 节点
# SFLA参数
N = 100
n = 10
d = tasks.__len__()
m = N // n
L = 5
G = 100
D_max = 10
P = []

# step1 生成蛙群
for i in range(N):
    t = [[], 0]
    for j in range(d):
        t[0].append(random.randint(0, 2))
    t[0] = np.array(t[0])
    t[1] = F(t[0])
    P.append(t)
# sorted(P,key=lambda P:P[1])
P.sort(key=operator.itemgetter(1))
Xg = P[0]#最好的青蛙

for k in range(G):
    # step2 划分子群
    M = []
    for i in range(m):
        M.append([])
    for i in range(N):
        M[i % m].append(P[i])
    Xb = []#性能好的青蛙
    Xw = []#性能差的青蛙
    for i in range(m):
        Xb.append(M[i][0])
        Xw.append(M[i][M[i].__len__() - 1])

    # step3 局部搜索
    for i in range(m):
        for j in range(L):
            D = random.randint(0, 1) * (Xb[i][0] - Xw[i][0])
            temp = Xw[i][0] + D
            if (F(temp) < F(Xw[i][0])):
                f = 0
                Xw[i][0] = temp
                Xw[i][1] = F(temp)
                M[i][M[i].__len__() - 1] = Xw[i]
            else:
                Xb[i] = Xg
                f = 2
            if (f == 2):
                t = [[], 0]
                for j in range(d):
                    t[0].append(random.randint(0, 2))
                t[0] = np.array(t[0])
                t[1] = F(t[0])
                Xw[i] = t

    P = []
    for i in range(m):
        for j in range(n):
            P.append(M[i][j])

    # sorted(P,key=lambda P:P[1])
    P.sort(key=operator.itemgetter(1))
    Xg = P[0]
    x = []
    y = []
    for i in range(P.__len__()):
        x.append(k)
        y.append(P[i][1])
    plt.scatter(x, y, s=5)

print(P[0])
plt.show()

