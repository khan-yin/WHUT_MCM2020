# -*- coding: utf-8 -*-

"""have a nice day.

@author: Khan
@contact:  
@time: 2020/7/12 15:04
@file: GA.py
@desc:  
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DNA_SIZE=10          #编码DNA的长度
POP_SIZE=100         #初始化种群的大小
CROSS_RATE=0.8       #DNA交叉概率
MUTATION_RATE=0.003  #DNA变异概率
N_GENERATIONS=200    #迭代次数
X_BOUND=[0,5]        #x upper and lower bounds x的区间

def F(x):
    return np.sin(10*x)*x+np.cos(2*x)*x


def get_fitness(pred):
    return pred+1e-3-np.min(pred)#获得适应值，这里是与min比较，越大则适应度越大，便于我们寻找最大值为了防止结果为0特意加了一个较小的数来作为偏正

def translateDNA(pop):#DNA的二进制序列映射到x范围内
    return pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2**DNA_SIZE-1) * (X_BOUND[1]-X_BOUND[0])

def select(pop,fitness):

    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True,
                           p=fitness / fitness.sum())
    #我们只要按照适应程度 fitness 来选 pop 中的 parent 就好. fitness 越大, 越有可能被选到.
    return pop[idx]

#交叉
def crossover(parent,pop):
    if np.random.rand() < CROSS_RATE:
        i_ = np.random.randint(0, POP_SIZE, size=1)  # 选择一个母本的index
        print("i_:",i_)
        print("i_.size:",i_.size)
        cross_points = np.random.randint(0, 2, size=DNA_SIZE).astype(np.bool)  # choose crossover points
        print("cross_point",cross_points)
        print("corsssize:",cross_points.size)
        print(parent[cross_points])
        print(pop[i_, cross_points])
        parent[cross_points] = pop[i_, cross_points]  # mating and produce one child
        print(parent[cross_points])
    return parent

#变异
def mutate(child):
    for point in range(DNA_SIZE):
        if np.random.rand() < MUTATION_RATE:
            child[point] = 1 if child[point] == 0 else 0
    return child

pop = np.random.randint(2, size=(POP_SIZE, DNA_SIZE))   # initialize the pop DNA

plt.ion()       # something about plotting
x = np.linspace(*X_BOUND, 200)
plt.plot(x, F(x))

for _ in range(N_GENERATIONS):
    F_values = F(translateDNA(pop))    # compute function value by extracting DNA

    # something about plotting
    # if 'sca' in globals():
    #     sca.remove()
    # sca = plt.scatter(translateDNA(pop), F_values, s=200, lw=0, c='red', alpha=0.5)
    # plt.pause(0.05)

    # GA part (evolution)
    fitness = get_fitness(F_values)
    print("Most fitted DNA: ", pop[np.argmax(fitness), :])
    pop = select(pop, fitness)
    pop_copy = pop.copy()
    for parent in pop:
        child = crossover(parent, pop_copy)
        child = mutate(child)
        parent[:] = child       # parent is replaced by its child
plt.scatter(translateDNA(pop), F_values, s=200, lw=0, c='red', alpha=0.5)
print("x:",translateDNA(pop))
print(type(translateDNA(pop)))
print(len(translateDNA(pop)))
print("max:",F_values)
plt.ioff()
plt.show()
