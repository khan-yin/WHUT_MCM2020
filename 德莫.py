# -*- coding: utf-8 -*-

"""have a nice day.

@author: Khan
@contact:  
@time: 2020/8/1 10:39
@file: 德莫.py
@desc:  
"""
def finddic(s):
    for i in dic:
        if i == s:
            return True
    return False

def isright(s):
    lis = s.split("-")
    if int(lis[1])==2:
        if int(lis[2]) in [2004, 2008, 2012, 2016,2020]:
            if int(lis[0]) <= 0 or int(lis[0]) > 29:
                return False
        else:
            if int(lis[0]) <= 0 or int(lis[0]) > 28:
                return False
    if int(lis[0]) <= 0 or int(lis[0]) > 31:
        return False
    if int(lis[1]) <= 0 or int(lis[1]) > 12:
        return False
    if int(lis[1]) in [4,6,9,11]:
        print("小")
        if int(lis[0])==31:
            return False
    if int(lis[2]) <= 2000 or int(lis[2]) > 2020:
        return False
    return True





a=input()
lenga=int((len(a)+2)/10)
# print("lenga=",lenga)
k=0
dic=dict()
for i in range(lenga):
    s=a[k:k+10]
    k=k+8
    if isright(s)==True:
        # print(s)
        if finddic(s)==True:
            dic[s]+=1
        else:
            dic[s] = 1

m=0
ms=""
for i in dic:
    if dic[i]>m:
        m=dic[i]
        ms=i
print(ms)
# print(a[:10])
# dic={"20-12-2030-12-2020":1}
# dic["20-12-2030-12-2020"]+=1


# for i in range(2001,2021):
#     for j in range(1,13):
#         for k in range()
