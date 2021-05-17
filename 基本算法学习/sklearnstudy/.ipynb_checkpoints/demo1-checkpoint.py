# -*- coding: utf-8 -*-

"""have a nice day.

@author: Khan
@contact:  
@time: 2020/7/14 21:16
@file: demo1.py
@desc:  
"""

from sklearn import preprocessing
import numpy as np
#标准化，获得期望为0，方差为1的矩阵
X_train = np.array([[ 1., -1.,  2.],
                    [ 2.,  0.,  0.],
                    [ 0.,  1., -1.]])
X_scaled = preprocessing.scale(X_train)
print(X_scaled)

enc = preprocessing.OrdinalEncoder()
X = [['male', 'from US', 'uses Safari'],
     ['female', 'from Europe', 'uses Firefox'],
     ['female', 'from Africa', 'uses IE']]
enc.fit(X)
#我怀疑是按字母大小排的序
X_enc=enc.transform([['female', 'from US', 'uses Safari']])
print(X_enc)
label=enc.inverse_transform(X_enc)
print(label)

#one-hot编码
genders = ['female', 'male']
locations = ['from Africa', 'from Asia', 'from Europe', 'from US']
browsers = ['uses Chrome', 'uses Firefox', 'uses IE', 'uses Safari']
enc = preprocessing.OneHotEncoder(categories=[genders, locations, browsers])
X = [['male', 'from US', 'uses Safari'], ['female', 'from Europe', 'uses Firefox']]
enc.fit(X)
X_onehot=enc.transform([['female', 'from Asia', 'uses Chrome']]).toarray()
print(X_onehot)
#[[1. 0. 0. 1. 0. 0. 1. 0. 0. 0.]]
#[['female', 'male'],['from Africa', 'from Asia', 'from Europe', 'from US'],['uses Chrome', 'uses Firefox', 'uses IE', 'uses Safari']]
#选中则为1

#文本处理
'''
sklearn.feature_extraction.text模块能够提取文本特征，将文本转化为向量，供后续的处理。
常用的特征提取方法有：
词频向量：CountVectorizer
TF-IDF向量：TfidfVectorizer
'''

# import numpy as np
# from wordcloud import WordCloud
# from sklearn.feature_extraction.text import CountVectorizer
# # 打开文件,读取每行文本
# data=[]
# with open('test.txt',encoding='UTF-8') as f:
#     for line in f.readlines():
#         data.append(line)
# # 统计每行文本的词频，并计算每个单词的总词频
# vectorizer = CountVectorizer(stop_words='english')
# X = vectorizer.fit_transform(data).toarray()
# count=np.sum(X,axis=0)
# name=vectorizer.get_feature_names()
# text=dict(zip(name,count))
# # 根据词频生成词云图
# w = WordCloud(background_color='white',width=800, height=400,max_words=10)
# w.generate_from_frequencies(text)
# # 保存文件
# w.to_file('wordcloud.png')
