#!/usr/bin/env python3
# coding=utf-8
__author__ = 'songshu'

import pandas as pd
import os
from datetime import datetime
from pyecharts import Line,Page,WordCloud,Bar,Style,Overlap

# 数据预处理
df = pd.read_excel('./data/'+os.listdir("./data/")[0]).fillna(value=0)
df = df[df['交易状态'] == '交易成功']
df['商家实际收入（元）'] = df['平台红包(元)'] + df['用户实付(元)'] - df['第三方支付费用(元)'] - df['渠道推广费用(元)'] - df['平台服务费用(元)']

# 计算销售总收入
sum_income = df['商家实际收入（元）'].sum()

# 汇总每日销售收入
df['订单日期'] = df['订单时间'].apply(lambda s:datetime.strptime(s,"%Y/%m/%d %H:%M") .date())
day_income = df.groupby('订单日期')['商家实际收入（元）'].sum()


# 图表宽度参数
self_width = 1800


# 总收入
name = ["Total Income: "+str(round(sum_income,0))+'元']
value = [1000]
wordcloud = WordCloud("总的销售收入统计",width = self_width,height=150)
wordcloud.add("", name, value,shape='diamond')


# 每日收入
day = day_income.index.tolist()
income = day_income.values.round().tolist()
# 折线图
line = Line("每日课程收入")
line.add("", day,income, is_smooth=True, line_width=3,mark_line=["max", "average"],label_color=['red'])
# 柱形图
bar_income = Bar("每日课程收入")
bar_income.add("",day,income,label_color=['grey'],is_label_show = True)

# 整合每日收入折线图与柱形图
overlap = Overlap(width = self_width,height=350)
overlap.add(bar_income)
overlap.add(line)

'''
# 汇总top10课程收入与销量
course_income = df.groupby('商品名称')['商家实际收入（元）'].sum().sort_values().tail(10)
course_count = df[df['商品名称'].isin(course_income.index)].groupby('商品名称')['订单编号'].count().sort_values()

# 课程收入与销量
x = course_income.index.tolist()
y = course_income.values.round(2).tolist()  # 销售额
z = course_count.values.tolist()            # 销售量
'''


# 汇总top10课程收入
course_income = df.groupby('商品名称')['商家实际收入（元）'].sum().sort_values().tail(10)
course_income = pd.DataFrame(course_income).reset_index()

# 收入top10课程销量
course_count = df[df['商品名称'].isin(course_income['商品名称'])].groupby('商品名称')['订单编号'].count()
course_count = pd.DataFrame(course_count).reset_index()

# 收入和销量数据合并
course_income_count = pd.merge(course_income,course_count,on = '商品名称',how ='left')
course_income_count.columns = ['课程','销售额','销量']

x = course_income_count['课程'].tolist()
y = course_income_count['销售额'].round().tolist()
z = course_income_count['销量'].tolist()


style = Style()

# 柱形图外观设置
style_bar = style.add(
    legend_top="bottom",        # 图例位置
    yaxis_label_textsize=9,     # y轴标签文字大小
    yaxis_rotate=45,            # y轴标签选择角度
    is_label_show = True,       # 展示柱形图上面的数值标签
    label_pos="right",          # 柱形图上面的数值标签显示位置
    label_text_size=12          # 数值标签文本大小
)

bar = Bar("单课销售量与销售额统计",width = self_width,height=400)
bar.add("销售额",x,y,**style_bar)
bar.add("销售量",x,z,is_convert=True,**style_bar)


# 组合输出
page = Page()
page.add(wordcloud)
page.add(overlap)
page.add(bar)
page.render('./result.html')


print("数据处理完毕，请使用浏览器打开result.html文件")