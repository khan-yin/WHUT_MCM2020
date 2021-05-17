# import numpy as np
import pandas as pd
# from pyecharts import Map
# # 导入excel表举例
df = pd.read_excel('aaa.xls')
# print(df.head(5))
# # 导入自定义的地点经纬度
# geo_cities_coords = {'china': [df.iloc[i]['经度']/10, df.iloc[i]['纬度']/10]
#                      for i in range(len(df))}  # 根据文件大小生成字典
#
# areas = ['china' for i in range(len(df))]  # 字典的每个键值
# values = list(df['强度'] )
# test_map = Map("test", width=1200, height=600)
# test_map.add("", areas, values, maptype='china', is_visualmap=True,
#         visual_text_color='#000', is_label_show=True,geo_cities_coords=geo_cities_coords)
# test_map.render() #notebook上会直接显示，不行就加上.render() 然后在当前文件目录上找


from pyecharts import Scatter
df0 = df[df['强度']==0]
df1 = df[df['强度']==1]
df2 = df[df['强度']==2]
df3 = df[df['强度']==3]
df4 = df[df['强度']==4]
df5 = df[df['强度']==5]
df6 = df[df['强度']==6]
df9 = df[df['强度']==9]

v01 = df0['经度']/10
v02 = df0['纬度']/10
v11 = df1['经度']/10
v12 = df1['纬度']/10
v21 = df2['经度']/10
v22 = df2['纬度']/10
v31 = df3['经度']/10
v32 = df3['纬度']/10
v41 = df4['经度']/10
v42 = df4['纬度']/10
v51 = df5['经度']/10
v52 = df5['纬度']/10
v61 = df6['经度']/10
v62 = df6['纬度']/10
v91 = df9['经度']/10
v92 = df9['纬度']/10
scatter = Scatter('台风经纬度图',width=1200, height=800)
scatter.add('强度0',v01,v02,
            is_visualmap = True,            #显示滑动条
            symbol_size = 3)    #显示滑动范围
scatter.add('强度1',v11,v12,
            is_visualmap = True,            #显示滑动条
            symbol_size = 3)    #显示滑动范围
scatter.add('强度2',v21,v22,
            is_visualmap = True,            #显示滑动条
            symbol_size = 3)    #显示滑动范围
scatter.add('强度3',v31,v32,
            is_visualmap = True,            #显示滑动条
            symbol_size = 3)    #显示滑动范围
scatter.add('强度4',v41,v42,
            is_visualmap = True,            #显示滑动条
            symbol_size = 3)    #显示滑动范围
scatter.add('强度5',v51,v52,
            is_visualmap = True,            #显示滑动条
            symbol_size = 3)    #显示滑动范围
scatter.add('强度6',v61,v62,
            is_visualmap = True,            #显示滑动条
            symbol_size = 3)    #显示滑动范围
scatter.add('强度9',v91,v92,
            is_visualmap = True,            #显示滑动条
            symbol_size = 3)    #显示滑动范围
scatter.render('./scatter02.html')
