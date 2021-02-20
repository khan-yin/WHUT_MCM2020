# import seaborn as sns
# import matplotlib.pyplot as plt
# import numpy as np
# sns.set()
# f,ax=plt.subplots()
# C2= np.array([[176,27],[50,37]])
# sns.heatmap(C2,annot=True,ax=ax,fmt="d") #画热力图
#
# ax.set_title('confusion matrix') #标题
# ax.set_xlabel('predict') #x轴
# ax.set_ylabel('Postive') #y轴
# plt.show()
# plt.savefig('confusion matrix.pdf')
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from model import HornetsClassifier

#writer就相当于一个日志，保存你要做图的所有信息。第二句就是在你的项目目录下建立一个文件夹log，存放画图用的文件。刚开始的时候是空的
from tensorboardX import SummaryWriter
writer = SummaryWriter('log') #建立一个保存数据用的东西

model = HornetsClassifier('tf_efficientnet_b4_ns', 2, pretrained=True).cuda()
model.load_state_dict(torch.load('./model/50/tf_efficientnet_b4_ns_fold_2_16.pth'))
print(model)

# dummy_input = torch.rand(16, 3, 64, 64)  # 假设输入20张1*28*28的图片
# dummy_input=dummy_input.cuda()
# with SummaryWriter(comment='EfficientNet') as w:
#     w.add_graph(model, input_to_model=dummy_input)