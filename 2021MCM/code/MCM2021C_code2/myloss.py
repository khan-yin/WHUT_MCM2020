
from torch import nn
import torch
from torch.nn import functional as F
from torch.nn.modules.loss import _WeightedLoss
from torch.autograd import Variable

# class BCEFocalLoss(torch.nn.Module):
#     """
#     二分类的Focalloss alpha 固定
#     """
#
#     def __init__(self, gamma=2, alpha=0.25, reduction='elementwise_mean'):
#         super().__init__()
#         self.gamma = gamma
#         self.alpha = alpha
#         self.reduction = reduction
#
    # def forward(self, _input, target):
    #     pt = torch.sigmoid(_input)
    #     alpha = self.alpha
    #     loss = - alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - \
    #            (1 - alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)
    #     if self.reduction == 'elementwise_mean':
    #         loss = torch.mean(loss)
    #     elif self.reduction == 'sum':
    #         loss = torch.sum(loss)
    #     return loss

class MyCrossEntropyLoss(_WeightedLoss):
    def __init__(self, weight=None, gamma=2, alpha=0.25, reduction='elementwise_mean'):
        super().__init__(weight=weight, reduction=reduction)
        self.weight = weight
        self.reduction = reduction
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        pt=torch.sigmoid(inputs)
        # pt =torch.log_softmax(inputs,dim=1)
        # targets.view(targets.shape[0],-1)
        # print(pt.shape)
        # pd = - torch.log(pt)
        # print(torch.log(pt[:,1]))
        # print((1 - pt[:,1]))
        # print((1 - pt[:,1]) ** self.gamma)
        loss = - self.alpha * (1 - pt[:,1]) ** self.gamma * targets * torch.log(pt[:,1]) - \
               (1 - self.alpha) * pt[:,1] ** self.gamma * (1 - targets) * torch.log(1 - pt[:,1])
        if self.reduction == 'elementwise_mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss
