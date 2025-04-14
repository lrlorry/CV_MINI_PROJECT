import torch
import torch.nn as nn
import torch.nn.functional as F

# 自注意力模块
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))  # 可学习的权重参数

    def forward(self, x):
        batch_size, C, height, width = x.size()
        
        # 生成查询、键和值
        proj_query = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)  # B x (H*W) x C'
        proj_key = self.key(x).view(batch_size, -1, height * width)  # B x C' x (H*W)
        attention = torch.bmm(proj_query, proj_key)  # B x (H*W) x (H*W)
        attention = F.softmax(attention, dim=-1)
        
        proj_value = self.value(x).view(batch_size, -1, height * width)  # B x C x (H*W)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # B x C x (H*W)
        out = out.view(batch_size, C, height, width)
        
        out = self.gamma * out + x  # 添加残差连接
        return out