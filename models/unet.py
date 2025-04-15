import torch
import torch.nn as nn
import torch.nn.functional as F
from models.attention import SelfAttention
from utils.lab_processor import LabColorProcessor

# === 核心函数：风格对齐 ===
def adaptive_instance_normalization(content_feat, style_feat, eps=1e-5):
    c_mean = content_feat.mean(dim=[2, 3], keepdim=True)
    c_std = content_feat.std(dim=[2, 3], keepdim=True) + eps
    s_mean = style_feat.mean(dim=[2, 3], keepdim=True)
    s_std = style_feat.std(dim=[2, 3], keepdim=True)
    normalized = (content_feat - c_mean) / c_std
    return normalized * s_std + s_mean

# 修改后的U-Net架构，添加样式编码器和注意力机制
class SketchDepthColorizer(nn.Module):
    def __init__(self, base_filters=16, with_style_encoder=True):
        super().__init__()
        self.with_style_encoder = with_style_encoder
        
        # 编码器 - 接收素描+深度图作为输入(2通道)
        self.enc1 = nn.Conv2d(2, base_filters, 3, 1, 1)       # 素描(1通道)+深度图(1通道)
        self.enc2 = nn.Conv2d(base_filters, base_filters*2, 4, 2, 1)  # 降采样到1/2
        self.enc3 = nn.Conv2d(base_filters*2, base_filters*4, 4, 2, 1)  # 降采样到1/4
        
        # 样式编码器 - 处理参考图像(3通道RGB)
        if self.with_style_encoder:
            self.style_enc1 = nn.Conv2d(3, base_filters, 3, 1, 1)
            self.style_enc2 = nn.Conv2d(base_filters, base_filters*2, 4, 2, 1)
            self.style_enc3 = nn.Conv2d(base_filters*2, base_filters*4, 4, 2, 1)
            
            # 样式处理
            self.style_processor = nn.Sequential(
                nn.AdaptiveAvgPool2d((8, 8)),
                nn.Flatten(),
                nn.Linear(base_filters*4 * 8 * 8, base_filters*4),
                nn.ReLU(),
                nn.Linear(base_filters*4, base_filters*4),
                nn.ReLU()
            )
            
            # 样式调制
            self.style_modulation = nn.Sequential(
                nn.Conv2d(base_filters*4 + base_filters*4, base_filters*4, 3, 1, 1),
                nn.ReLU()
            )
        
        # 添加自注意力模块
        self.attention = SelfAttention(base_filters*4)
        
        # 瓶颈层
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_filters*4, base_filters*4, 3, 1, 1),
            nn.ReLU()
        )
        
        # 解码器(带跳跃连接)
        self.dec3 = nn.ConvTranspose2d(base_filters*4 + base_filters*4, base_filters*2, 4, 2, 1)  # 上采样到1/2
        self.dec2 = nn.ConvTranspose2d(base_filters*2 + base_filters*2, base_filters, 4, 2, 1)    # 上采样到原始大小
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_filters + base_filters, 3, 3, 1, 1),
            # 移除默认的tanh，我们将在forward中使用自定义版本
        )
    
    def forward(self, sketch, depth, style_image=None, original_image=None, use_lab_colorspace=False):
        """
        Forward pass with simple and effective processing like in U_Net_block_v1.py
        """
        # 合并素描和深度信息 - 不做任何梯度操作
        x = torch.cat([sketch, depth], dim=1)
        
        # 编码器前向传播 - 保持与原版相同
        e1 = F.relu(self.enc1(x))
        e2 = F.relu(self.enc2(e1))
        e3 = F.relu(self.enc3(e2))
        
        # 处理样式信息 - 复制原版逻辑
        if self.with_style_encoder and style_image is not None:
            s1 = F.relu(self.style_enc1(style_image))
            s2 = F.relu(self.style_enc2(s1))
            s3 = F.relu(self.style_enc3(s2))
            
            style_features = self.style_processor(s3)
            
            # batch_size = e3.size(0)
            # style_spatial = style_features.view(batch_size, -1, 1, 1).expand_as(e3)
            #!!!!
            # e3 = self.style_modulation(torch.cat([e3, style_spatial], dim=1))
            # style_spatial 是形状和 e3 一样的 style 特征图
            # e3 = adaptive_instance_normalization(e3, style_spatial)
            # 统一 shape
            if style_features.dim() == 2:
                style_feat = style_features.view(style_features.size(0), style_features.size(1), 1, 1)
                style_feat = style_feat.expand_as(e3)
            else:
                style_feat = style_features

            # AdaIN with alpha
            alpha = 0.5
            e3_adain = adaptive_instance_normalization(e3, style_feat)
            e3 = alpha * e3_adain + (1 - alpha) * e3

        
        # 应用自注意力
        e3 = self.attention(e3)
        
        # 瓶颈层
        b = self.bottleneck(e3)
        
        # 解码器前向传播(使用跳跃连接)
        d3 = F.relu(self.dec3(torch.cat([b, e3], dim=1)))
        d2 = F.relu(self.dec2(torch.cat([d3, e2], dim=1)))
        
        # 核心改变: 采用与原版完全相同的简单处理
        d1 = torch.tanh(self.dec1(torch.cat([d2, e1], dim=1)))
        
        # 保留Lab空间处理功能，但简化处理
        if use_lab_colorspace and original_image is not None:
            lab_processor = LabColorProcessor(device=d1.device)
            d1 = lab_processor.process_ab_channels(original_image, d1)
        
        return d1

