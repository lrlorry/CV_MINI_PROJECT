import torch
import torch.nn as nn
import torch.nn.functional as F
from models.attention import SelfAttention
from utils.lab_processor import LabColorProcessor

# 修改后的U-Net架构，使用频域滤波去除水平伪影
class SketchDepthColorizer(nn.Module):
    def __init__(self, base_filters=16, with_style_encoder=True, with_semantic=True):
        super().__init__()
        self.with_style_encoder = with_style_encoder
        self.with_semantic = with_semantic
        
        # 修改输入通道：素描(1) + 深度(1) + 语义掩码(1)（如果启用）
        input_channels = 2
        if with_semantic:
            input_channels += 1
            
        # 修改编码器第一层
        self.enc1 = nn.Conv2d(input_channels, base_filters, 3, 1, 1)
        # 其余层保持不变...
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
        self.dec1 = nn.Conv2d(base_filters + base_filters, 3, 3, 1, 1)    
    
    def forward(self, sketch, depth, semantic=None, style_image=None, original_image=None, use_lab_colorspace=False):
        """
        Forward pass with semantic mask
        """
        # 合并素描、深度和语义信息
        if semantic is not None and self.with_semantic:
            x = torch.cat([sketch, depth, semantic], dim=1)
        else:
            x = torch.cat([sketch, depth], dim=1)
        
        # 编码器前向传播
        e1 = F.relu(self.enc1(x))
        e2 = F.relu(self.enc2(e1))
        e3 = F.relu(self.enc3(e2))
        
        # 应用水平线条滤波器
        if hasattr(self, 'h_line_filter'):
            e3 = self.h_line_filter(e3)
        
        # 处理样式信息
        if self.with_style_encoder and style_image is not None:
            # 编码风格图像
            s1 = F.relu(self.style_enc1(style_image))
            s2 = F.relu(self.style_enc2(s1))
            s3 = F.relu(self.style_enc3(s2))
            
            # 提取样式特征
            style_features = self.style_processor(s3)
            
            # 安全处理样式特征
            if style_features.dim() == 2:
                batch_size = e3.size(0)
                feature_dim = style_features.size(1)
                style_feat = style_features.view(batch_size, feature_dim, 1, 1)
                h, w = e3.size(2), e3.size(3)
                style_feat = style_feat.expand(batch_size, feature_dim, h, w)
            else:
                style_feat = style_features
                
            # 使用非常小的强度进行风格混合，几乎保持原始内容
            e3_with_style = self.style_modulation(torch.cat([e3, style_feat], dim=1))
            e3 = e3 * 0.99 + e3_with_style * 0.01  # 使用99%原始特征，1%风格特征
        
        # 应用自注意力
        e3 = self.attention(e3)
        
        # 瓶颈层
        b = self.bottleneck(e3)
        
        # 解码器前向传播(使用跳跃连接)
        d3 = F.relu(self.dec3(torch.cat([b, e3], dim=1)))
        d2 = F.relu(self.dec2(torch.cat([d3, e2], dim=1)))
        
        # 再次应用水平线条滤波器
        if hasattr(self, 'h_line_filter'):
            d2 = self.h_line_filter(d2)
        
        # 输出
        d1 = torch.tanh(self.dec1(torch.cat([d2, e1], dim=1)))
        
        # 应用Lab颜色空间处理（如果启用）
        if use_lab_colorspace and original_image is not None:
            lab_processor = LabColorProcessor(device=d1.device)
            d1 = lab_processor.process_ab_channels(original_image, d1)
        
        return d1
