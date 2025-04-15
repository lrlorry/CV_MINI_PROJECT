import torch
import torch.nn as nn
import torch.nn.functional as F
from models.attention import SelfAttention
from utils.lab_processor import LabColorProcessor

def adaptive_instance_normalization(content_feat, style_feat, intensity=0.1):
    """
    自适应实例归一化，使用较低的强度来保留更多原始信息
    
    Args:
        content_feat: 内容特征张量 [B, C, H, W]
        style_feat: 风格特征张量 [B, C, H, W]
        intensity: 风格强度，较小值保留更多原始颜色
    
    Returns:
        Normalized content feature tensor
    """
    size = content_feat.size()
    
    # 计算内容特征的均值和标准差
    content_mean = content_feat.view(size[0], size[1], -1).mean(dim=2).view(size[0], size[1], 1, 1)
    content_std = content_feat.view(size[0], size[1], -1).std(dim=2).view(size[0], size[1], 1, 1) + 1e-5
    
    # 计算风格特征的均值和标准差
    style_mean = style_feat.view(size[0], size[1], -1).mean(dim=2).view(size[0], size[1], 1, 1)
    style_std = style_feat.view(size[0], size[1], -1).std(dim=2).view(size[0], size[1], 1, 1) + 1e-5
    
    # 标准化内容特征，但仅轻微应用风格特征的统计
    normalized = (content_feat - content_mean) / content_std
    
    # 混合内容和风格的统计信息，保持更多原始内容
    blended_std = content_std * (1-intensity) + style_std * intensity
    blended_mean = content_mean * (1-intensity) + style_mean * intensity
    
    return normalized * blended_std + blended_mean

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
            
            # 样式调制 - 更温和的方式
            self.style_modulation = nn.Sequential(
                nn.Conv2d(base_filters*4 + base_filters*4, base_filters*4, 3, 1, 1),
                nn.InstanceNorm2d(base_filters*4),  # 添加实例归一化以稳定特征
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
        self.dec1 = nn.Conv2d(base_filters + base_filters, 3, 3, 1, 1)                         # 输出RGB
        
        # 移除之前的颜色增强器，这可能是导致过度白化的原因
    
    def smooth_horizontal_lines(self, x, strength=0.3):
        """应用水平平滑以减少条纹伪影"""
        # 创建仅水平方向的模糊核
        kernel_size = 5
        kernel = torch.ones((1, kernel_size), device=x.device) / kernel_size
        kernel = kernel.view(1, 1, 1, kernel_size).repeat(x.shape[1], 1, 1, 1)
        
        # 应用水平方向的模糊
        padding = (0, kernel_size//2)
        blurred = F.conv2d(x, kernel, padding=padding, groups=x.shape[1])
        
        # 与原始图像混合
        return x * (1-strength) + blurred * strength
    
    def forward(self, sketch, depth, style_image=None, original_image=None, use_lab_colorspace=False):
        # 合并素描和深度信息
        x = torch.cat([sketch, depth], dim=1)
        
        # 编码器前向传播
        e1 = F.relu(self.enc1(x))
        e2 = F.relu(self.enc2(e1))
        e3 = F.relu(self.enc3(e2))
        
        # 应用水平平滑来减少条纹，使用较低的强度
        e3 = self.smooth_horizontal_lines(e3, strength=0.25)
        
        # 处理样式信息
        if self.with_style_encoder and style_image is not None:
            # 使用提供的风格图像
            # 编码风格图像
            s1 = F.relu(self.style_enc1(style_image))
            s2 = F.relu(self.style_enc2(s1))
            s3 = F.relu(self.style_enc3(s2))
            
            # 提取样式特征
            style_features = self.style_processor(s3)
            
            # 修改：安全处理和重新形塑样式特征，防止伪影
            if style_features.dim() == 2:
                # 确保批次大小匹配
                batch_size = e3.size(0)
                feature_dim = style_features.size(1)
                # 安全地重塑特征
                style_feat = style_features.view(batch_size, feature_dim, 1, 1)
                # 使用更精确的扩展方法
                h, w = e3.size(2), e3.size(3)
                style_feat = style_feat.expand(batch_size, feature_dim, h, w)
            else:
                style_feat = style_features
                
            # 应用温和的风格影响，几乎不改变原始内容
            alpha = 0.05  # 更低的强度
            e3 = self.style_modulation(torch.cat([e3, style_feat], dim=1))
        
        # 应用自注意力
        e3 = self.attention(e3)
        
        # 瓶颈层
        b = self.bottleneck(e3)
        
        # 解码器前向传播(使用跳跃连接)
        d3 = F.relu(self.dec3(torch.cat([b, e3], dim=1)))
        d2 = F.relu(self.dec2(torch.cat([d3, e2], dim=1)))
        
        # 应用温和的水平平滑
        d2 = self.smooth_horizontal_lines(d2, strength=0.2)
        
        # 输出层前应用最后的平滑
        d1_pre = torch.cat([d2, e1], dim=1)
        d1_pre = self.smooth_horizontal_lines(d1_pre, strength=0.15)
        
        # 基本输出 - 使用tanh但不做额外处理
        d1 = torch.tanh(self.dec1(d1_pre))
        
        # 应用Lab颜色空间处理（如果启用）
        if use_lab_colorspace and original_image is not None:
            lab_processor = LabColorProcessor(device=d1.device)
            d1 = lab_processor.process_ab_channels(original_image, d1)
        
        return d1
        
    def preserve_colors(self, content_img, style_img, preserve_ratio=0.5):
        """
        保留内容图像中的颜色，同时应用风格图像的纹理
        这个方法在输出阶段使用，而不是在特征空间中
        
        Args:
            content_img: 内容图像张量 [B, C, H, W]
            style_img: 风格图像张量 [B, C, H, W]
            preserve_ratio: 保留原始颜色的比例
        
        Returns:
            结合了内容颜色和风格纹理的图像
        """
        # 转换到YUV颜色空间
        content_y = 0.299 * content_img[:, 0:1] + 0.587 * content_img[:, 1:2] + 0.114 * content_img[:, 2:3]
        content_u = -0.147 * content_img[:, 0:1] - 0.289 * content_img[:, 1:2] + 0.436 * content_img[:, 2:3]
        content_v = 0.615 * content_img[:, 0:1] - 0.515 * content_img[:, 1:2] - 0.100 * content_img[:, 2:3]
        
        # 从风格图像中提取亮度
        style_y = 0.299 * style_img[:, 0:1] + 0.587 * style_img[:, 1:2] + 0.114 * style_img[:, 2:3]
        
        # 混合亮度通道，保留色度通道
        blended_y = style_y * (1-preserve_ratio) + content_y * preserve_ratio
        
        # 转换回RGB
        blended_r = blended_y + 1.140 * content_v
        blended_g = blended_y - 0.395 * content_u - 0.581 * content_v
        blended_b = blended_y + 2.032 * content_u
        
        # 合并通道
        return torch.cat([blended_r, blended_g, blended_b], dim=1)