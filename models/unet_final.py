import torch
import torch.nn as nn
import torch.nn.functional as F
from models.attention import SelfAttention
from utils.lab_processor import LabColorProcessor

class DirectHorizontalLineFilter(nn.Module):
    """
    直接针对水平线条的滤波器，使用更激进的方法移除伪影
    """
    def __init__(self, strength=0.8):
        super().__init__()
        self.strength = strength
        
    def forward(self, x):
        """
        直接移除水平线条
        Args:
            x: 输入特征图 [B, C, H, W]
        Returns:
            去除水平线条后的特征图
        """
        batch_size, channels, height, width = x.size()
        device = x.device
        
        # 创建中值滤波器 - 垂直方向有效去除水平线
        # 对每个通道单独进行处理
        filtered = torch.zeros_like(x)
        
        # 垂直方向的均值滤波，保留水平方向的细节
        kernel_size = 5  # 垂直方向的核大小，较大的值会更强烈地消除水平线
        padding = kernel_size // 2
        
        for b in range(batch_size):
            for c in range(channels):
                # 获取当前通道数据
                current = x[b, c].unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                
                # 计算垂直方向的滑动平均
                # 创建只在垂直方向有长度的核
                vertical_kernel = torch.ones(kernel_size, 1, device=device) / kernel_size
                vertical_kernel = vertical_kernel.view(1, 1, kernel_size, 1)
                
                # 使用卷积实现垂直方向的滑动平均
                padded = F.pad(current, (0, 0, padding, padding), mode='replicate')
                vertical_smoothed = F.conv2d(padded, vertical_kernel)
                
                # 检测水平线 - 如果一行的值相似度高，可能是水平线
                # 计算每一行的标准差
                row_std = torch.std(current, dim=3, keepdim=True)  # [1, 1, H, 1]
                
                # 创建掩码 - 低标准差表示水平线（因为整行都很相似）
                # 使用平滑的sigmoid而不是硬阈值以获得更自然的过渡
                threshold = 0.05
                line_mask = torch.sigmoid((threshold - row_std) * 10)  # [1, 1, H, 1]
                
                # 平滑掩码以避免锐利边缘
                smooth_kernel = torch.ones(5, 1, device=device) / 5
                smooth_kernel = smooth_kernel.view(1, 1, 5, 1)
                padded_mask = F.pad(line_mask, (0, 0, 2, 2), mode='replicate')
                line_mask = F.conv2d(padded_mask, smooth_kernel)
                
                # 应用掩码 - 水平线区域使用垂直平滑结果，其他区域保持原样
                strength = self.strength  # 控制效果强度
                blended = current * (1 - line_mask * strength) + vertical_smoothed * (line_mask * strength)
                
                filtered[b, c] = blended.squeeze()
                
        return filtered

# 修改后的U-Net架构
class SketchDepthColorizer(nn.Module):
    def __init__(self, base_filters=16, with_style_encoder=True):
        super().__init__()
        self.with_style_encoder = with_style_encoder
        
        # 编码器 - 接收素描+深度图作为输入(2通道)
        self.enc1 = nn.Conv2d(2, base_filters, 3, 1, 1)
        self.enc2 = nn.Conv2d(base_filters, base_filters*2, 4, 2, 1)
        self.enc3 = nn.Conv2d(base_filters*2, base_filters*4, 4, 2, 1)
        
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
        self.dec3 = nn.ConvTranspose2d(base_filters*4 + base_filters*4, base_filters*2, 4, 2, 1)
        self.dec2 = nn.ConvTranspose2d(base_filters*2 + base_filters*2, base_filters, 4, 2, 1)
        self.dec1 = nn.Conv2d(base_filters + base_filters, 3, 3, 1, 1)
        
        # 添加直接水平线过滤器
        self.line_filter = DirectHorizontalLineFilter(strength=0.8)
        
        # 添加最终输出阶段的抗伪影处理
        self.final_filter = DirectHorizontalLineFilter(strength=0.9)
    
    def forward(self, sketch, depth, style_image=None, original_image=None, use_lab_colorspace=False):
        # 合并素描和深度信息
        x = torch.cat([sketch, depth], dim=1)
        
        # 编码器前向传播
        e1 = F.relu(self.enc1(x))
        e2 = F.relu(self.enc2(e1))
        e3 = F.relu(self.enc3(e2))
        
        # 应用水平线条过滤器
        e3 = self.line_filter(e3)
        
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
        
        # 再次应用水平线条过滤器
        d2 = self.line_filter(d2)
        
        # 预输出
        pre_out = self.dec1(torch.cat([d2, e1], dim=1))
        
        # 应用最终的水平线条过滤器
        filtered_out = self.final_filter(pre_out)
        
        # 最终输出
        d1 = torch.tanh(filtered_out)
        
        # 应用Lab颜色空间处理（如果启用）
        if use_lab_colorspace and original_image is not None:
            lab_processor = LabColorProcessor(device=d1.device)
            d1 = lab_processor.process_ab_channels(original_image, d1)
        
        return d1