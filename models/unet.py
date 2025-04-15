import torch
import torch.nn as nn
import torch.nn.functional as F
from models.attention import SelfAttention
from utils.lab_processor import LabColorProcessor

# 修改后的U-Net架构，使用频域滤波去除水平伪影
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
        
        # 添加专门的水平线条消除过滤器
        self.h_line_filter = HorizontalLineFilter()
    
    def simple_blur(self, x, kernel_size=3):
        """应用简单的高斯模糊"""
        padding = kernel_size // 2
        return F.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=padding)
    
    def forward(self, sketch, depth, style_image=None, original_image=None, use_lab_colorspace=False):
        # 合并素描和深度信息
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

class HorizontalLineFilter(nn.Module):
    """专门用于去除水平线条伪影的频域滤波器"""
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        """
        使用频域滤波去除水平线条
        
        Args:
            x: 输入特征图 [B, C, H, W]
            
        Returns:
            去除水平线条后的特征图
        """
        batch_size, channels, height, width = x.size()
        
        # 对每个通道单独处理
        output = []
        for c in range(channels):
            channel_data = x[:, c:c+1]  # [B, 1, H, W]
            
            # 如果可用，使用torch的FFT
            try:
                # 将数据转换到频域
                fft_data = torch.fft.rfft2(channel_data)
                
                # 创建频域掩码 - 抑制水平频率
                mask = torch.ones_like(fft_data)
                
                # 抑制水平线（位于频域的y轴上）
                # 注意：在FFT中，低频在中心，高频在边缘
                # 我们使用低通过滤去除刚好对应水平线的频率
                h_center = height // 2
                
                # 对应于水平线的频率
                line_width = 1  # 控制抑制的频带宽度，越大效果越强但可能影响细节
                for i in range(1, 10):  # 锁定水平线对应的不同频率
                    freq_y = i * (height // 10)
                    if freq_y < mask.shape[2]:
                        # 只抑制这些特定频率，保留其他频率
                        mask[:, :, freq_y-line_width:freq_y+line_width+1, :] *= 0.2  # 不完全消除，而是减弱
                
                # 应用掩码并转换回空域
                filtered_fft = fft_data * mask
                filtered_channel = torch.fft.irfft2(filtered_fft, s=(height, width))
                
                output.append(filtered_channel)
            except:
                # 如果FFT不可用，回退到空域滤波
                # 创建专门针对水平线的滤波器
                kernel_v = torch.ones((5, 1), device=x.device) / 5  # 垂直方向的模糊核
                kernel_v = kernel_v.view(1, 1, 5, 1).repeat(1, 1, 1, 1)
                
                # 应用垂直模糊，这会平滑水平线
                padded = F.pad(channel_data, (0, 0, 2, 2), mode='replicate')
                filtered_channel = F.conv2d(padded, kernel_v, padding=0, groups=1)
                
                # 使用高频增强补偿模糊引起的细节损失
                high_freq = channel_data - filtered_channel
                enhanced = filtered_channel + high_freq * 0.5  # 保留50%的高频细节
                
                output.append(enhanced)
        
        # 合并所有处理后的通道
        return torch.cat(output, dim=1)