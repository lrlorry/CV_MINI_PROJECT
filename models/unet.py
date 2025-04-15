import torch
import torch.nn as nn
import torch.nn.functional as F
from models.attention import SelfAttention
from utils.lab_processor import LabColorProcessor

# 修改后的U-Net架构，专注于去除伪影
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
    
    def anti_artifact_filter(self, x):
        """
        简单专注的抗伪影滤波器，只针对水平线条
        """
        # 只在水平方向应用平滑，垂直方向保持锐利
        kernel_size = 5
        # 创建水平平滑核
        kernel = torch.ones((1, kernel_size), device=x.device) / kernel_size
        kernel = kernel.view(1, 1, 1, kernel_size).repeat(x.shape[1], 1, 1, 1)
        
        # 应用水平平滑
        padded_x = F.pad(x, (kernel_size//2, kernel_size//2, 0, 0), mode='replicate')
        smoothed = F.conv2d(padded_x, kernel, groups=x.shape[1])
        
        # 检测水平线条 - 水平方向上的高频信息
        # 我们不是对整个图像应用平滑，而是仅对检测到水平伪影的区域
        
        # 计算水平梯度（检测垂直变化）
        padded_x = F.pad(x, (0, 0, 1, 1), mode='replicate')
        vertical_grad = padded_x[:,:,1:,:] - padded_x[:,:,:-1,:]
        vertical_grad = torch.abs(vertical_grad)
        
        # 计算垂直梯度（检测水平变化）
        padded_x = F.pad(x, (1, 1, 0, 0), mode='replicate')
        horizontal_grad = padded_x[:,:,:,1:] - padded_x[:,:,:,:-1]
        horizontal_grad = torch.abs(horizontal_grad)
        
        # 比较梯度 - 垂直梯度小且水平梯度大的区域可能是水平线伪影
        # 梯度比率：垂直梯度 / 水平梯度（小值表示水平线）
        grad_ratio = vertical_grad / (horizontal_grad + 1e-6)
        
        # 只平滑明显的水平线区域
        # 创建平滑掩码 - 值越小，平滑程度越高
        smoothing_mask = torch.sigmoid(1.0 - grad_ratio)
        # 扩展维度以匹配输入
        smoothing_mask = F.interpolate(smoothing_mask, size=x.shape[2:], mode='nearest')
        
        # 控制总体平滑强度
        smoothing_strength = 0.4  # 只应用40%的平滑效果，可以调整
        
        # 应用自适应平滑
        result = x * (1.0 - smoothing_mask * smoothing_strength) + smoothed * (smoothing_mask * smoothing_strength)
        
        return result
    
    def forward(self, sketch, depth, style_image=None, original_image=None, use_lab_colorspace=False):
        # 合并素描和深度信息
        x = torch.cat([sketch, depth], dim=1)
        
        # 编码器前向传播
        e1 = F.relu(self.enc1(x))
        e2 = F.relu(self.enc2(e1))
        e3 = F.relu(self.enc3(e2))
        
        # 应用抗伪影滤波器
        e3 = self.anti_artifact_filter(e3)
        
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
            alpha = 0.01  # 几乎不用风格特征
            e3_with_style = self.style_modulation(torch.cat([e3, style_feat], dim=1))
            e3 = e3 * (1-alpha) + e3_with_style * alpha
        
        # 应用自注意力
        e3 = self.attention(e3)
        
        # 瓶颈层
        b = self.bottleneck(e3)
        
        # 解码器前向传播(使用跳跃连接)
        d3 = F.relu(self.dec3(torch.cat([b, e3], dim=1)))
        d2 = F.relu(self.dec2(torch.cat([d3, e2], dim=1)))
        
        # 在最终输出前再次应用抗伪影滤波器
        d2 = self.anti_artifact_filter(d2)
        
        # 输出
        d1 = torch.tanh(self.dec1(torch.cat([d2, e1], dim=1)))
        
        # 应用Lab颜色空间处理（如果启用）
        if use_lab_colorspace and original_image is not None:
            lab_processor = LabColorProcessor(device=d1.device)
            d1 = lab_processor.process_ab_channels(original_image, d1)
        
        return d1