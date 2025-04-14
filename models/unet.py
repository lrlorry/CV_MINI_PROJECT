import torch
import torch.nn as nn
import torch.nn.functional as F
from models.attention import SelfAttention
from utils.lab_processor import LabColorProcessor

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
        self.dec1 = nn.Conv2d(base_filters + base_filters, 3, 3, 1, 1)                         # 输出RGB
    
    def forward(self, sketch, depth, style_image=None, original_image=None, use_lab_colorspace=False):
        """
        Forward pass with fixed gradient flow
        """
        # Explicitly ensure inputs require gradients
        sketch = sketch.detach().requires_grad_(True)
        depth = depth.detach().requires_grad_(True)
        
        # Merge sketch and depth information
        x = torch.cat([sketch, depth], dim=1)
        
        # Encoder forward pass
        e1 = F.relu(self.enc1(x))
        e2 = F.relu(self.enc2(e1))
        e3 = F.relu(self.enc3(e2))
        
        # Process style information if available
        if self.with_style_encoder and style_image is not None:
            # Encode style image
            s1 = F.relu(self.style_enc1(style_image))
            s2 = F.relu(self.style_enc2(s1))
            s3 = F.relu(self.style_enc3(s2))
            
            # Extract style features
            style_features = self.style_processor(s3)
            
            # Reshape style features to feature map
            batch_size = e3.size(0)
            style_spatial = style_features.view(batch_size, -1, 1, 1).expand_as(e3)
            
            # Modulate content features
            e3 = self.style_modulation(torch.cat([e3, style_spatial], dim=1))
        
        # Apply self-attention
        e3 = self.attention(e3)
        
        # Bottleneck layer
        b = self.bottleneck(e3)
        
        # Decoder forward pass (with skip connections)
        d3 = F.relu(self.dec3(torch.cat([b, e3], dim=1)))
        d2 = F.relu(self.dec2(torch.cat([d3, e2], dim=1)))
        d1 = torch.tanh(self.dec1(torch.cat([d2, e1], dim=1)))
        
        # Apply Lab color space processing if enabled and original image is provided
        if use_lab_colorspace and original_image is not None:
            # Create Lab color processor
            lab_processor = LabColorProcessor(device=d1.device)
            # Process color: keep L channel from original image, use ab channels from generated image
            d1 = lab_processor.process_ab_channels(original_image, d1)
        
        return d1