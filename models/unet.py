import torch
import torch.nn as nn
import torch.nn.functional as F
from models.attention import SelfAttention
from utils.lab_processor import LabColorProcessor

# Modified U-Net architecture focusing on artifact removal
class SketchDepthColorizer(nn.Module):
    def __init__(self, base_filters=16, with_style_encoder=True):
        super().__init__()
        self.with_style_encoder = with_style_encoder
        
        # Encoder - takes sketch+depth as input (2 channels)
        self.enc1 = nn.Conv2d(2, base_filters, 3, 1, 1)
        self.enc2 = nn.Conv2d(base_filters, base_filters*2, 4, 2, 1)
        self.enc3 = nn.Conv2d(base_filters*2, base_filters*4, 4, 2, 1)
        
        # Style encoder - processes reference image (3 RGB channels)
        if self.with_style_encoder:
            self.style_enc1 = nn.Conv2d(3, base_filters, 3, 1, 1)
            self.style_enc2 = nn.Conv2d(base_filters, base_filters*2, 4, 2, 1)
            self.style_enc3 = nn.Conv2d(base_filters*2, base_filters*4, 4, 2, 1)
            
            # Style processing
            self.style_processor = nn.Sequential(
                nn.AdaptiveAvgPool2d((8, 8)),
                nn.Flatten(),
                nn.Linear(base_filters*4 * 8 * 8, base_filters*4),
                nn.ReLU(),
                nn.Linear(base_filters*4, base_filters*4),
                nn.ReLU()
            )
            
            # Style modulation
            self.style_modulation = nn.Sequential(
                nn.Conv2d(base_filters*4 + base_filters*4, base_filters*4, 3, 1, 1),
                nn.ReLU()
            )
        
        # Self-attention module
        self.attention = SelfAttention(base_filters*4)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_filters*4, base_filters*4, 3, 1, 1),
            nn.ReLU()
        )
        
        # Decoder (with skip connections)
        self.dec3 = nn.ConvTranspose2d(base_filters*4 + base_filters*4, base_filters*2, 4, 2, 1)
        self.dec2 = nn.ConvTranspose2d(base_filters*2 + base_filters*2, base_filters, 4, 2, 1)
        self.dec1 = nn.Conv2d(base_filters + base_filters, 3, 3, 1, 1)
    
    def anti_artifact_filter(self, x):
        """
        Fixed anti-artifact filter that ensures gradient tensors have matching dimensions
        """
        # Apply horizontal smoothing with a simple kernel
        kernel_size = 5
        kernel = torch.ones((1, kernel_size), device=x.device) / kernel_size
        kernel = kernel.view(1, 1, 1, kernel_size).repeat(x.shape[1], 1, 1, 1)
        
        # Apply horizontal smoothing only
        padding = (kernel_size//2, kernel_size//2, 0, 0)
        padded_x = F.pad(x, padding, mode='replicate')
        smoothed = F.conv2d(padded_x, kernel, groups=x.shape[1])
        
        # Calculate horizontal gradient (detects vertical edges)
        horizontal_grad = torch.zeros_like(x)
        horizontal_grad[:,:,:,1:] = torch.abs(x[:,:,:,1:] - x[:,:,:,:-1])
        
        # Calculate vertical gradient (detects horizontal edges)
        vertical_grad = torch.zeros_like(x)
        vertical_grad[:,:,1:,:] = torch.abs(x[:,:,1:,:] - x[:,:,:-1,:])
        
        # Calculate ratio of gradients - ensure dimensions match
        # Small values indicate horizontal artifacts
        epsilon = 1e-6  # Prevent division by zero
        grad_ratio = vertical_grad / (horizontal_grad + epsilon)
        
        # Create smoothing mask - areas with horizontal lines get more smoothing
        # Use threshold to detect horizontal artifacts
        threshold = 0.5
        smoothing_mask = (grad_ratio < threshold).float()
        
        # Apply smoothing selectively
        smoothing_strength = 0.4  # Adjust as needed
        result = x * (1.0 - smoothing_mask * smoothing_strength) + smoothed * (smoothing_mask * smoothing_strength)
        
        return result
    
    def simple_horizontal_smoothing(self, x, strength=0.2):
        """
        Simple horizontal smoothing as a fallback
        """
        # Create horizontal-only blur kernel
        kernel_size = 5
        kernel = torch.ones((1, kernel_size), device=x.device) / kernel_size
        kernel = kernel.view(1, 1, 1, kernel_size).repeat(x.shape[1], 1, 1, 1)
        
        # Apply horizontal blur
        padding = (kernel_size//2, kernel_size//2, 0, 0)
        padded_x = F.pad(x, padding, mode='replicate')
        blurred = F.conv2d(padded_x, kernel, groups=x.shape[1])
        
        # Blend with original
        return x * (1-strength) + blurred * strength
    
    def forward(self, sketch, depth, style_image=None, original_image=None, use_lab_colorspace=False):
        # Combine sketch and depth information
        x = torch.cat([sketch, depth], dim=1)
        
        # Encoder forward pass
        e1 = F.relu(self.enc1(x))
        e2 = F.relu(self.enc2(e1))
        e3 = F.relu(self.enc3(e2))
        
        # Apply anti-artifact filter with error handling
        try:
            e3 = self.anti_artifact_filter(e3)
        except RuntimeError:
            # Fall back to simple smoothing if advanced filter fails
            e3 = self.simple_horizontal_smoothing(e3)
        
        # Process style information
        if self.with_style_encoder and style_image is not None:
            # Encode style image
            s1 = F.relu(self.style_enc1(style_image))
            s2 = F.relu(self.style_enc2(s1))
            s3 = F.relu(self.style_enc3(s2))
            
            # Extract style features
            style_features = self.style_processor(s3)
            
            # Safely handle style features
            if style_features.dim() == 2:
                batch_size = e3.size(0)
                feature_dim = style_features.size(1)
                style_feat = style_features.view(batch_size, feature_dim, 1, 1)
                h, w = e3.size(2), e3.size(3)
                style_feat = style_feat.expand(batch_size, feature_dim, h, w)
            else:
                style_feat = style_features
                
            # Apply minimal style influence
            alpha = 0.01  # Very minimal style influence
            e3_with_style = self.style_modulation(torch.cat([e3, style_feat], dim=1))
            e3 = e3 * (1-alpha) + e3_with_style * alpha
        
        # Apply self-attention
        e3 = self.attention(e3)
        
        # Bottleneck
        b = self.bottleneck(e3)
        
        # Decoder forward pass (with skip connections)
        d3 = F.relu(self.dec3(torch.cat([b, e3], dim=1)))
        d2 = F.relu(self.dec2(torch.cat([d3, e2], dim=1)))
        
        # Apply anti-artifact filter before final output
        try:
            d2 = self.anti_artifact_filter(d2)
        except RuntimeError:
            # Fall back to simple smoothing if advanced filter fails
            d2 = self.simple_horizontal_smoothing(d2)
        
        # Final output
        d1 = torch.tanh(self.dec1(torch.cat([d2, e1], dim=1)))
        
        # Apply Lab color space processing (if enabled)
        if use_lab_colorspace and original_image is not None:
            lab_processor = LabColorProcessor(device=d1.device)
            d1 = lab_processor.process_ab_channels(original_image, d1)
        
        return d1