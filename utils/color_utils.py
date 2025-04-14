import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 从配置文件导入颜色方案
from config.color_palettes import COLOR_PALETTES

# 颜色处理函数 - 匹配到最接近的调色板颜色
def apply_color_palette(image, palette_name='abao', strength=1.0):
    """
    将图像的颜色映射到指定的调色板上
    image: 输入图像张量 [B, C, H, W] 范围在[-1, 1]之间
    palette_name: 调色板名称，如'abao'、'cyberpunk'等
    strength: 颜色映射强度，1.0表示完全映射，0.0表示不变
    """
    # 确保palette_name有效
    if palette_name not in COLOR_PALETTES:
        print(f"警告: 未找到名为 '{palette_name}' 的调色板，使用默认的阿宝色")
        palette_name = 'abao'
    
    # 获取调色板
    palette = COLOR_PALETTES[palette_name]
    
    # 转换到[0, 1]范围
    image_np = ((image.detach().cpu() + 1) / 2).clamp(0, 1)
    B, C, H, W = image_np.size()
    
    # 将图像重新形状为[B, H*W, C]
    image_reshaped = image_np.permute(0, 2, 3, 1).reshape(B, H*W, C)
    
    # 对每张图像进行处理
    mapped_images = []
    for i in range(B):
        # 将图像像素转为numpy
        pixels = image_reshaped[i].float().numpy()
        
        # 将每个像素映射到最近的调色板颜色
        distances = np.sqrt(((pixels[:, np.newaxis, :] - palette[np.newaxis, :, :]) ** 2).sum(axis=2))
        nearest_indices = np.argmin(distances, axis=1)
        mapped_pixels = palette[nearest_indices]
        
        # 混合原始颜色和映射颜色
        if strength < 1.0:
            mapped_pixels = pixels * (1 - strength) + mapped_pixels * strength
        
        # 重新形状为原始图像
        mapped_image = mapped_pixels.reshape(H, W, C)
        
        # 转回为PyTorch张量并归一化到[-1, 1]
        mapped_tensor = torch.from_numpy(mapped_image).float().permute(2, 0, 1)
        mapped_tensor = mapped_tensor * 2 - 1
        
        mapped_images.append(mapped_tensor)
    
    # 堆叠成批次
    return torch.stack(mapped_images)

# HSV增强函数
def enhance_hsv(image, saturation_factor=1.5, value_factor=1.2):
    """
    在HSV空间增强图像的饱和度和亮度
    image: 输入图像张量 [B, C, H, W] 范围在[-1, 1]之间
    saturation_factor: 饱和度增强系数
    value_factor: 亮度增强系数
    """
    # 转换到[0, 1]范围
    image_np = ((image.detach().cpu() + 1) / 2).clamp(0, 1)
    B, C, H, W = image_np.size()
    
    # 转换为numpy数组
    image_np = image_np.permute(0, 2, 3, 1).numpy()
    
    enhanced_images = []
    for i in range(B):
        # 转换到HSV空间
        hsv = cv2.cvtColor(image_np[i], cv2.COLOR_RGB2HSV)
        
        # 增强饱和度和亮度
        hsv[:,:,1] = np.clip(hsv[:,:,1] * saturation_factor, 0, 1)  # 饱和度
        hsv[:,:,2] = np.clip(hsv[:,:,2] * value_factor, 0, 1)       # 亮度
        
        # 转回RGB
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # 转为PyTorch张量并归一化到[-1, 1]
        enhanced = torch.from_numpy(rgb).float().permute(2, 0, 1)
        enhanced = enhanced * 2 - 1
        
        enhanced_images.append(enhanced)
    
    return torch.stack(enhanced_images)

# 色彩量化处理函数
def color_quantization(image, num_colors=8):
    """
    Perform color quantization on the image
    image: Input image tensor [B, C, H, W] in range [-1, 1]
    num_colors: Number of colors after quantization
    """
    # Convert to [0, 255] range
    image_np = ((image.detach().cpu() + 1) * 127.5).type(torch.uint8)
    B, C, H, W = image_np.size()
    
    # Reshape image to [B, H*W, C]
    image_reshaped = image_np.permute(0, 2, 3, 1).reshape(B, H*W, C)
    
    # Process each image in batch
    quantized_images = []
    for i in range(B):
        # Convert pixels to numpy
        pixels = image_reshaped[i].float().numpy()
        
        # Add small random noise to prevent duplicate points
        # This helps K-means find distinct clusters
        pixels_jittered = pixels + np.random.normal(0, 1.0, pixels.shape) * 0.5
        
        # Ensure we have enough unique colors for K-means
        unique_colors = np.unique(pixels_jittered.reshape(-1, C), axis=0)
        actual_num_colors = min(num_colors, len(unique_colors))
        
        if actual_num_colors < num_colors:
            print(f"Warning: Image only has {actual_num_colors} unique colors. Using {actual_num_colors} clusters instead of {num_colors}.")
        
        # Only run K-means if we have enough unique colors
        if actual_num_colors > 1:
            # K-means clustering
            kmeans = KMeans(n_clusters=actual_num_colors, random_state=0, n_init=1)
            labels = kmeans.fit_predict(pixels_jittered)
            colors = kmeans.cluster_centers_.astype(np.uint8)
            
            # Replace original pixels with cluster centers
            quantized = colors[labels].reshape(H, W, C)
        else:
            # If only one unique color, just use that color
            quantized = pixels.reshape(H, W, C).astype(np.uint8)
        
        # Convert back to PyTorch tensor and normalize to [-1, 1]
        quantized_tensor = torch.from_numpy(quantized).permute(2, 0, 1).float()
        quantized_tensor = quantized_tensor / 127.5 - 1
        
        quantized_images.append(quantized_tensor)
    
    # Stack into batch
    return torch.stack(quantized_images)

# 生成颜色方案可视化函数
def visualize_color_palettes(output_path="color_palettes.png"):
    """生成所有可用颜色方案的可视化"""
    n_palettes = len(COLOR_PALETTES)
    fig, axes = plt.subplots(n_palettes, 1, figsize=(10, n_palettes * 1.5))
    
    for i, (name, palette) in enumerate(COLOR_PALETTES.items()):
        # 创建色条
        palette_reshaped = palette.reshape(1, -1, 3)
        axes[i].imshow(palette_reshaped)
        axes[i].set_title(f"{name} 调色板")
        axes[i].set_yticks([])
        
        # 设置x轴刻度
        n_colors = len(palette)
        axes[i].set_xticks(range(0, n_colors * 3, 3))
        axes[i].set_xticklabels(range(1, n_colors + 1))
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"颜色方案可视化已保存到 {output_path}")