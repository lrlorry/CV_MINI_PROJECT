import numpy as np
import cv2
from PIL import Image, ImageEnhance

def remove_horizontal_lines(image, strength=1.0):
    """
    针对性地移除水平线伪影，保留图像细节
    
    Args:
        image: PIL Image或numpy数组
        strength: 去除强度，0-1之间
        
    Returns:
        处理后的图像
    """
    # 检查输入类型
    is_pil = isinstance(image, Image.Image)
    if is_pil:
        # 保存原始尺寸
        original_size = image.size
        # 转换为numpy数组
        img_np = np.array(image)
        # 检查是否需要归一化
        if img_np.dtype == np.uint8:
            img_np = img_np.astype(np.float32) / 255.0
    else:
        img_np = image.copy()
        if img_np.max() > 1.0:
            img_np = img_np.astype(np.float32) / 255.0
    
    # 确保是3通道图像
    if len(img_np.shape) == 2:
        img_np = np.stack([img_np, img_np, img_np], axis=2)
    
    # 创建结果数组
    result = img_np.copy()
    
    # 第1步：创建水平线检测掩码
    height, width = img_np.shape[:2]
    gray = cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    
    # 水平Sobel算子检测水平梯度
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # 计算梯度方向和幅度
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    direction = np.arctan2(sobely, sobelx) * 180 / np.pi
    
    # 创建水平线掩码：梯度方向接近0°或180°的区域
    horizontal_mask = np.logical_or(
        np.logical_and(direction > -10, direction < 10),
        np.logical_or(
            np.logical_and(direction > 170, direction < 190),
            np.logical_and(direction < -170, direction > -190)
        )
    )
    horizontal_mask = horizontal_mask.astype(np.float32)
    
    # 扩展掩码区域
    kernel = np.ones((3, 15), np.uint8)  # 水平方向更宽的核
    horizontal_mask = cv2.dilate(horizontal_mask, kernel, iterations=1)
    
    # 将掩码平滑化
    horizontal_mask = cv2.GaussianBlur(horizontal_mask, (0, 0), 3)
    horizontal_mask = horizontal_mask * strength  # 应用强度系数
    
    # 第2步：逐通道处理
    for c in range(3):
        channel = img_np[:, :, c]
        
        # 方法1：水平中值滤波
        # 创建小的水平中值核
        filtered1 = cv2.medianBlur(channel, 5)
        
        # 方法2：各向异性滤波
        # 转换到uint8以使用OpenCV的各向异性滤波
        temp_channel = (channel * 255).astype(np.uint8)
        filtered2 = cv2.ximgproc.anisotropicDiffusion(temp_channel, 0.15, 0.25, 12)
        filtered2 = filtered2.astype(np.float32) / 255.0
        
        # 智能融合两种滤波结果
        filtered = filtered1 * 0.3 + filtered2 * 0.7
        
        # 基于掩码融合原始和滤波结果
        result[:, :, c] = channel * (1 - horizontal_mask) + filtered * horizontal_mask
    
    # 微调对比度以增强细节
    if is_pil:
        result_uint8 = (result * 255).astype(np.uint8)
        result_pil = Image.fromarray(result_uint8)
        enhancer = ImageEnhance.Contrast(result_pil)
        result_pil = enhancer.enhance(1.1)  # 轻微增强对比度
        
        # 调整大小并返回
        if result_pil.size != original_size:
            result_pil = result_pil.resize(original_size, Image.LANCZOS)
        return result_pil
    else:
        # 如果输入是值范围在[0,255]的numpy数组，则输出也应该是
        if image.max() > 1.0:
            result = result * 255.0
        return result