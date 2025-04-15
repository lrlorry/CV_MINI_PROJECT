def fft_horizontal_stripe_removal(img, strength=0.7):
    """使用FFT去除水平条纹，同时保留图像细节
    
    Args:
        img: 输入图像，可以是PIL Image对象或numpy数组，值范围为[0,1]或[0,255]
        strength: 去除强度，范围0-1，值越大去除效果越强
        
    Returns:
        处理后的图像，与输入格式相同
    """
    import numpy as np
    from PIL import Image
    
    # 检查输入类型并转换为numpy数组
    is_pil = isinstance(img, Image.Image)
    if is_pil:
        # 保存原始大小以便之后恢复
        original_size = img.size
        img_np = np.array(img).astype(np.float32)
        # 归一化到[0,1]
        if img_np.max() > 1.0:
            img_np = img_np / 255.0
    else:
        img_np = img.copy()
        if img_np.max() > 1.0:
            img_np = img_np / 255.0
    
    # 确保图像是3通道彩色
    if len(img_np.shape) == 2:
        # 灰度图转RGB
        img_np = np.stack([img_np, img_np, img_np], axis=2)
    
    # 创建结果数组
    result = img_np.copy()
    
    # 对每个通道分别处理
    for c in range(img_np.shape[2]):
        # 应用二维FFT
        f_transform = np.fft.fft2(img_np[:, :, c])
        f_shift = np.fft.fftshift(f_transform)
        
        # 创建频域掩码
        rows, cols = f_shift.shape
        crow, ccol = rows // 2, cols // 2
        
        # 创建掩码，默认全通过
        mask = np.ones((rows, cols), dtype=np.float32)
        
        # 根据强度参数计算衰减率
        # 调整strength使其更有效地控制衰减强度
        base_attenuation = 1.0 - strength * 0.9  # 强度为1时衰减到0.1
        
        # 计算要衰减的区域宽度，图像越高，需要处理的区域可能越窄
        stripe_height = max(1, int(rows * 0.01))  # 至少1像素，默认为高度的1%
        
        # 在水平频率区域应用衰减
        # 中心区域是低频，周围是高频
        mask_center_width = max(3, int(cols * 0.02))  # 中心区域宽度
        
        # 创建中心竖条掩码
        center_stripe = np.ones((rows, mask_center_width), dtype=np.float32)
        for i in range(stripe_height):
            # 计算当前行的衰减率，离中心越远衰减越小
            row_factor = 1.0 - (i / stripe_height)
            current_attenuation = base_attenuation + (1.0 - base_attenuation) * (1.0 - row_factor**2)
            
            # 应用到中心对称的上下两行
            if crow+i < rows and i > 0:  # 避开正中心行
                center_stripe[crow+i, :] = current_attenuation
            if crow-i >= 0 and i > 0:    # 避开正中心行
                center_stripe[crow-i, :] = current_attenuation
        
        # 将中心条嵌入完整掩码
        center_start = ccol - mask_center_width // 2
        mask[:, center_start:center_start+mask_center_width] = center_stripe
        
        # 应用掩码并进行逆变换
        f_shift_masked = f_shift * mask
        f_inverse = np.fft.ifftshift(f_shift_masked)
        img_back = np.fft.ifft2(f_inverse)
        result[:, :, c] = np.clip(np.abs(img_back), 0, 1)  # 确保结果在[0,1]范围内
    
    # 根据输入类型返回相应的输出
    if is_pil:
        # 转回PIL图像
        result_uint8 = (result * 255).astype(np.uint8)
        return Image.fromarray(result_uint8).resize(original_size)
    else:
        # 如果输入是值范围在[0,255]的numpy数组，则输出也应该是
        if img.max() > 1.0:
            result = result * 255.0
        return result