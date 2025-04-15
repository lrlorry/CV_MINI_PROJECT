import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
from utils.image_filters import remove_horizontal_lines  # 假设你将函数放在这个模块

from utils.color_utils import apply_color_palette, enhance_hsv, color_quantization

def process_high_res_image(model, sketch_path, depth_path, output_path, 
                          style_path=None, block_size=512, overlap=64,
                          color_mode="original", palette_name='abao', 
                          palette_strength=0.8, hsv_saturation=1.5, hsv_value=1.2,
                          use_lab_colorspace=False):  
    """
    Process high-resolution images using block-wise processing and output full-size result
    
    Args:
        model: Trained model
        sketch_path: Path to sketch image
        depth_path: Path to depth image
        output_path: Path for output image
        style_path: Path to style image (optional)
        block_size: Size of each processing block (default 512)
        overlap: Number of overlapping pixels between adjacent blocks (default 64)
        color_mode: Color processing mode ("original", "palette", "hsv", "quantized")
        palette_name: Palette name (effective when color_mode is "palette")
        palette_strength: Palette application strength (effective when color_mode is "palette")
        hsv_saturation: HSV saturation enhancement factor (effective when color_mode is "hsv")
        hsv_value: HSV brightness enhancement factor (effective when color_mode is "hsv")
        use_lab_colorspace: Whether to use Lab color space processing
    
    Returns:
        output_pil: Processed PIL image object
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Load original images
    sketch_pil = Image.open(sketch_path).convert('L')
    depth_pil = Image.open(depth_path).convert('L')
    
    # Get original dimensions
    original_width, original_height = sketch_pil.size
    print(f"Processing image size: {original_width}x{original_height}")
    
    # Ensure output folder exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Load style image (if provided)
    style_tensor = None
    if style_path and hasattr(model, 'with_style_encoder') and model.with_style_encoder:
        style_pil = Image.open(style_path).convert('RGB')
        transform_color = transforms.Compose([
            transforms.Resize((block_size, block_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Convert to [-1,1]
        ])
        style_tensor = transform_color(style_pil).unsqueeze(0).to(device)
    
    # Create empty output array (use float for precise weighting)
    output_array = np.zeros((original_height, original_width, 3), dtype=np.float32)
    weight_map = np.zeros((original_height, original_width), dtype=np.float32)
    
    # Calculate stride
    stride = block_size - overlap
    
    # Create window weights (higher in center, lower at edges) - use cosine window for smooth transitions
    window = np.ones((block_size, block_size), dtype=np.float32)
    for i in range(overlap//2):
        factor = 0.5 * (1 - np.cos(np.pi * i / (overlap//2)))  # Cosine smooth transition
        window[i, :] *= factor
        window[block_size-i-1, :] *= factor
        window[:, i] *= factor
        window[:, block_size-i-1] *= factor
    
    # Transformation function
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Convert to [-1,1]
    ])
    
    transform_color = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Convert to [-1,1]
    ])
    
    # Calculate total blocks to create progress bar
    x_blocks = max(1, (original_width - overlap) // stride)
    y_blocks = max(1, (original_height - overlap) // stride)
    total_blocks = x_blocks * y_blocks
    
    # Create progress bar using tqdm
    pbar = tqdm(total=total_blocks, desc="Processing image blocks")
    
    # Process each block
    for y_start in range(0, original_height - overlap, stride):
        # Ensure not beyond image boundaries
        y_end = min(y_start + block_size, original_height)
        if y_end - y_start < block_size:
            y_start = max(0, y_end - block_size)
        
        for x_start in range(0, original_width - overlap, stride):
            # Ensure not beyond image boundaries
            x_end = min(x_start + block_size, original_width)
            if x_end - x_start < block_size:
                x_start = max(0, x_end - block_size)
            
            # Extract current block
            sketch_block = sketch_pil.crop((x_start, y_start, x_end, y_end))
            depth_block = depth_pil.crop((x_start, y_start, x_end, y_end))
            
            # If block size is incorrect, resize
            if sketch_block.size != (block_size, block_size):
                sketch_block = sketch_block.resize((block_size, block_size), Image.LANCZOS)
                depth_block = depth_block.resize((block_size, block_size), Image.LANCZOS)
                
            # Convert to tensors
            sketch_tensor = transform(sketch_block).unsqueeze(0).to(device)
            depth_tensor = transform(depth_block).unsqueeze(0).to(device)
            
            # 调试: 输出输入张量的范围
            if x_start == 0 and y_start == 0:  # 只对第一个块输出，避免太多日志
                print(f"[调试] 输入张量范围: sketch={sketch_tensor.min().item():.4f}-{sketch_tensor.max().item():.4f}, depth={depth_tensor.min().item():.4f}-{depth_tensor.max().item():.4f}")
            
            # For Lab color processing, we need a 3-channel version of the sketch
            sketch_rgb = None
            if use_lab_colorspace:
                # Create a proper 3-channel RGB tensor for Lab processing
                # This needs to be in the right format for the model
                sketch_rgb = sketch_tensor.repeat(1, 3, 1, 1)  # Create 3-channel RGB from grayscale
            
            # Use model to generate output
            with torch.no_grad():
                try:
                    output = model(sketch_tensor, depth_tensor, style_image=style_tensor,
                                original_image=sketch_rgb,
                                use_lab_colorspace=use_lab_colorspace)
                    
                    # 调试: 输出模型输出的范围
                    if x_start == 0 and y_start == 0:  # 只对第一个块输出，避免太多日志
                        print(f"[调试] 模型输出张量范围: min={output.min().item():.4f}, max={output.max().item():.4f}")
                    
                    # Check for NaN values in output
                    if torch.isnan(output).any():
                        print(f"Warning: NaN values detected in model output for block at ({x_start},{y_start})")
                        output = torch.nan_to_num(output, nan=0.0)
                    
                    # Apply color processing
                    if color_mode == "palette":
                        output = apply_color_palette(output, palette_name=palette_name, strength=palette_strength)
                    elif color_mode == "hsv":
                        output = enhance_hsv(output, saturation_factor=hsv_saturation, value_factor=hsv_value)
                    elif color_mode == "quantized":
                        output = color_quantization(output, num_colors=8)
                except Exception as e:
                    print(f"Error processing block at ({x_start},{y_start}): {e}")
                    # Use fallback - just a gray block
                    output = torch.zeros(1, 3, block_size, block_size, device=device)
            
            # Convert to numpy array [-1,1] -> [0,1]
            output_np = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
            output_np = (output_np + 1) / 2
            output_np = np.clip(output_np, 0, 1)
            
            # 调试: 输出转换后的numpy数组范围
            if x_start == 0 and y_start == 0:  # 只对第一个块输出，避免太多日志
                print(f"[调试] 转换后numpy数组范围: min={output_np.min():.4f}, max={output_np.max():.4f}")
            
            # Check for NaN or Inf values in the numpy array
            if np.isnan(output_np).any() or np.isinf(output_np).any():
                print(f"Warning: NaN or Inf values detected in numpy array for block at ({x_start},{y_start})")
                output_np = np.nan_to_num(output_np, nan=0.5, posinf=1.0, neginf=0.0)
            
            # Resize back to original block size (if needed)
            if sketch_block.size != (x_end - x_start, y_end - y_start):
                output_np = np.array(Image.fromarray((output_np * 255).astype(np.uint8)).resize(
                    (x_end - x_start, y_end - y_start), Image.LANCZOS)) / 255.0
                window_resized = np.array(Image.fromarray(window).resize(
                    (x_end - x_start, y_end - y_start), Image.LANCZOS))
            else:
                window_resized = window
            
            # Apply window weights
            weighted_output = output_np * window_resized[:, :, np.newaxis]

            # NaN check - prevent NaN propagation before accumulation
            if np.isnan(weighted_output).any():
                print(f"NaN detected in weighted output for block at ({x_start},{y_start})")
                weighted_output = np.nan_to_num(weighted_output, nan=0.0)

            if np.isnan(window_resized).any():
                print(f"NaN detected in window weights for block at ({x_start},{y_start})")
                window_resized = np.nan_to_num(window_resized, nan=1.0)

            # Accumulate to final output
            output_array[y_start:y_end, x_start:x_end] += weighted_output
            weight_map[y_start:y_end, x_start:x_end] += window_resized
            
            # Update progress bar
            pbar.update(1)
    
    # Close progress bar
    pbar.close()
    
    # 调试: 输出归一化前的数据范围
    print(f"[调试] 归一化前输出数组范围: min={output_array.min():.4f}, max={output_array.max():.4f}")
    print(f"[调试] 权重图范围: min={weight_map.min():.4f}, max={weight_map.max():.4f}")
    
    # Normalize output (according to weights)
    weight_map = np.maximum(weight_map, 1e-6)[:, :, np.newaxis]  # Avoid division by zero, broadcast to 3 channels
    output_array = output_array / weight_map
    
    # 调试: 输出归一化后的数据范围
    print(f"[调试] 归一化后输出数组范围: min={output_array.min():.4f}, max={output_array.max():.4f}")
    
    # Check for NaN or Inf values in the final output
    if np.isnan(output_array).any() or np.isinf(output_array).any():
        print(f"Warning: NaN or Inf values detected in final output")
        output_array = np.nan_to_num(output_array, nan=0.5, posinf=1.0, neginf=0.0)

    output_array = np.clip(output_array, 0.0, 1.0)
    # Convert to 8-bit image
    output_uint8 = (output_array * 255).astype(np.uint8)
    output_pil = Image.fromarray(output_uint8)
    # # # !!!!应用FFT去除水平条纹
    # output_pil = remove_horizontal_lines(output_pil, strength=0.8)  # 强度可以调整
    # Save result
    output_pil.save(output_path)
    print(f"Processing complete! Result saved to {output_path}")
    
    return output_pil