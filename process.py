import torch
import argparse
import os
from config.color_palettes import COLOR_PALETTES
from models.unet import SketchDepthColorizer
from process.high_res import process_high_res_image
from utils.color_utils import visualize_color_palettes

def load_model(model_path, device=None, force_disable_style=False, force_disable_semantic=False):
    """加载模型"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载检查点
    print(f"加载模型: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    # 检查检查点格式
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # 检查模型是否有样式编码器
    has_style_encoder = any('style' in key for key in state_dict.keys())
    # 检查模型是否有语义输入支持
    has_semantic_input = any('enc1.weight' in key for key in state_dict.keys() if 'enc1.weight' in key and state_dict[key].shape[1] > 2)
    
    # 如果强制禁用，则设置为False
    if force_disable_style:
        has_style_encoder = False
    if force_disable_semantic:
        has_semantic_input = False
        
    print(f"检测到模型{'有' if has_style_encoder else '没有'}样式编码器")
    print(f"检测到模型{'支持' if has_semantic_input else '不支持'}语义输入")
    
    # 创建模型
    model = SketchDepthColorizer(base_filters=16, with_style_encoder=has_style_encoder, with_semantic=has_semantic_input)
    
    # 加载权重
    try:
        model.load_state_dict(state_dict)
        print("模型加载成功!")
    except Exception as e:
        print(f"加载模型出错: {e}")
        print("尝试使用不同的配置重新加载...")
        
        # 尝试反转样式编码器设置
        model = SketchDepthColorizer(base_filters=16, with_style_encoder=not has_style_encoder)
        try:
            model.load_state_dict(state_dict)
            print("使用反向配置加载模型成功!")
        except Exception as e2:
            print(f"反向配置加载也失败: {e2}")
            print("请检查模型文件是否正确")
            raise e2
            
    # 移动到设备并设置为评估模式
    model = model.to(device)
    model.eval()
    
    return model


def process_image(model_path, sketch_path, depth_path, output_dir="output",
                 semantic_path=None, style_path=None, block_size=512, overlap=64, color_mode="original", 
                 palette_name="abao", palette_strength=0.8, hsv_saturation=1.5, hsv_value=1.2,
                 use_lab_colorspace=False, force_disable_style=False, force_disable_semantic=False):
    """处理图像"""
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载模型
    model = load_model(model_path, force_disable_style=force_disable_style, force_disable_semantic=force_disable_semantic)
    
    # 确定输出路径
    output_filename = f"result_{color_mode}_{os.path.basename(sketch_path)}"
    output_path = os.path.join(output_dir, output_filename)
    
    # 如果使用调色板模式，创建调色板可视化
    if color_mode == "palette":
        visualize_color_palettes(os.path.join(output_dir, "color_palettes.png"))
    
    # 打印处理信息
    print(f"处理图像:")
    print(f"  素描: {sketch_path}")
    print(f"  深度图: {depth_path}")
    
    # 添加语义掩码信息
    if semantic_path and hasattr(model, 'with_semantic') and model.with_semantic:
        print(f"  语义掩码: {semantic_path}")
    elif semantic_path and not (hasattr(model, 'with_semantic') and model.with_semantic):
        print(f"  警告: 模型不支持语义掩码，忽略语义掩码")
        semantic_path = None

    if style_path and hasattr(model, 'with_style_encoder') and model.with_style_encoder:
        print(f"  风格图: {style_path}")
    elif style_path and not (hasattr(model, 'with_style_encoder') and model.with_style_encoder):
        print(f"  警告: 模型不支持风格，忽略风格图像")
        style_path = None
    
    # 处理图像
    process_high_res_image(
        model=model,
        sketch_path=sketch_path,
        depth_path=depth_path,
        semantic_path=semantic_path,  # 新增参数
        output_path=output_path,
        style_path=style_path,
        block_size=block_size,
        overlap=overlap,
        color_mode=color_mode,
        palette_name=palette_name,
        palette_strength=palette_strength,
        hsv_saturation=hsv_saturation,
        hsv_value=hsv_value,
        use_lab_colorspace=use_lab_colorspace
    )
    
    print(f"处理完成! 结果保存到 {output_path}")
    return output_path

def batch_process_all_styles(model_path, sketch_path, depth_path, output_dir="all_styles", 
                            semantic_path=None, style_path=None, force_disable_style=False, 
                            force_disable_semantic=False, use_lab_colorspace=False):
    """批量处理所有颜色方案"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载模型
    model = load_model(model_path, force_disable_style=force_disable_style, force_disable_semantic=force_disable_semantic)
    
    output_paths = []
    
    # 处理原始输出
    print("\n[1/8] 处理原始颜色...")
    output_path = os.path.join(output_dir, f"original_{os.path.basename(sketch_path)}")
    process_high_res_image(
        model=model,
        sketch_path=sketch_path,
        depth_path=depth_path,
        semantic_path=semantic_path,  # 新增参数
        output_path=output_path,
        style_path=style_path,
        color_mode="original",
        use_lab_colorspace=use_lab_colorspace
    )
    output_paths.append(output_path)
    
    # 处理所有调色板
    for i, (palette_name, _) in enumerate(COLOR_PALETTES.items()):
        print(f"\n[{i+2}/{len(COLOR_PALETTES)+2}] 处理 {palette_name} 调色板...")
        output_path = os.path.join(output_dir, f"palette_{palette_name}_{os.path.basename(sketch_path)}")
        process_high_res_image(
            model=model,
            sketch_path=sketch_path,
            depth_path=depth_path,
            output_path=output_path,
            style_path=style_path,
            color_mode="palette",
            palette_name=palette_name,
            palette_strength=0.8,
            use_lab_colorspace=use_lab_colorspace
        )
        output_paths.append(output_path)
    
    # 处理HSV增强
    print(f"\n[{len(COLOR_PALETTES)+3}/8] 处理HSV增强...")
    output_path = os.path.join(output_dir, f"hsv_{os.path.basename(sketch_path)}")
    process_high_res_image(
        model=model,
        sketch_path=sketch_path,
        depth_path=depth_path,
        output_path=output_path,
        style_path=style_path,
        color_mode="hsv",
        hsv_saturation=1.5,
        hsv_value=1.2,
        use_lab_colorspace=use_lab_colorspace
    )
    output_paths.append(output_path)
    
    # 处理颜色量化
    print(f"\n[8/8] 处理颜色量化...")
    output_path = os.path.join(output_dir, f"quantized_{os.path.basename(sketch_path)}")
    process_high_res_image(
        model=model,
        sketch_path=sketch_path,
        depth_path=depth_path,
        output_path=output_path,
        style_path=style_path,
        color_mode="quantized",
        use_lab_colorspace=use_lab_colorspace
    )
    output_paths.append(output_path)
    
    print(f"\n所有样式处理完成! 结果保存在 {output_dir} 目录下")
    return output_paths

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="单图像深度学习 - 处理脚本")
    
    # 必需参数
    parser.add_argument("--model", required=True, help="训练好的模型路径")
    parser.add_argument("--sketch", required=True, help="素描图路径")
    parser.add_argument("--depth", required=True, help="深度图路径")
    
    # 可选参数
    parser.add_argument("--semantic", default=None, help="语义掩码路径 (可选)")  # 新增参数
    parser.add_argument("--output", default="output", help="输出目录")
    parser.add_argument("--style", default=None, help="风格图像路径（可选）")
    
    # 现有参数...
    
    # 功能开关
    parser.add_argument("--no_semantic", action="store_true", help="强制禁用语义分割")  # 新增参数
    parser.add_argument("--no_style", action="store_true", help="强制禁用风格编码器")
    
    args = parser.parse_args()
    
    # 根据选择的模式执行处理
    if args.all_styles:
        batch_process_all_styles(
            model_path=args.model,
            sketch_path=args.sketch,
            depth_path=args.depth,
            semantic_path=args.semantic,  # 新增参数
            output_dir=args.output,
            style_path=args.style,
            force_disable_style=args.no_style,
            force_disable_semantic=args.no_semantic,  # 新增参数
            use_lab_colorspace=args.use_lab
        )
    else:
        process_image(
            model_path=args.model,
            sketch_path=args.sketch,
            depth_path=args.depth,
            semantic_path=args.semantic,  # 新增参数
            output_dir=args.output,
            style_path=args.style,
            block_size=args.block_size,
            overlap=args.overlap,
            color_mode=args.color_mode,
            palette_name=args.palette,
            palette_strength=args.palette_strength,
            hsv_saturation=args.hsv_saturation,
            hsv_value=args.hsv_value,
            use_lab_colorspace=args.use_lab,
            force_disable_style=args.no_style,
            force_disable_semantic=args.no_semantic  # 新增参数
        )