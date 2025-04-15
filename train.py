import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.utils as vutils
from PIL import Image, ImageEnhance, ImageDraw, ImageFont
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import random
import argparse

# 导入自定义模块
from models.unet import SketchDepthColorizer
from data.dataset import SketchDepthPatchDataset
from loss.combined_loss import CombinedLoss
from utils.image_utils import save_triplet
from utils.visualization import record_loss, generate_loss_curve, generate_video_and_metrics

# ==== 配置 ====
VIS_INTERVAL = 10
image_size = (256, 256)

vis_dir = "outputs/epoch_vis_full"
gif_path = "outputs/training_progress.gif"
mp4_path = "outputs/training_progress.mp4"
curve_path = "outputs/loss_curve.png"
metric_txt_path = "outputs/metrics_summary.txt"
metric_md_path = "outputs/metrics_summary.md"
metric_xlsx_path = "outputs/metrics_summary.xlsx"
log_path = "outputs/training.log"
os.makedirs(vis_dir, exist_ok=True)

loss_history = []

def train_from_sketch_depth(sketch_path, depth_path, target_path, output_dir="models", 
                          epochs=100, batch_size=8, patch_size=256, n_patches=1000,
                          lr=0.0002, use_style_loss=True, finetune_epochs=10, 
                          use_lab_colorspace=True, use_vgg_loss=True):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建数据集和加载器
    dataset = SketchDepthPatchDataset(sketch_path, depth_path, target_path, patch_size, n_patches)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建模型
    model = SketchDepthColorizer(base_filters=16, with_style_encoder=use_style_loss)
    model = model.to(device)
    model.train()
    
    # 验证模型有可训练参数
    trainable_params = sum(p.requires_grad for p in model.parameters())
    total_params = len(list(model.parameters()))
    print(f"Model has {trainable_params}/{total_params} trainable parameters")
    
    # 优化器
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                          lr=lr, betas=(0.5, 0.999))
    
    # 损失函数
    if use_vgg_loss:
        print("Using VGG perceptual loss")
        combined_loss = CombinedLoss(lambda_l1=1.0, lambda_perceptual=0.1)
        combined_loss = combined_loss.to(device)
    else:
        print("Using L1 loss only")
        l1_loss = nn.L1Loss()
    
    # 训练循环
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        loss_components = {'l1': 0.0, 'perceptual': 0.0, 'total': 0.0}
        
        with tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            for i, batch in enumerate(pbar):
                # 获取数据
                sketch = batch['sketch'].to(device)
                depth = batch['depth'].to(device)
                target = batch['target'].to(device)
                
                # 清除梯度
                optimizer.zero_grad()
                
                # 前向传播
                output = model(sketch, depth, 
                              style_image=target if use_style_loss else None,
                              original_image=target if use_lab_colorspace else None,
                              use_lab_colorspace=use_lab_colorspace)
                # 在计算损失之前添加颜色增强（可选，如果颜色仍然不够鲜艳）
                if not use_lab_colorspace:  # 仅当不使用Lab空间时添加
                    # 略微增强颜色，不影响内容
                    color_enhancement = 0.05  # 小的增强因子
                    mean_color = output.mean(dim=[2, 3], keepdim=True)  # 计算平均颜色
                    output = output + color_enhancement * (output - mean_color)  # 增强颜色差异
                # 计算损失
                if use_vgg_loss:
                    total_loss, loss_info = combined_loss(output, target)
                    for k, v in loss_info.items():
                        loss_components[k] += v
                else:
                    total_loss = l1_loss(output, target)
                    loss_val = total_loss.item()
                    loss_components['l1'] += loss_val
                    loss_components['total'] += loss_val
                
                # 检查损失是否正确连接到计算图
                if not total_loss.requires_grad:
                    print("WARNING: Loss doesn't require gradients!")
                    # 添加一个小的需要梯度的项以确保反向传播正常工作
                    total_loss = total_loss + 0.0 * sum(p.sum() for p in model.parameters() if p.requires_grad)
                
                # 反向传播
                total_loss.backward()
                
                # 应用梯度裁剪以防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # 更新权重
                optimizer.step()
                
                # 更新统计信息
                epoch_loss += total_loss.item()
                
                # 更新进度条
                if use_vgg_loss:
                    pbar.set_postfix(loss=total_loss.item(), 
                                   l1=loss_info['l1'], 
                                   perceptual=loss_info['perceptual'])
                else:
                    pbar.set_postfix(loss=total_loss.item())
                
        # 计算平均epoch损失
        avg_epoch_loss = epoch_loss / len(dataloader)
        global loss_history
        loss_history = record_loss(avg_epoch_loss, loss_history)
        
        # 可视化结果
        if (epoch+1) % VIS_INTERVAL == 0 or epoch == epochs-1:
            model.eval()
            with torch.no_grad():
                val_sketch = dataset[0]['sketch'].unsqueeze(0).to(device)
                val_depth = dataset[0]['depth'].unsqueeze(0).to(device)
                val_target = dataset[0]['target'].unsqueeze(0).to(device)
                
                val_output = model(val_sketch, val_depth, 
                                  original_image=val_target if use_lab_colorspace else None,
                                  use_lab_colorspace=use_lab_colorspace)
                
                save_triplet(sketch_path, depth_path, val_output, epoch+1, avg_epoch_loss)
            model.train()
        
        # 保存模型检查点
        if (epoch+1) % 10 == 0 or epoch == epochs-1:
            model_path = os.path.join(output_dir, f"model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), model_path)
            # 同时保存为latest以便轻松恢复
            torch.save(model.state_dict(), os.path.join(output_dir, "latest.pth"))
    
    # 保存最终模型
    final_model_path = os.path.join(output_dir, "final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Training complete! Final model saved to {final_model_path}")
    
    # 生成可视化
    generate_loss_curve(loss_history)
    generate_video_and_metrics()
    
    # 使用完整图像微调（如果请求）
    if finetune_epochs > 0:
        print(f"\nFine-tuning with full image for {finetune_epochs} epochs...")
        finetune_with_full_images(
            model, sketch_path, depth_path, target_path, output_dir,
            epochs=finetune_epochs, device=device,
            use_lab_colorspace=use_lab_colorspace,
            use_vgg_loss=use_vgg_loss
        )
    
    return final_model_path


def finetune_with_full_images(model, sketch_path, depth_path, target_path, output_dir, 
                             epochs=10, lr=0.0001, device=None, use_lab_colorspace=True, use_vgg_loss=True):
    """
    使用全图fine-tune模型,适用于已有素描和深度图的情况
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 设置模型为训练模式
    model.train()
    
    # 加载并预处理图像
    sketch = Image.open(sketch_path).convert('L')
    depth = Image.open(depth_path).convert('L')
    target = Image.open(target_path).convert('RGB')
    
    # 强制将图像调整到非常小的尺寸以适应GPU内存
    max_size = 512  # 使用非常保守的值，适合8GB显存
    w, h = sketch.size
    print(f"原始图像尺寸: {w}x{h}")
    
    # 强制调整大小，不使用条件判断
    scale = max_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # 确保尺寸是8的倍数（对于U-Net架构更安全）
    new_w = (new_w // 8) * 8
    new_h = (new_h // 8) * 8
    
    print(f"强制调整图像尺寸至: {new_w}x{new_h}")
    sketch = sketch.resize((new_w, new_h), Image.LANCZOS)
    depth = depth.resize((new_w, new_h), Image.LANCZOS)
    target = target.resize((new_w, new_h), Image.LANCZOS)
    
    # 转换为张量
    transform_gray = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # 转换到[-1,1]
    ])
    transform_color = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 转换到[-1,1]
    ])
    
    sketch_tensor = transform_gray(sketch).unsqueeze(0).to(device)
    depth_tensor = transform_gray(depth).unsqueeze(0).to(device)
    target_tensor = transform_color(target).unsqueeze(0).to(device)
    
    # 优化器 - 使用较小的学习率
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))
    
    # 损失函数 - 使用新的组合损失或仅L1
    if use_vgg_loss:
        combined_loss = CombinedLoss(lambda_l1=1.0, lambda_perceptual=0.5)
        combined_loss = combined_loss.to(device)
    else:
        l1_loss = nn.L1Loss()
    
    # Fine-tune循环
    with tqdm(range(epochs), desc="Fine-tuning with full image") as pbar:
        for epoch in pbar:
            # 前向传播
            output = model(sketch_tensor, depth_tensor, 
                          style_image=target_tensor if model.with_style_encoder else None,
                          original_image=target_tensor if use_lab_colorspace else None,
                          use_lab_colorspace=use_lab_colorspace)
            
            # 计算损失
            if use_vgg_loss:
                total_loss, loss_info = combined_loss(output, target_tensor)
                current_loss = total_loss.item()
                loss_l1 = loss_info['l1']
                loss_perceptual = loss_info['perceptual']
            else:
                total_loss = l1_loss(output, target_tensor)
                current_loss = total_loss.item()
                loss_l1 = current_loss
                loss_perceptual = 0.0
            
            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # 更新进度条
            if use_vgg_loss:
                pbar.set_postfix(loss=current_loss, l1=loss_l1, perceptual=loss_perceptual)
            else:
                pbar.set_postfix(loss=current_loss)
            
            # 保存中间可视化结果
            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                with torch.no_grad():
                    model.eval()
                    val_output = model(sketch_tensor, depth_tensor,
                                     original_image=target_tensor if use_lab_colorspace else None,
                                     use_lab_colorspace=use_lab_colorspace)
                    
                    # 保存可视化
                    result_img = torch.cat([
                        sketch_tensor.expand(-1, 3, -1, -1),  # 扩展通道到RGB
                        depth_tensor.expand(-1, 3, -1, -1),  # 扩展通道到RGB
                        val_output,
                        target_tensor
                    ], dim=0)
                    
                    vutils.save_image((result_img + 1) / 2, f"{output_dir}/finetune_epoch_{epoch+1}.png")
                
                # 恢复训练模式
                model.train()
    
    # 保存fine-tune后的模型
    ft_model_path = f"{output_dir}/finetuned_model.pth"
    torch.save(model.state_dict(), ft_model_path)
    print(f"Fine-tuned model saved to {ft_model_path}")
    
    return ft_model_path


# 主函数
if __name__ == "__main__":
    # 创建必要的目录
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    
    parser = argparse.ArgumentParser(description="单图像深度学习 - 使用patch方式训练和重叠拼接推理")
    # 新增参数
    parser.add_argument('--use_lab', action='store_true', help='使用Lab颜色空间处理')
    parser.add_argument('--use_vgg', action='store_true', help='使用VGG感知损失')
    parser.add_argument("--depth", default="depth.png", help="深度图路径 (仅process模式)")
    parser.add_argument("--image", default="jcsmr.jpg", help="输入图像路径")
    parser.add_argument("--sketch", default="sketch.jpg", help="素描图路径 (仅process模式)")
    parser.add_argument("--output", default="models", help="输出目录")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=8, help="no_style训练批次大小")
    parser.add_argument("--finetune_epochs", type=int, default=10, help="全图fine-tune轮数")
    parser.add_argument('--resume', type=str, default=None, help='Path to resume checkpoint')
    parser.add_argument('--no_style', action='store_true', help='强制禁用样式编码器')
    
    args = parser.parse_args()
    
    # 运行训练
    train_from_sketch_depth(
        sketch_path=args.sketch,
        depth_path=args.depth,
        target_path=args.image,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        finetune_epochs=args.finetune_epochs,
        use_lab_colorspace=args.use_lab,
        use_vgg_loss=args.use_vgg,
    )