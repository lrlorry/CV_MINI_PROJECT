"""
Visualization utilities for training and evaluation.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import imageio
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import pandas as pd

# ==== 记录 loss ====
def record_loss(loss, loss_history=None, log_path="outputs/training.log"):
    """
    记录损失值到历史记录和日志文件
    
    Args:
        loss (float): 损失值
        loss_history (list): 损失历史记录列表，如果为None则创建新列表
        log_path (str): 日志文件路径
        
    Returns:
        list: 更新后的损失历史记录列表
    """
    if loss_history is None:
        loss_history = []
        
    loss_history.append(loss)
    
    # 确保目录存在
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    with open(log_path, "a") as f:
        f.write(f"Loss: {loss:.6f}\n")
        
    return loss_history

# ==== 生成 loss 曲线图 ====
def generate_loss_curve(loss_history, curve_path="outputs/loss_curve.png"):
    """
    生成并保存损失曲线图
    
    Args:
        loss_history (list): 损失历史记录列表
        curve_path (str): 保存路径
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(loss_history)+1), loss_history, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.savefig(curve_path)
    plt.close()
    
    print(f"损失曲线已保存到 {curve_path}")

# ==== 生成 gif/mp4 & 评估指标 ====
def generate_video_and_metrics(vis_dir="outputs/epoch_vis_full", 
                              mp4_path="outputs/training_progress.mp4",
                              metric_txt_path="outputs/metrics_summary.txt",
                              metric_md_path="outputs/metrics_summary.md",
                              metric_xlsx_path="outputs/metrics_summary.xlsx"):
    """
    生成训练进度视频和计算图像质量指标
    
    Args:
        vis_dir (str): 包含可视化图像的目录
        mp4_path (str): 视频输出路径
        metric_txt_path (str): 指标文本输出路径
        metric_md_path (str): 指标Markdown输出路径
        metric_xlsx_path (str): 指标Excel输出路径
    """
    frames, ssim_scores, psnr_scores, records = [], [], [], []
    image_size = (256, 256)  # 默认尺寸

    # 获取所有可视化图像
    files = sorted([f for f in os.listdir(vis_dir) if f.endswith(".png")])
    
    for f in files:
        path = os.path.join(vis_dir, f)
        img = Image.open(path)
        
        # 自动检测图像大小
        if not frames:
            w, h = img.size
            image_size = (w // 3, h - 25)  # 假设三联布局，顶部有文本
        
        frames.append(np.array(img))

        # 提取预测结果和真实输入区域
        pred = img.crop((image_size[0]*2, 25, image_size[0]*3, 25+image_size[1]))
        gt = img.crop((0, 25, image_size[0], 25+image_size[1]))  # 替换为真实 GT 可提高准确性

        # 转换为浮点数组并确保在范围[0,1]内
        pred_np = np.asarray(pred).astype(np.float32) / 255.0
        gt_np = np.asarray(gt).astype(np.float32) / 255.0

        # 计算指标
        ssim_val = ssim(gt_np, pred_np, channel_axis=-1, data_range=1.0)
        psnr_val = psnr(gt_np, pred_np, data_range=1.0)
        
        ssim_scores.append(ssim_val)
        psnr_scores.append(psnr_val)
        records.append((f, ssim_val, psnr_val))

    # 确保有足够的帧来创建视频
    if frames:
        try:
            imageio.mimsave(mp4_path, frames, fps=2)
            print(f"视频已保存到 {mp4_path}")
        except Exception as e:
            print(f"生成视频时发生错误: {e}")

    # 确保有评估指标可以写入
    if records:
        # 文本格式
        with open(metric_txt_path, "w") as f:
            for fname, s, p in records:
                f.write(f"{fname}: SSIM={s:.4f}, PSNR={p:.2f}dB\n")
            f.write(f"\nAverage SSIM: {np.mean(ssim_scores):.4f}\n")
            f.write(f"Average PSNR: {np.mean(psnr_scores):.2f} dB\n")

        # Markdown和Excel格式
        df = pd.DataFrame(records, columns=["Epoch", "SSIM", "PSNR(dB)"])
        df.loc[len(df)] = ["**Average**", np.mean(ssim_scores), np.mean(psnr_scores)]
        
        with open(metric_md_path, "w") as f:
            f.write(df.to_markdown(index=False))
            
        df.to_excel(metric_xlsx_path, index=False)
        
        print(f"评估指标已保存到 {metric_txt_path}, {metric_md_path}, 和 {metric_xlsx_path}")
    else:
        print("没有可用的评估结果，可能是因为没有找到可视化文件")