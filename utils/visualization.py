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


def generate_finetune_metrics_with_plot(vis_dir="models", 
                                        metric_txt_path=None,
                                        metric_md_path=None,
                                        metric_xlsx_path=None,
                                        curve_path=None):
    """
    生成微调阶段的评估指标和可视化曲线图，风格与训练阶段保持一致
    
    Args:
        vis_dir (str): 包含微调可视化图像的目录
        metric_txt_path (str): 指标文本输出路径
        metric_md_path (str): 指标Markdown输出路径
        metric_xlsx_path (str): 指标Excel输出路径
        curve_path (str): 损失/指标曲线图保存路径
    """
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from PIL import Image
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr
    import re
    
    # 如果路径为None，使用默认值
    if metric_txt_path is None:
        metric_txt_path = os.path.join(vis_dir, "finetune_metrics_summary.txt")
    if metric_md_path is None:
        metric_md_path = os.path.join(vis_dir, "finetune_metrics_summary.md")
    if metric_xlsx_path is None:
        metric_xlsx_path = os.path.join(vis_dir, "finetune_metrics_summary.xlsx")
    if curve_path is None:
        curve_path = os.path.join(vis_dir, "finetune_metrics_curve.png")
    
    # 查找所有微调图像
    finetune_files = sorted([f for f in os.listdir(vis_dir) if f.startswith("finetune_epoch_") and f.endswith(".png")])
    
    if not finetune_files:
        print("未找到微调可视化文件")
        return
    
    print(f"找到 {len(finetune_files)} 个微调可视化文件")
    ssim_scores, psnr_scores, epochs = [], [], []
    
    # 从第一个图像获取尺寸信息
    sample_img = Image.open(os.path.join(vis_dir, finetune_files[0]))
    w, h = sample_img.size
    single_w = w // 4  # 假设图像有四个部分：素描、深度、预测和目标
    
    for f in finetune_files:
        # 从文件名提取轮次数
        match = re.search(r'epoch_(\d+)', f)
        if match:
            epoch = int(match.group(1))
        else:
            continue  # 如果无法提取轮次，跳过
            
        epochs.append(epoch)
        path = os.path.join(vis_dir, f)
        img = Image.open(path)
        
        # 提取预测结果和目标区域
        pred = img.crop((single_w*2, 0, single_w*3, h))
        gt = img.crop((single_w*3, 0, single_w*4, h))

        # 转换为浮点数组并确保在范围[0,1]内
        pred_np = np.asarray(pred).astype(np.float32) / 255.0
        gt_np = np.asarray(gt).astype(np.float32) / 255.0

        # 计算指标
        ssim_val = ssim(gt_np, pred_np, channel_axis=2, data_range=1.0)
        psnr_val = psnr(gt_np, pred_np, data_range=1.0)
        
        ssim_scores.append(ssim_val)
        psnr_scores.append(psnr_val)
    
    # 确保有记录再继续
    if not epochs:
        print("未能从文件名提取轮次信息，无法生成指标")
        return
        
    # 对数据按轮次排序
    sorted_data = sorted(zip(epochs, ssim_scores, psnr_scores))
    epochs, ssim_scores, psnr_scores = zip(*sorted_data)
    
    # 文本格式
    with open(metric_txt_path, "w") as f:
        for epoch, s, p in zip(epochs, ssim_scores, psnr_scores):
            f.write(f"epoch_{epoch:03d}.png: SSIM={s:.4f}, PSNR={p:.2f}dB\n")
        f.write(f"\nAverage SSIM: {np.mean(ssim_scores):.4f}\n")
        f.write(f"Average PSNR: {np.mean(psnr_scores):.2f} dB\n")
        f.write(f"Best SSIM: {np.max(ssim_scores):.4f} (Epoch {epochs[np.argmax(ssim_scores)]})\n")
        f.write(f"Best PSNR: {np.max(psnr_scores):.2f} dB (Epoch {epochs[np.argmax(psnr_scores)]})\n")

    # Markdown和Excel格式 - 与训练阶段风格保持一致
    data = []
    for epoch, s, p in zip(epochs, ssim_scores, psnr_scores):
        data.append([f"epoch_{epoch:03d}.png", s, p])
    
    df = pd.DataFrame(data, columns=["Epoch", "SSIM", "PSNR(dB)"])
    df.loc[len(df)] = ["**Average**", np.mean(ssim_scores), np.mean(psnr_scores)]
    
    with open(metric_md_path, "w") as f:
        f.write(df.to_markdown(index=False))
        
    df.to_excel(metric_xlsx_path, index=False)
    
    # 绘制指标曲线 - 生成两个子图，风格与训练损失曲线图相似
    plt.figure(figsize=(15, 6))
    
    # SSIM曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, ssim_scores, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("SSIM")
    plt.title("SSIM Curve during Fine-tuning")
    plt.grid(True)
    
    # PSNR曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, psnr_scores, marker='o', color='orange')
    plt.xlabel("Epoch")
    plt.ylabel("PSNR (dB)")
    plt.title("PSNR Curve during Fine-tuning")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(curve_path)
    plt.close()
    
    # 如果finetune输出图像足够多，还可以生成GIF
    if len(finetune_files) > 1:
        try:
            import imageio
            from natsort import natsorted
            
            frames = []
            sorted_files = natsorted(finetune_files)  # 确保正确的自然排序
            for f in sorted_files:
                frames.append(np.array(Image.open(os.path.join(vis_dir, f))))
                
            gif_path = os.path.join(vis_dir, "finetune_progress.gif")
            imageio.mimsave(gif_path, frames, fps=1)
            print(f"微调进度GIF已保存到 {gif_path}")
        except Exception as e:
            print(f"无法创建GIF: {e}")
    
    print(f"微调评估指标已保存到 {metric_txt_path}, {metric_md_path}, 和 {metric_xlsx_path}")
    print(f"指标曲线图已保存到 {curve_path}")