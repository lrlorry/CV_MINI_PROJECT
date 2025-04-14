import os
import torch
import numpy as np
from PIL import Image, ImageEnhance, ImageDraw,ImageFont
import torchvision.utils as vutils
import torchvision.transforms as T

# ==== 保存图像 ====
def save_image(tensor, filename):
    """Save tensor as image file"""
    # Ensure directory exists
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
    
    # Check for NaN or infinity values
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        print(f"Warning: NaN or Inf values detected in tensor before saving {filename}")
        # Replace NaN/Inf with zeros
        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # Convert from [-1,1] to [0,1]
    tensor = (tensor + 1) / 2
    tensor = tensor.clamp(0, 1)
    
    # Save image
    vutils.save_image(tensor, filename)

# ==== 保存拼接图像 ====
def save_triplet(sketch_path, depth_path, pred_tensor, epoch, loss, vis_dir="outputs/epoch_vis_full"):
    to_pil = T.ToPILImage()
    resize = T.Resize((256, 256))

    sketch = resize(Image.open(sketch_path).convert("RGB"))
    depth = resize(Image.open(depth_path).convert("L")).convert("RGB")
    pred = to_pil(pred_tensor.squeeze(0).cpu().clamp(0, 1))

    w, h = sketch.size
    canvas = Image.new("RGB", (w * 3, h + 25), "black")
    canvas.paste(sketch, (0, 25))
    canvas.paste(depth, (w, 25))
    canvas.paste(pred, (w * 2, 25))

    draw = ImageDraw.Draw(canvas)
    draw.text((10, 5), f"Epoch {epoch} | Loss: {loss:.4f}", fill=(255, 255, 255))

    os.makedirs(vis_dir, exist_ok=True)
    canvas.save(os.path.join(vis_dir, f"epoch_{epoch:03d}.png"))
