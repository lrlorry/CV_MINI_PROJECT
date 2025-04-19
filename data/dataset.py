import torch
import random
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import os

class SketchDepthPatchDataset(Dataset):
    def __init__(self, sketch_path, depth_path, target_path, semantic_path=None, 
            patch_size=256, n_patches=1000, augment=True):
        # 现有初始化保持不变
        self.sketch = Image.open(sketch_path).convert('L')
        self.depth = Image.open(depth_path).convert('L')
        self.target = Image.open(target_path).convert('RGB')

        # 确保所有图像尺寸相同
        self.width, self.height = self.sketch.size
        if self.depth.size != (self.width, self.height) or self.target.size != (self.width, self.height):
            raise ValueError("素描图、深度图和目标图尺寸必须相同")
        
        # 添加语义掩码
        self.semantic = None
        if semantic_path and semantic_path.strip() and os.path.exists(semantic_path):
            self.semantic = Image.open(semantic_path).convert('L')
            # 确保掩码尺寸与其他图像匹配
            if self.semantic.size != (self.width, self.height):
                self.semantic = self.semantic.resize((self.width, self.height), Image.NEAREST)
        
        # 确保所有图像尺寸相同
        self.width, self.height = self.sketch.size
        if self.depth.size != (self.width, self.height) or self.target.size != (self.width, self.height):
            raise ValueError("素描图、深度图和目标图尺寸必须相同")
        
        self.patch_size = patch_size
        self.n_patches = n_patches
        self.augment = augment
        
        # 基本变换
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # 转换到[-1,1]
        ])
        
        self.transform_color = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 转换到[-1,1]
        ])
        
        # 数据增强变换
        self.augment_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.RandomRotation(10),
        ])
        
        # 生成随机patch位置
        # 确保patch完全在图像内
        self.patches = []
        for _ in range(n_patches):
            x = random.randint(0, self.width - self.patch_size)
            y = random.randint(0, self.height - self.patch_size)
            self.patches.append((x, y))
    
    def __len__(self):
        return self.n_patches
    
    def __getitem__(self, idx):
        x, y = self.patches[idx]
        
        # 提取patch
        sketch_patch = self.sketch.crop((x, y, x + self.patch_size, y + self.patch_size))
        depth_patch = self.depth.crop((x, y, x + self.patch_size, y + self.patch_size))
        target_patch = self.target.crop((x, y, x + self.patch_size, y + self.patch_size))

        # 提取语义patch（如果有）
        semantic_patch = None
        if self.semantic is not None:
            semantic_patch = self.semantic.crop((x, y, x + self.patch_size, y + self.patch_size))

        # 数据增强 - 必须对所有三个图像应用相同的变换
        if self.augment:
            # 将三个图像堆叠为一个PIL图像进行同步变换
            combined = Image.new('RGB', (self.patch_size*3, self.patch_size))
            combined.paste(sketch_patch, (0, 0))
            combined.paste(depth_patch, (self.patch_size, 0))
            combined.paste(target_patch, (self.patch_size*2, 0))
            
            # 应用变换
            combined = self.augment_transforms(combined)
            
            # 分离变换后的图像
            sketch_patch = combined.crop((0, 0, self.patch_size, self.patch_size)).convert('L')
            depth_patch = combined.crop((self.patch_size, 0, self.patch_size*2, self.patch_size)).convert('L')
            target_patch = combined.crop((self.patch_size*2, 0, self.patch_size*3, self.patch_size))
        
        # 转换为张量
        sketch_tensor = self.transform(sketch_patch)
        depth_tensor = self.transform(depth_patch)
        target_tensor = self.transform_color(target_patch)
        
        # 转换语义掩码（如果有）
        semantic_tensor = None
        if semantic_patch is not None:
            semantic_tensor = self.transform(semantic_patch)
        
        result = {
            'sketch': sketch_tensor, 
            'depth': depth_tensor, 
            'target': target_tensor
        }
        
        if semantic_tensor is not None:
            result['semantic'] = semantic_tensor
            
        return result
   