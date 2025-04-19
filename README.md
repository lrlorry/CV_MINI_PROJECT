project/
├── config/
│   └── color_palettes.py       # 颜色方案定义
├── models/
│   ├── attention.py            # 自注意力模块
│   └── unet.py                 # U-Net模型定义
├── utils/
│   ├── color_utils.py          # 颜色处理相关函数
│   ├── image_utils.py          # 图像处理工具函数
│   ├── lab_processor.py        # Lab颜色空间处理
│   └── visualization.py        # 可视化函数
├── data/
│   └── dataset.py              # 数据集类
├── loss/
│   └── combined_loss.py        # 损失函数
├── process/
│   ├── high_res.py             # 高分辨率图像处理
│   └── sketch_depth.py         # 素描深度图生成
├── train.py                    # 训练脚本
├── process.py                  # 处理图像脚本
├── batch_process.py            # 批处理脚本
└── generate_styles.py          # 风格生成脚本


# train.sh
#!/bin/bash
# 训练脚本

# 创建必要的目录
mkdir -p models metrics_train

# 在tmux会话中运行训练，便于持久化
tmux new -s sketch_train -d "\
python train.py \
  --sketch $1 \
  --depth $2 \
  --image $3 \
  --output models \
  --epochs 100 \
  --finetune_epochs 10 \
  --use_lab \
  --use_vgg \
  2>&1 | tee metrics_train/training.log"

echo "训练已在tmux会话 'sketch_train' 中启动"
echo "可以通过 tmux attach -t sketch_train 连接到该会话"
echo "训练日志保存在 metrics_train/training.log"

# process.sh
#!/bin/bash
# 处理脚本

# 创建输出目录
mkdir -p results

# 原始处理 (无特殊颜色模式)
echo "使用原始颜色处理..."
python process.py \
  --model models/final_model.pth \
  --sketch $1 \
  --depth $2 \
  --output results_original \
  --use_lab

# 启用Lab颜色空间处理
echo "使用Lab颜色空间处理..."
python process.py \
  --model models/final_model.pth \
  --sketch $1 \
  --depth $2 \
  --output results_with_lab \
  --use_lab

# 生成所有风格和比较图
echo "生成所有风格..."
python generate_styles.py \
  --model models/final_model.pth \
  --sketch $1 \
  --depth $2 \
  --output all_styles \
  --use_lab

echo "处理完成! 请查看输出目录中的结果。"

# batch_process.sh
#!/bin/bash
# 批量处理脚本

# 创建输出目录
mkdir -p all_styles

# 批量处理所有颜色方案
python process.py \
  --model models/final_model.pth \
  --sketch $1 \
  --depth $2 \
  --output all_styles \
  --batch \
  --use_lab

echo "批量处理完成! 结果保存在 all_styles 目录下"



# my own command

# 训练模式（从已有素描和深度图训练模型）：
python U-Net-block.py --mode train --sketch sketch.png --depth depth.png --image original.png --output models --epochs 100 --finetune_epochs 10

# 处理模式（应用训练好的模型）：
python3 process.py --mode process --model models/final_model.pth --sketch sketch.jpg --depth depth.png --output results --block_size 512 --overlap 64 --palette abao --color_mode palette

# 使用特定颜色风格处理：
python3 process.py --mode process --model models/final_model.pth --sketch sketch.jpg --depth depth.png --output results --color_mode palette --palette cyberpunk

# 批量处理所有颜色方案：
python process.py --mode batch --model models/final_model.pth --sketch sketch.png --depth depth.png --output all_styles

python process.py --mode batch --model models/final_model.pth --sketch sketch.jpg --depth depth.png --output all_styles


python3 train.py --mode train --sketch sketch.png --depth depth.png --image jcsmr.jpg --output models --epochs 100 --finetune_epochs 10



chmod +x train.sh
bash train.sh
tmux attach -t sketch_train
cat metrics_train/training.log


chmod +x predict_full.sh


会不会是因为梯度爆炸，因为train.py为了vgg全部改成了tensor 张量

随机采样

base 素描图 深度图 分块 自注意力机制  l1
-样式编码器
-数据增强
-finetune
-语义分割
+vgg+CombinedLoss
-lab
混合 损失函数
+all
功能：

伪影-调试batchsize 和 overlapsize解决的，尝试了各种方法

断点保存  --resume models/latest.pth
，vgg lab 调色盘/多种风格 数据增强 梯度爆炸 数据增强-后处理-块状伪影/色块边界 3d 素描图 深度图 finetune 语义分割



绝对可以。对于这种情况，您可以在报告的 "方法" 和 "结果" 部分这样描述：

方法部分可以写：
"我们采用随机patch采样策略，通过在单一图像上随机裁剪patches来增加训练数据的多样性。这种方法不仅扩充了训练数据，还帮助模型学习图像的局部和整体特征。"
结果部分可以写：
"训练过程中，模型的性能指标呈现一定波动。这种波动是由于随机patch采样和单图像训练的固有特性。尽管指标存在波动，但总体趋势显示模型在学习和重建图像细节方面取得了进展。"
反思部分可以写：
"随机采样策略为单图像深度学习带来了挑战和机遇。未来的工作可以探索更稳定的采样方法，以减少训练过程中的波动性。"

这种写法既客观地描述了方法和结果，又为潜在的不完美之处提供了合理的解释。