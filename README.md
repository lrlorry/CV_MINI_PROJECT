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
mkdir -p models outputs

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
  2>&1 | tee outputs/training.log"

echo "训练已在tmux会话 'sketch_train' 中启动"
echo "可以通过 tmux attach -t sketch_train 连接到该会话"
echo "训练日志保存在 outputs/training.log"

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
python3 process.py --mode process --model models/final_model.pth --sketch sketch.jpg --depth depth.png --output results --block_size 128 --overlap 64 --palette abao --color_mode palette
python3 process.py --mode process --model models/final_model.pth --sketch sketch.jpg --depth depth.png --output results --block_size 128 --overlap 64 --color_mode original
# 使用特定颜色风格处理：
python3 process.py --mode process --model models/final_model.pth --sketch sketch.jpg --depth depth.png --output results --color_mode palette --palette cyberpunk

# 批量处理所有颜色方案：
python process.py --mode batch --model models/final_model.pth --sketch sketch.png --depth depth.png --output all_styles

python process.py --mode batch --model models/final_model.pth --sketch sketch.jpg --depth depth.png --output all_styles


python3 train.py --mode train --sketch sketch.png --depth depth.png --image jcsmr.jpg --output models --epochs 100 --finetune_epochs 10



python process.py --mode process --model models/final_model.pth --sketch sketch.png --depth depth.png --output output


scp -i ~/.ssh/id_rsa paperspace@184.105.3.72:/home/paperspace/CV_MINI_PROJECT/output/result_original_sketch.png .
scp -i ~/.ssh/id_rsa -r paperspace@184.105.5.162:/home/paperspace/CV_MINI_PROJECT/all_styles_with_lab/ .

scp -i ~/.ssh/id_rsa -r paperspace@184.105.4.46:/home/paperspace/CV_MINI_PROJECT/results/ .

scp -i ~/.ssh/id_rsa -r paperspace@184.105.3.39:/home/paperspace/CV_MINI_PROJECT/outputs .

chmod +x train.sh
bash train.sh
tmux attach -t sketch_train
cat outputs/training.log


chmod +x predict_full.sh


会不会是因为梯度爆炸，因为train.py为了vgg全部改成了tensor 张量


分块，vgg lab 调色盘/多种风格 数据增强 梯度爆炸 数据增强-后处理-块状伪影/色块边界 3d 素描图 深度图 finetune 语意切割