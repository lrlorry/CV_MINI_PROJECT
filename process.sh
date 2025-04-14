#!/bin/bash
# 使用Lab颜色空间处理进行推理

# 原始处理
python3 process.py \
  --mode process \
  --sketch sketch.jpg \
  --depth depth.png \
  --model models/final_model.pth \
  --output results_original \
  --color_mode original

# 启用Lab空间处理
python3 process.py \
  --mode process \
  --sketch sketch.jpg \
  --depth depth.png \
  --model models/final_model.pth \
  --output results_with_lab \
  --color_mode original \
  --use_lab

# 生成所有风格
python3 process.py \
  --mode all_styles \
  --sketch sketch.jpg \
  --depth depth.png \
  --model models/final_model.pth \
  --output all_styles_with_lab \
  --use_lab