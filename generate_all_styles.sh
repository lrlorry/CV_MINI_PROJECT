# 运行生成所有风格的命令
python3 process.py \
  --mode all_styles \
  --sketch sketch.jpg \
  --depth depth.png \
  --model models/final_model.pth \
  --output all_styles_output \
  --no_style

echo "所有风格已生成完成！结果保存在 all_styles_output 目录中。"