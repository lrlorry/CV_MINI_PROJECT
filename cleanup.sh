echo "开始清理项目目录..."

# 要删除的目录列表
DIRS_TO_REMOVE=(
  "./metrics_finetune"
  "./outputs"
  "./metrics_train"
  "./results_original"
  "./results_v2"
  "./results_with_lab"
  "./all_styles_with_lab"
  "./models"
  "./results"
  "./final_model_result"
  "./finetune_model_result"
)

# 遍历并删除每个目录
for dir in "${DIRS_TO_REMOVE[@]}"; do
  if [ -d "$dir" ]; then
    echo "正在删除目录: $dir"
    rm -rf "$dir"
    echo "✓ 已删除: $dir"
  else
    echo "⚠️ 目录不存在，跳过: $dir"
  fi
done

echo "清理完成!"
echo "已删除的目录:"
for dir in "${DIRS_TO_REMOVE[@]}"; do
  echo " - $dir"
done