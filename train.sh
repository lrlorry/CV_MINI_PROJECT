# tmux new -s sketch_train -d "\
# python3 train.py \
#   --depth depth.png \
#   --image jcsmr.jpg \
#   --sketch sketch.jpg \
#   --output models \
#   --epochs 10 \
#   --finetune_epochs 10 \
#   --resume models/latest.pth \
#   --no_style \
#   2>&1 | tee metrics_train/training.log"


# tmux new -s sketch_train -d "\
# python3 train.py \
#   --depth depth.png \
#   --image jcsmr.jpg \
#   --sketch sketch.jpg \
#   --semantic semantic.png \
#   --output models \
#   --epochs 10 \
#   --finetune_epochs 10 \
#   --resume models/latest.pth \
#   --no_style \
#   2>&1 | tee metrics_train/training.log"
#!/bin/bash

declare -a dirs=(
  "metrics_train"
  "metrics_finetune"
  "final_model_result"
  "finetune_model_result"
)

tmux new -s sketch_train -d "\
python3 train.py \
  --depth depth.png \
  --image jcsmr.jpg \
  --sketch sketch.jpg \
  --semantic semantic.png \
  --output models \
  --epochs 100 \
  --finetune_epochs 50 \
  --resume models/latest.pth \
  --no_style \
  2>&1 | tee metrics_train/training.log"