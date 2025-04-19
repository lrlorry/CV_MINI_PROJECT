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
#   2>&1 | tee train_metrics/training.log"


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
#   2>&1 | tee train_metrics/training.log"

tmux new -s sketch_train -d "\
python3 train.py \
  --depth depth.png \
  --image jcsmr.jpg \
  --sketch sketch.jpg \
  --semantic “” \
  --output models \
  --epochs 10 \
  --finetune_epochs 10 \
  --resume models/latest.pth \
  --no_style \
  2>&1 | tee train_metrics/training.log"