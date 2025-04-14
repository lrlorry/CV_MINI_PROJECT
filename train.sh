tmux new -s sketch_train -d "\
python3 train.py \
  --depth depth.png \
  --image jcsmr.jpg \
  --sketch sketch.jpg \
  --output models \
  --epochs 5 \
  --finetune_epochs 2 \
  --resume models/latest.pth \
  2>&1 | tee outputs/training.log"