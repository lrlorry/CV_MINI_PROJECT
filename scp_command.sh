MY_IP=184.105.6.89
REMOTE_USER=paperspace
OUTPUT_DIR=./cv/base  # 本地输出目录

mkdir -p $OUTPUT_DIR  # 如果目录不存在则创建

scp -i ~/.ssh/id_rsa -r $REMOTE_USER@$MY_IP:/home/paperspace/CV_MINI_PROJECT/metrics_train/         $OUTPUT_DIR/
scp -i ~/.ssh/id_rsa -r $REMOTE_USER@$MY_IP:/home/paperspace/CV_MINI_PROJECT/metrics_finetune/      $OUTPUT_DIR/
scp -i ~/.ssh/id_rsa -r $REMOTE_USER@$MY_IP:/home/paperspace/CV_MINI_PROJECT/final_model_result/    $OUTPUT_DIR/
scp -i ~/.ssh/id_rsa -r $REMOTE_USER@$MY_IP:/home/paperspace/CV_MINI_PROJECT/finetune_model_result/ $OUTPUT_DIR/
