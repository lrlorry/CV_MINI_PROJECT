MY_IP=192.168.1.88
scp -i ~/.ssh/id_rsa -r paperspace@$MY_IP:/home/paperspace/CV_MINI_PROJECT/train_metrics/ .
scp -i ~/.ssh/id_rsa -r paperspace@$MY_IP:/home/paperspace/CV_MINI_PROJECT/metrics_finetune/ .
scp -i ~/.ssh/id_rsa -r paperspace@$MY_IP:/home/paperspace/CV_MINI_PROJECT/final_model_result/ .
scp -i ~/.ssh/id_rsa -r paperspace@$MY_IP:/home/paperspace/CV_MINI_PROJECT/finetune_model_result/ .
