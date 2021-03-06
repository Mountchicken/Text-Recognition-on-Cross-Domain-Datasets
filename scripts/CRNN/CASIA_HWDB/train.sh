CUDA_VISIBLE_DEVICES=0,1 python new_main.py \
  --synthetic_train_data_dir ../text_recognition_datasets/CASIA_HWDB2022/HWDB2_0train ../text_recognition_datasets/CASIA_HWDB2022/HWDB2_1train ../text_recognition_datasets/CASIA_HWDB2022/HWDB2_2train \
  --test_data_dir ../text_recognition_datasets/CASIA_HWDB2022/HWDB2_1test \
  --batch_size 32 \
  --workers 4 \
  --arch ResNet_IAM \
  --decode_type CTC \
  --with_lstm \
  --height 192 \
  --width 2048 \
  --max_len 128 \
  --epoch 150 \
  --stepLR 80 120 \
  --padresize \
  --evaluation_metric word_accuracy \
  --alphabets casia_360cc \
