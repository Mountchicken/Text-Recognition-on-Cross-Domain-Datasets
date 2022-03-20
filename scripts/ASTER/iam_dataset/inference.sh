CUDA_VISIBLE_DEVICES=0,1 python inferrence.py \
  --image_path test_images/ta5.jpg \
  --arch ResNet_IAM \
  --decode_type Attention \
  --with_lstm \
  --height 192 \
  --width 2048 \
  --max_len 128 \
  --resume runs/best_model/ASTER/iam_dataset/model_best.pth.tar \
  --alphabets allcases \
  --punc \
  --padresize \
  --evaluate \
  