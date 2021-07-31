CUDA_VISIBLE_DEVICES=0,1 python inferrence.py \
  --image_path test_images/c8.jpg \
  --arch CRNN_IAM \
  --with_lstm \
  --height 192 \
  --width 2048 \
  --max_len 128 \
  --resume runs/best_model/CASIA/model_best.pth.tar \
  --alphabets casia_360cc \
  --padresize
  