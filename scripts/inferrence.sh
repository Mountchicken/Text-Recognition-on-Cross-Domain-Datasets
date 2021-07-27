CUDA_VISIBLE_DEVICES=0,1 python inferrence.py \
  --image_path 3.jpg \
  --arch CRNN \
  --with_lstm \
  --max_len 25 \
  --resume runs/train/best/weights/model_best.pth.tar