CUDA_VISIBLE_DEVICES=0,1 python inferrence.py \
  --image_path 1.png \
  --arch CRNN_Scene \
  --with_lstm \
  --height 32 \
  --width 100 \
  --max_len 25 \
  --lower \
  --resume runs/best_model/scene/model_best.pth.tar \
  --alphabets 0123456789abcdefghijklmnopqrstuvwxyz\
