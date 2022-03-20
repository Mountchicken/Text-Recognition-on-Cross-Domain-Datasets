CUDA_VISIBLE_DEVICES=0,1 python inferrence.py \
  --image_path test_images/as1.JPG \
  --arch ResNet_Scene \
  --decode_type Attention \
  --with_lstm \
  --height 64 \
  --width 256 \
  --max_len 50 \
  --resume runs/best_model/ASTER/scene/model_best.pth.tar \
  --alphabets allcases_symbols \
  --STN_ON \
  --tps_inputsize 32 64 \
  --tps_outputsize 32 100 \
  --tps_margins 0.05 0.05 \
  --stn_activation none \
  --num_control_points 20 \
  