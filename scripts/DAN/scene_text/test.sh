CUDA_VISIBLE_DEVICES=0,1 python new_main.py \
  --test_data_dir ../text_recognition_datasets/scene_text_benchmarks/cute80_288 \
  --batch_size 512 \
  --height 32 \
  --width 128 \
  --arch 1D \
  --decode_type DAN \
  --evaluate \
  --with_lstm \
  --max_len 25 \
  --alphabets allcases \
  --resume runs/train/exp13/weights/model_best.pth.tar \