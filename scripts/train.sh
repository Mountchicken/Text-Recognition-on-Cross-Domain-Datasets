CUDA_VISIBLE_DEVICES=0,1 python new_main.py \
  --synthetic_train_data_dir ../text_recognition_datasets/NIPS2014/NIPS2014 ../text_recognition_datasets/CVPR2016 \
  --test_data_dir ../text_recognition_datasets/scene_text_benchmarks/IIIT5K_3000/ \
  --batch_size 512 \
  --workers 0 \
  --arch CRNN \
  --with_lstm \
  --max_len 25 \
