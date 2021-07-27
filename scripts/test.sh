CUDA_VISIBLE_DEVICES=0,1 python new_main.py \
  --synthetic_train_data_dir ../text_recognition_datasets/NIPS2014/NIPS2014 \
  --test_data_dir ../text_recognition_datasets/scene_text_benchmarks/cocotextval_9896 \
  --batch_size 512 \
  --workers 0 \
  --arch CRNN \
  --with_lstm \
  --max_len 25 \
  --evaluate \
  --resume runs/train/exp53/weights/model_best.pth.tar