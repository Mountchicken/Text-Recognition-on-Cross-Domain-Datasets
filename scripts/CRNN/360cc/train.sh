CUDA_VISIBLE_DEVICES=0,1 python new_main.py \
  --synthetic_train_data_dir ../text_recognition_datasets/360wchinese/360cc_train_lmdb\
  --test_data_dir ../text_recognition_datasets/360wchinese/360cc_test_lmdb \
  --batch_size 256 \
  --workers 0 \
  --arch ResNet_Scene \
  --decode_type CTC \
  --with_lstm \
  --height 32 \
  --width 280 \
  --max_len 70 \
  --epoch 6 \
  --stepLR 4 5 \
  --adamdelta \
  --lr 1 \
  --evaluation_metric accuracy \
  --alphabets casia_360cc \
  --iter_mode \
  --randomsequentialsampler \