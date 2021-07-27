from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
sys.path.append('./')

import os
import math
import argparse

parser = argparse.ArgumentParser(description="CRNN")
# data
parser.add_argument('--synthetic_train_data_dir', nargs='+', type=str, metavar='PATH')
parser.add_argument('--test_data_dir', type=str, metavar='PATH')
parser.add_argument('--alphabets',type=str,default='0123456789abcdefghijklmnopqrstuvwxyz')
parser.add_argument('--height',type=int,default=32, help='height of the input image')
parser.add_argument('--width',type=int,default=100, help='width of the input image')
parser.add_argument('--max_len',type=int,default=25, help='max decode and encode length')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--workers', type=int, default=1)
parser.add_argument('--keep_ratio', action='store_true', default=False,help='length fixed or lenghth variable.')
parser.add_argument('--aug', action='store_true', default=False,help='whether use data augmentation.')
parser.add_argument('--lexicon_type', type=str, default='0', choices=['0', '50', '1k', 'full'], help='which lexicon associated to image is used.')
parser.add_argument('--image_path', type=str, default='', help='the path of single image, used in demo.py.')
# model
parser.add_argument('--arch', type=str, default='CRNN')
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--n_group', type=int, default=1)
## lstm
parser.add_argument('--with_lstm', action='store_true', default=False,help='whether append lstm after cnn in the encoder part.')
# optimizer
parser.add_argument('--lr', type=float, default=1,help="learning rate")
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=0.0) # the model maybe under-fitting, 0.0 gives much better results.
parser.add_argument('--grad_clip', type=float, default=1.0)
parser.add_argument('--loss_weights', nargs='+', type=float, default=[1,1,1])
# training configs
parser.add_argument('--resume', type=str, default='', metavar='PATH')
parser.add_argument('--evaluate', action='store_true', help="evaluation only")
parser.add_argument('--epochs', type=int, default= 6)
parser.add_argument('--start_save', type=int, default=0,help="start saving checkpoints after specific epoch")
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--cuda', default=True, type=bool,help='whether use cuda support.')
parser.add_argument('--tensorboard_log_freq',type=int, default=10,help='frequency to log tensorbaord')
parser.add_argument('--evaluation_freq',type=int, default=1000,help='frequency to log tensorbaord')
# testing configs
parser.add_argument('--evaluation_metric', type=str, default='accuracy')
parser.add_argument('--evaluate_with_lexicon', action='store_true', default=False)
parser.add_argument('--beam_width', type=int, default=1) # something wrong with beam search, use grady search as defaults
# logs
parser.add_argument('--logs_dir',type=str)

def get_args(sys_args):
  global_args = parser.parse_args(sys_args)
  return global_args