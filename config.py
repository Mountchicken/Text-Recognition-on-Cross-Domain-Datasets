from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
sys.path.append('./')

import os
import math
import argparse

parser = argparse.ArgumentParser(description="ASTER")
# data
parser.add_argument('--synthetic_train_data_dir', nargs='+', type=str, metavar='PATH')
parser.add_argument('--test_data_dir', type=str, metavar='PATH')
parser.add_argument('--alphabets',type=str,default='0123456789abcdefghijklmnopqrstuvwxyz')
parser.add_argument('--punc',action='store_true',help="add  ,.!?;': to the alphabets")
parser.add_argument('--height',type=int,default=192, help='height of the input image')
parser.add_argument('--width',type=int,default=2048, help='width of the input image')
parser.add_argument('--padresize',action='store_true',help='to use pad resize (recommend on iam , not on scene text)')
parser.add_argument('--lower',action='store_true',help='lower all labels')
parser.add_argument('--max_len',type=int,default=128, help='max decode and encode length')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--workers', type=int, default=1)
parser.add_argument('--keep_ratio', action='store_true', help='length fixed or lenghth variable.')
parser.add_argument('--augmentation', type=str,help='whether use data augmentation. [IAM]')
parser.add_argument('--lexicon_type', type=str, default='0', choices=['0', '50', '1k', 'full'], help='which lexicon associated to image is used.')
parser.add_argument('--image_path', type=str, default='', help='the path of single image, used in demo.py.')
# model
parser.add_argument('--arch', type=str, default='CRNN')
parser.add_argument('--dropout', type=float, default=0.7)
parser.add_argument('--n_group', type=int, default=1)
## lstm
parser.add_argument('--with_lstm', action='store_true', help='whether append lstm after cnn in the encoder part.')
# optimizer
parser.add_argument('--lr', type=float, default=3e-4,help="learning rate")
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=3e-4) # the model maybe under-fitting, 0.0 gives much better results.
parser.add_argument('--grad_clip', type=float, default=3.0)
parser.add_argument('--loss_weights', nargs='+', type=float, default=[1,1,1])
# training configs
parser.add_argument('--resume', type=str, default='', metavar='PATH')
parser.add_argument('--adamdelta',action='store_true',help='whether to use adamdelta(recommened on scene text)')
parser.add_argument('--randomsequentialsampler',action='store_true',help='in my case, i use random sequential sampler in scene text')

parser.add_argument('--evaluate', action='store_true', help="evaluation only")
parser.add_argument('--epochs', type=int, default= 400)
parser.add_argument('--stepLR',type=int, nargs='+', help='step learning rate, ex [100,200]')
parser.add_argument('--iter_mode', action='store_true', help="train on epoch model(small dataset) or on iteration model(large dataset)")
parser.add_argument('--start_save', type=int, default=0,help="start saving checkpoints after specific epoch")
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--cuda', default=True, type=bool,help='whether use cuda support.')
parser.add_argument('--tensorboard_freq_iter',type=int, default=10,help='frequency to log tensorbaord')
parser.add_argument('--evaluation_freq_iter',type=int, default=2000,help='frequency to log tensorbaord')
parser.add_argument('--tensorboard_freq_epoch',type=int, default=1,help='frequency to log tensorbaord')
parser.add_argument('--evaluation_freq_epoch',type=int, default=20,help='frequency to log tensorbaord')

# testing configs
parser.add_argument('--evaluation_metric', type=str, default='accuracy')
parser.add_argument('--evaluate_with_lexicon', action='store_true', default=False)
parser.add_argument('--beam_width', type=int, default=1) # something wrong with beam search, use grady search as defaults
# logs
parser.add_argument('--logs_dir',type=str)

def get_args(sys_args):
  global_args = parser.parse_args(sys_args)
  return global_args
