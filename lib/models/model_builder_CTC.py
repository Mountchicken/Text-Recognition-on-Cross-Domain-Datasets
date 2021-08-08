from __future__ import absolute_import
import sys
import einops
import torch
from torch import nn
import torch.nn.functional as F
from . import create
from torch.nn import CTCLoss
from config import get_args
global_args = get_args(sys.argv[1:])


class ModelBuilder_CTC(nn.Module):
  """
  This is the integrated model.
  """
  def __init__(self, arch, rec_num_classes):
    super(ModelBuilder_CTC, self).__init__()

    self.arch = arch
    self.rec_num_classes = rec_num_classes
    self.encoder = create(self.arch,
                      with_lstm=True,
                      n_group=global_args.n_group)
    self.decoder = nn.Linear(self.encoder.out_planes, rec_num_classes)
    self.rec_crit = CTCLoss(zero_infinity=True)


  def forward(self, input_dict):
    return_dict = {}
    return_dict['loss'] = {}
    return_dict['output'] = {}

    x, rec_targets, rec_lengths = input_dict['images'], \
                                  input_dict['rec_targets'], \
                                  input_dict['rec_lengths']

    feature = self.encoder(x) # rec_pred == CNN + BLSTM
    feature = feature.contiguous()
    rec_pred = self.decoder(feature)
    # compute ctc loss
    rec_pred = einops.rearrange(rec_pred, 'B T C -> T B C') # required by CTCLoss
    rec_pred_log_softmax = F.log_softmax(rec_pred,dim=2)
    pred_size = torch.IntTensor([rec_pred.shape[0]]*rec_pred.shape[1]) # (timestep) * batchsize
    loss_rec = self.rec_crit(rec_pred_log_softmax, rec_targets, pred_size, rec_lengths)
    return_dict['loss']['loss_rec'] = loss_rec
    if not self.training:
       return_dict['output']['pred_rec'] = einops.rearrange(rec_pred, 'T B C -> B T C')
    return return_dict

  def inferrence(self, x):
    feature = self.encoder(x) # rec_pred == CNN + BLSTM
    feature = feature.contiguous()
    rec_pred = self.decoder(feature)
    return rec_pred