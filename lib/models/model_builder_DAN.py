from __future__ import absolute_import
import sys
from torch import nn
from torch.nn import functional as F
from . import create
from ..models.decoder.decoupled_attention import DTD
from ..loss.sequenceCrossEntropyLoss import SequenceCrossEntropyLoss

from config import get_args
global_args = get_args(sys.argv[1:])

class ModelBuilder_DAN(nn.Module):
  """
  This is the integrated model.
  """
  def __init__(self, arch, rec_num_classes,  max_len_labels):
    super(ModelBuilder_DAN, self).__init__()

    self.arch = arch
    self.rec_num_classes = rec_num_classes
    self.max_len_labels = max_len_labels
    self.backbone = create('ResNet_DAN_'+self.arch,
                      max_len_labels
                      )
    self.CAM = create('CAM_'+self.arch,
                      max_len_labels
                      )
    self.DTD = DTD(rec_num_classes,input_size=self.backbone.out_planes, hidden_size=512, max_decode_len=max_len_labels )
    self.rec_crit = SequenceCrossEntropyLoss()

  def forward(self, input_dict):
    return_dict = {}
    return_dict['loss'] = {}
    return_dict['output'] = {}

    x, rec_targets, rec_lengths = input_dict['images'], \
                                  input_dict['rec_targets'], \
                                  input_dict['rec_lengths']

    features = self.backbone(x)
    attention = self.CAM(features)
    if self.training:
      rec_pred = self.DTD(features[-1], attention, rec_targets, rec_lengths)
      loss_rec = self.rec_crit(rec_pred, rec_targets, rec_lengths)
      return_dict['loss']['loss_rec'] = loss_rec
    else:
      rec_pred = self.DTD.sample(features[-1], attention)
      rec_pred_ = self.DTD(features[-1], attention, rec_targets, rec_lengths)
      loss_rec = self.rec_crit(rec_pred_, rec_targets, rec_lengths)
      return_dict['loss']['loss_rec'] = loss_rec
      return_dict['output']['pred_rec'] = rec_pred
    return return_dict

  def inferrence(self, x):
    features = self.backbone(x)
    attention = self.CAM(features)
    rec_pred = self.DTD.sample(features[-1], attention)
    return rec_pred
  