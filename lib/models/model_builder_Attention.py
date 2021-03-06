from __future__ import absolute_import
import sys
from torch import nn
from torch.nn import functional as F
from . import create
from .decoder.attention_recognition_head import AttentionRecognitionHead
from ..loss.sequenceCrossEntropyLoss import SequenceCrossEntropyLoss
from .rectification.tps_spatial_transformer import TPSSpatialTransformer
from .rectification.stn_head import STNHead

from config import get_args
global_args = get_args(sys.argv[1:])

class ModelBuilder_Att(nn.Module):
  """
  This is the integrated model.
  """
  def __init__(self, arch, rec_num_classes, sDim, attDim, max_len_labels, STN_ON=False):
    super(ModelBuilder_Att, self).__init__()

    self.arch = arch
    self.rec_num_classes = rec_num_classes
    self.sDim = sDim
    self.attDim = attDim
    self.max_len_labels = max_len_labels
    self.STN_ON = STN_ON
    self.tps_inputsize = global_args.tps_inputsize

    self.encoder = create(self.arch,
                      with_lstm=True,
                      n_group=global_args.n_group)
    encoder_out_planes = self.encoder.out_planes

    self.decoder = AttentionRecognitionHead(
                      num_classes=rec_num_classes,
                      in_planes=encoder_out_planes,
                      sDim=sDim,
                      attDim=attDim,
                      max_len_labels=max_len_labels)
    self.rec_crit = SequenceCrossEntropyLoss()

    if self.STN_ON:
      self.tps = TPSSpatialTransformer(
        output_image_size=tuple(global_args.tps_outputsize),
        num_control_points=global_args.num_control_points,
        margins=tuple(global_args.tps_margins))
      self.stn_head = STNHead(
        in_planes=3,
        num_ctrlpoints=global_args.num_control_points,
        activation=global_args.stn_activation)

  def forward(self, input_dict):
    return_dict = {}
    return_dict['loss'] = {}
    return_dict['output'] = {}

    x, rec_targets, rec_lengths = input_dict['images'], \
                                  input_dict['rec_targets'], \
                                  input_dict['rec_lengths']

    # rectification
    if self.STN_ON:
      # input images are downsampled before being fed into stn_head.
      stn_input = F.interpolate(x, self.tps_inputsize, mode='bilinear', align_corners=True)
      stn_img_feat, ctrl_points = self.stn_head(stn_input)
      x, _ = self.tps(x, ctrl_points)
      if not self.training:
        # save for visualization
        return_dict['output']['ctrl_points'] = ctrl_points
        return_dict['output']['rectified_images'] = x

    encoder_feats = self.encoder(x)
    encoder_feats = encoder_feats.contiguous()

    if self.training:
      rec_pred = self.decoder([encoder_feats, rec_targets, rec_lengths])
      loss_rec = self.rec_crit(rec_pred, rec_targets, rec_lengths)
      return_dict['loss']['loss_rec'] = loss_rec
    else:
      rec_pred = self.decoder.sample([encoder_feats,None,None])
      rec_pred_ = self.decoder([encoder_feats, rec_targets, rec_lengths])
      loss_rec = self.rec_crit(rec_pred_, rec_targets, rec_lengths)
      return_dict['loss']['loss_rec'] = loss_rec
      return_dict['output']['pred_rec'] = rec_pred

    return return_dict
  
  def inferrence(self, x):
    if self.STN_ON:
      # input images are downsampled before being fed into stn_head.
      stn_input = F.interpolate(x, self.tps_inputsize, mode='bilinear', align_corners=True)
      _, ctrl_points = self.stn_head(stn_input)
      x, _ = self.tps(x, ctrl_points)
    encoder_feats = self.encoder(x)
    encoder_feats = encoder_feats.contiguous()
    rec_pred = self.decoder.sample([encoder_feats,None,None])

    return (rec_pred, x)