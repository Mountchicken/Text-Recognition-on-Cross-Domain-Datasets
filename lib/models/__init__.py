from __future__ import absolute_import

from .backbone.ResNet import ResNet_CRNN_IAM, ResNet_CRNN_Scene, ResNet_CRNN_CASIA, ResNet_DAN_Scene_1D, ResNet_DAN_Scene_2D, ResNet_DAN_IAM
from .decoder.decoupled_attention import CAM_1D, CAM_2D, CAM_IAM
__factory = {
  'ResNet_IAM': ResNet_CRNN_IAM,
  'ResNet_Scene': ResNet_CRNN_Scene,
  'ResNet_CASIA': ResNet_CRNN_CASIA,
  'ResNet_DAN_1D': ResNet_DAN_Scene_1D ,
 'ResNet_DAN_2D': ResNet_DAN_Scene_2D ,
  'ResNet_DAN_IAM': ResNet_DAN_IAM ,
  'CAM_1D': CAM_1D,
  'CAM_2D': CAM_2D,
  'CAM_IAM': CAM_IAM

}

def names():
  return sorted(__factory.keys())


def create(name, *args, **kwargs):
  """Create a model instance.
  
  Parameters
  ----------
  name: str
    Model name. One of __factory
  pretrained: bool, optional
    If True, will use ImageNet pretrained model. Default: True
  num_classes: int, optional
    If positive, will change the original classifier the fit the new classifier with num_classes. Default: True
  with_words: bool, optional
    If True, the input of this model is the combination of image and word. Default: False
  """
  if name not in __factory:
    raise KeyError('Unknown model:', name)
  return __factory[name](*args, **kwargs)