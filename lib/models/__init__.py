from __future__ import absolute_import

from .backbone.resnet_aster import ResNet_ASTER

__factory = {
  'CRNN': ResNet_ASTER,
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