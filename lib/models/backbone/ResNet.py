import torch
import torch.nn as nn
import sys

from config import get_args
global_args = get_args(sys.argv[1:])


def conv3x3(in_planes, out_planes, stride=1):
  """3x3 convolution with padding"""
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                   padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
  """1x1 convolution"""
  return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class ResidualBlock(nn.Module):

  def __init__(self, inplanes, planes, stride=1, downsample=None):
    super(ResidualBlock, self).__init__()
    self.conv1 = conv1x1(inplanes, planes, stride)
    self.bn1 = nn.BatchNorm2d(planes)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = conv3x3(planes, planes)
    self.bn2 = nn.BatchNorm2d(planes)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)
    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)
    return out

class ResNet_CRNN_CASIA(nn.Module):
  def __init__(self, with_lstm=False, n_group=1):
    super(ResNet_CRNN_CASIA, self).__init__()
    self.with_lstm = with_lstm
    self.n_group = n_group

    in_channels = 3
    self.layer0 = nn.Sequential(
        nn.Conv2d(in_channels, 32, kernel_size=(3, 3), stride=1, padding=1, bias=False), # 32*512->32*512
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True))

    self.inplanes = 32
    self.layer1 = self._make_layer(32,  3, [2, 2]) # 32*512 -> 16*256
    self.layer2 = self._make_layer(64,  4, [2, 1]) # 16*256 -> 8 * 256
    self.layer3 = self._make_layer(128, 6, [2, 2]) # 8*256 -> 4 *128
    self.layer4 = self._make_layer(256, 6, [2, 1]) # 4*128-> 2*64
    self.layer5 = self._make_layer(512, 3, [2, 1]) # 2*64 -> 1* 64


    if with_lstm:
      self.rnn = nn.LSTM(512, 256, bidirectional=True, num_layers=2, batch_first=True)
      self.out_planes = 2 * 256
    else:
      self.out_planes = 512

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
      elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

  def _make_layer(self, planes, blocks, stride):
    downsample = None
    if stride != [1, 1] or self.inplanes != planes:
      downsample = nn.Sequential(
          conv1x1(self.inplanes, planes, stride),
          nn.BatchNorm2d(planes))

    layers = []
    layers.append(ResidualBlock(self.inplanes, planes, stride, downsample))
    self.inplanes = planes
    for _ in range(1, blocks):
      layers.append(ResidualBlock(self.inplanes, planes))
    return nn.Sequential(*layers)

  def forward(self, x):
    x0 = self.layer0(x)
    x1 = self.layer1(x0)
    x2 = self.layer2(x1)
    x3 = self.layer3(x2)
    x4 = self.layer4(x3)
    x5 = self.layer5(x4)
    cnn_feat = x5.squeeze(2) # [N, c, w]
    cnn_feat = cnn_feat.transpose(2, 1)
    if self.with_lstm:
      rnn_feat, _ = self.rnn(cnn_feat)
      return rnn_feat
    else:
      return cnn_feat

class ResNet_CRNN_IAM(nn.Module):

  def __init__(self, with_lstm=False, n_group=1):
    super(ResNet_CRNN_IAM, self).__init__()
    self.with_lstm = with_lstm
    self.n_group = n_group

    in_channels = 3
    self.layer0 = nn.Sequential(
        nn.Conv2d(in_channels, 32, kernel_size=(3, 3), stride=2, padding=1, bias=False), # 192*2048 -> 96*1024
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True))

    self.inplanes = 32
    self.layer1 = self._make_layer(32,  3, [2, 2]) # 96*1024 -> 48*512
    self.layer2 = self._make_layer(64,  4, [2, 1]) # 48*512 -> 24 * 512
    self.layer3 = self._make_layer(128, 6, [2, 2]) # 24*512 -> 12 *256
    self.layer4 = self._make_layer(256, 6, [2, 2]) # 12*256 -> 6*128
    self.layer5 = self._make_layer(512, 3, [2, 1]) # 6*128 -> 3* 128
    self.layer6 = nn.Sequential(
        nn.Conv2d(512, 512, (3, 1), padding=0, stride=1),
        nn.BatchNorm2d(512),
        nn.ReLU(inplace=True)
    ) # 3 * 128 -> 1* 128

    if with_lstm:
      self.rnn = nn.LSTM(512, 256, bidirectional=True, num_layers=2, batch_first=True)
      self.out_planes = 2 * 256
    else:
      self.out_planes = 512

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
      elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

  def _make_layer(self, planes, blocks, stride):
    downsample = None
    if stride != [1, 1] or self.inplanes != planes:
      downsample = nn.Sequential(
          conv1x1(self.inplanes, planes, stride),
          nn.BatchNorm2d(planes))

    layers = []
    layers.append(ResidualBlock(self.inplanes, planes, stride, downsample))
    self.inplanes = planes
    for _ in range(1, blocks):
      layers.append(ResidualBlock(self.inplanes, planes))
    return nn.Sequential(*layers)

  def forward(self, x):
    x0 = self.layer0(x)
    x1 = self.layer1(x0)
    x2 = self.layer2(x1)
    x3 = self.layer3(x2)
    x4 = self.layer4(x3)
    x5 = self.layer5(x4)
    x6 = self.layer6(x5)
    cnn_feat = x6.squeeze(2) # [N, c, w]
    cnn_feat = cnn_feat.transpose(2, 1)
    if self.with_lstm:
      rnn_feat, _ = self.rnn(cnn_feat)
      return rnn_feat
    else:
      return cnn_feat

class ResNet_CRNN_Scene(nn.Module):
  def __init__(self, with_lstm=False, n_group=1):
    super(ResNet_CRNN_Scene, self).__init__()
    self.with_lstm = with_lstm
    self.n_group = n_group

    in_channels = 3
    self.layer0 = nn.Sequential(
        nn.Conv2d(in_channels, 32, kernel_size=(3, 3), stride=1, padding=1, bias=False),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True))

    self.inplanes = 32
    self.layer1 = self._make_layer(32,  3, [2, 2]) # [16, 50]
    self.layer2 = self._make_layer(64,  4, [2, 2]) # [8, 25]
    self.layer3 = self._make_layer(128, 6, [2, 1]) # [4, 25]
    self.layer4 = self._make_layer(256, 6, [2, 1]) # [2, 25]
    self.layer5 = self._make_layer(512, 3, [2, 1]) # [1, 25]

    if with_lstm:
      self.rnn = nn.LSTM(512, 256, bidirectional=True, num_layers=2, batch_first=True)
      self.out_planes = 2 * 256
    else:
      self.out_planes = 512

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
      elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

  def _make_layer(self, planes, blocks, stride):
    downsample = None
    if stride != [1, 1] or self.inplanes != planes:
      downsample = nn.Sequential(
          conv1x1(self.inplanes, planes, stride),
          nn.BatchNorm2d(planes))

    layers = []
    layers.append(ResidualBlock(self.inplanes, planes, stride, downsample))
    self.inplanes = planes
    for _ in range(1, blocks):
      layers.append(ResidualBlock(self.inplanes, planes))
    return nn.Sequential(*layers)

  def forward(self, x):
    x0 = self.layer0(x)
    x1 = self.layer1(x0)
    x2 = self.layer2(x1)
    x3 = self.layer3(x2)
    x4 = self.layer4(x3)
    x5 = self.layer5(x4)

    cnn_feat = x5.squeeze(2) # [N, c, w]
    cnn_feat = cnn_feat.transpose(2, 1)
    if self.with_lstm:
      rnn_feat, _ = self.rnn(cnn_feat)
      return rnn_feat
    else:
      return cnn_feat

class ResNet_DAN_Scene_1D(nn.Module):
  def __init__(self, n_group=1):
    super(ResNet_DAN_Scene_1D, self).__init__()
    self.n_group = n_group

    in_channels = 3
    self.layer0 = nn.Sequential(
        nn.Conv2d(in_channels, 32, kernel_size=(3, 3), stride=1, padding=1, bias=False), # 32*128->32*128
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True))

    self.inplanes = 32
    self.layer1 = self._make_layer(32,  3, [2, 2]) # 32*128 -> 16*64 preserved
    self.layer2 = self._make_layer(64,  4, [2, 2]) # 16*64 -> 8 * 32 preserved
    self.layer3 = self._make_layer(128, 6, [2, 1]) # 8*32 -> 4 *32 preserved
    self.layer4 = self._make_layer(256, 6, [2, 1]) # 4*32-> 2*32 preserved
    self.layer5 = self._make_layer(512, 3, [2, 1]) # 2*32 -> 1* 32 preserved

    self.out_planes = 512

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
      elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

  def _make_layer(self, planes, blocks, stride):
    downsample = None
    if stride != [1, 1] or self.inplanes != planes:
      downsample = nn.Sequential(
          conv1x1(self.inplanes, planes, stride),
          nn.BatchNorm2d(planes))

    layers = []
    layers.append(ResidualBlock(self.inplanes, planes, stride, downsample))
    self.inplanes = planes
    for _ in range(1, blocks):
      layers.append(ResidualBlock(self.inplanes, planes))
    return nn.Sequential(*layers)

  def forward(self, x):
    feature_maps = []
    x0 = self.layer0(x)
    x1 = self.layer1(x0)
    feature_maps.append(x1)
    x2 = self.layer2(x1)
    feature_maps.append(x2)
    x3 = self.layer3(x2)
    feature_maps.append(x3)
    x4 = self.layer4(x3)
    feature_maps.append(x4)
    x5 = self.layer5(x4)
    feature_maps.append(x5)
    return  feature_maps

class ResNet_DAN_Scene_2D(nn.Module):
  def __init__(self, n_group=1):
    super(ResNet_DAN_Scene_2D, self).__init__()
    self.n_group = n_group

    in_channels = 3
    self.layer0 = nn.Sequential(
        nn.Conv2d(in_channels, 32, kernel_size=(3, 3), stride=1, padding=1, bias=False), # 32*128->32*128
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True))

    self.inplanes = 32
    self.layer1 = self._make_layer(32,  3, [2, 2]) # 32*128 -> 16*64 preserved
    self.layer2 = self._make_layer(64,  4, [1, 1]) # 16*64 -> 16*64
    self.layer3 = self._make_layer(128, 6, [2, 2]) # 16*64 -> 8*32 preserved
    self.layer4 = self._make_layer(256, 6, [1, 1]) # 8*32-> 8*32
    self.layer5 = self._make_layer(512, 3, [1, 1]) # 8*32 -> 8* 32 preserved

    self.out_planes = 512

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
      elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

  def _make_layer(self, planes, blocks, stride):
    downsample = None
    if stride != [1, 1] or self.inplanes != planes:
      downsample = nn.Sequential(
          conv1x1(self.inplanes, planes, stride),
          nn.BatchNorm2d(planes))

    layers = []
    layers.append(ResidualBlock(self.inplanes, planes, stride, downsample))
    self.inplanes = planes
    for _ in range(1, blocks):
      layers.append(ResidualBlock(self.inplanes, planes))
    return nn.Sequential(*layers)

  def forward(self, x):
    feature_maps = []
    x0 = self.layer0(x)
    x1 = self.layer1(x0)
    feature_maps.append(x1)
    x2 = self.layer2(x1)
    x3 = self.layer3(x2)
    feature_maps.append(x3)
    x4 = self.layer4(x3)
    x5 = self.layer5(x4)
    feature_maps.append(x5)
    return  feature_maps

class ResNet_DAN_IAM(nn.Module):
  def __init__(self, n_group=1):
      super(ResNet_DAN_IAM, self).__init__()
      self.n_group = n_group

      in_channels = 3
      self.layer0 = nn.Sequential(
          nn.Conv2d(in_channels, 32, kernel_size=(3, 3), stride=[2,2], padding=1, bias=False), # 192*2048->96*1024
          nn.BatchNorm2d(32),
          nn.ReLU(inplace=True))

      self.inplanes = 32
      self.layer1 = self._make_layer(32,  3, [2, 2]) # 96*1024-> 48*512 preserved
      self.layer2 = self._make_layer(64,  4, [2, 1]) # 48*512 -> 24*512 preserved
      self.layer3 = self._make_layer(128, 6, [2, 2]) # 24*512 -> 12*256 preserved
      self.layer4 = self._make_layer(256, 6, [2, 2]) # 12*256-> 6*128 preserved
      self.layer5 = self._make_layer(512, 3, [2, 1]) # 6*128 -> 3*128 preserved
      self.layer6 = nn.Sequential(
        nn.Conv2d(512, 512, (3, 1), padding=0, stride=1),
        nn.BatchNorm2d(512),
        nn.ReLU(inplace=True)) # 3 * 128 -> 1* 128 preserved

      self.out_planes = 512

      for m in self.modules():
        if isinstance(m, nn.Conv2d):
          nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, nn.BatchNorm2d):
          nn.init.constant_(m.weight, 1)
          nn.init.constant_(m.bias, 0)

  def _make_layer(self, planes, blocks, stride):
    downsample = None
    if stride != [1, 1] or self.inplanes != planes:
      downsample = nn.Sequential(
          conv1x1(self.inplanes, planes, stride),
          nn.BatchNorm2d(planes))

    layers = []
    layers.append(ResidualBlock(self.inplanes, planes, stride, downsample))
    self.inplanes = planes
    for _ in range(1, blocks):
      layers.append(ResidualBlock(self.inplanes, planes))
    return nn.Sequential(*layers)

  def forward(self, x):
    feature_maps = []
    x0 = self.layer0(x)
    x1 = self.layer1(x0)
    feature_maps.append(x1)
    x2 = self.layer2(x1)
    feature_maps.append(x2)
    x3 = self.layer3(x2)
    feature_maps.append(x3)
    x4 = self.layer4(x3)
    feature_maps.append(x4)
    x5 = self.layer5(x4)
    feature_maps.append(x5)
    x6 = self.layer6(x5)
    feature_maps.append(x6)
    return feature_maps

if __name__ == "__main__":
  x = torch.randn(3, 3, 32, 100)
  net = ResNet_CRNN_Scene(with_lstm=True)
  encoder_feat = net(x)
  print(encoder_feat.size())