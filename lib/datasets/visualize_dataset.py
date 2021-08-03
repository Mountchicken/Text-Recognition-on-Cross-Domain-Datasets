import PIL.Image as Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.transforms.transforms import Resize
from lib.datasets.dataset import lmdbDataset, alignCollate
from lib.datasets.concatdataset import ConcatDataset
import numpy as np
import time

def get_data(data_dir, height, width, batch_size, workers, is_train, keep_ratio, alphabets, punc):
  if punc:
    alphabets += " ,.!?;':"
  if isinstance(data_dir, list):
    dataset_list = []
    for data_dir_ in data_dir:
      dataset_list.append(lmdbDataset(alphabets, data_dir_))
    dataset = ConcatDataset(dataset_list)
  else:
    dataset = lmdbDataset(alphabets, data_dir)
  print('total image: ', len(dataset))

  if is_train:
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=workers,
      shuffle=True, pin_memory=True, drop_last=True,
      collate_fn=alignCollate(imgH=height, imgW=width, keep_ratio=keep_ratio))
  else:
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=workers,
      shuffle=False, pin_memory=True, drop_last=False,
      collate_fn=alignCollate(imgH=height, imgW=width, keep_ratio=keep_ratio))

  return dataset, data_loader

if __name__ == '__main__':
    dataset,loader = get_data(
        '../HWDB2_1test/',
        #'../cute80_288',
        height = 32,
        width = 320,
        batch_size=1,
        workers=0,
        is_train=True,
        keep_ratio=True,
        alphabets='0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ',
        punc=True)
    
    for batch in loader:
        im, label = batch
        im = im.squeeze()
        trans = transforms.ToPILImage()
        im=trans(im)
        print(im.size)
        im.show()
        print(label)
        time.sleep(2)
