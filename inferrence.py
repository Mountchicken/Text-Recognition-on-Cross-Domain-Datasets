import torch
import PIL.Image as Image
import torchvision
import time
import sys

from torchvision import transforms
from config import get_args
from lib.models.model_builder import ModelBuilder
from lib.utils.labelmaps import CTCLabelConverter
from lib.utils.serialization import load_checkpoint
from lib.evaluation_metrics.metrics import get_str_list
from lib.datasets.dataset import Padresize, resizeNormalize
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
  args = get_args(sys.argv[1:])
  if args.punc:
      args.alphabets += ' '
  model = ModelBuilder(arch=args.arch, rec_num_classes=len(args.alphabets)+1)
  checkpoint = load_checkpoint(args.resume)
  model.load_state_dict(checkpoint['state_dict'])
  model = model.to(device)
  model.eval()
  # creat converter
  converter = CTCLabelConverter(args.alphabets, args.max_len)
  # creat transform
  if args.padresize:
    print('using padresize')
    transform = Padresize(args.height, args.width)
  else:
    print('using normal resize')
    transform = resizeNormalize((args.width, args.height))
  # load img
  img = Image.open(args.image_path).convert('RGB')
  img = transform(img).unsqueeze(0).to(device)
  torchvision.utils.save_image(img,'transed_img.jpg')
  # inferrence
  with torch.no_grad():
    time1=time.time()
    pred = model.inferrence(img)
  # convert ctc prediction
  pred_string, pred_score = get_str_list(pred, converter)
  time_cost = time.time() - time1
  print('Prediction: ',pred_string, 'Predcition Score: ',pred_score, 'Cost time: ',time_cost)
