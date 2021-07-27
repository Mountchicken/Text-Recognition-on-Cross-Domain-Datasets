import torch
import cv2
import torchvision
import time
import sys

from torchvision import transforms
from config import get_args
from lib.models.model_builder import ModelBuilder
from lib.utils.labelmaps import CTCLabelConverter
from lib.utils.serialization import load_checkpoint
from lib.evaluation_metrics.metrics import get_str_list
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def predict(model,img):
  pass

if __name__ == '__main__':
  args = get_args(sys.argv[1:])
  # creat model
  model = ModelBuilder(arch=args.arch, rec_num_classes=len(args.alphabets)+1)
  checkpoint = load_checkpoint(args.resume)
  model.load_state_dict(checkpoint['state_dict'])
  model = model.to(device)
  model.eval()
  # creat converter
  converter = CTCLabelConverter(args.alphabets, args.max_len)
  # load img
  img = cv2.imread(args.image_path)
  transform = torchvision.transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.height, args.width)),
    transforms.ToTensor()
  ])
  img_tensor = transform(img).unsqueeze(dim=0).to(device)
  input_dict ={}
  input_dict['images'] = img_tensor
  # inferrence
  with torch.no_grad():
    time1=time.time()
    pred = model.inferrence(img_tensor)
  # convert ctc prediction
  pred_string, pred_score = get_str_list(pred, converter)
  time_cost = time.time() - time1
  print('Prediction: ',pred_string, 'Predcition Score: ',pred_score, 'Cost time: ',time_cost)