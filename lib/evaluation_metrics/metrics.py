from __future__ import absolute_import

import numpy as np
import editdistance
import string
import math

import torch
import torch.nn.functional as F
from tqdm import tqdm
from ..utils import to_torch, to_numpy


def _normalize_text(text):
  text = ''.join(filter(lambda x: x in (string.digits + string.ascii_letters), text))
  return text.lower()

def beam_search(pred , beam_size=5): #only for single Image inferrence
    pred = pred.squeeze().cpu()
    T, C = pred.shape
    log_y = pred  
    beam = [([], 0)]
    for t in range(T):  # for every timestep
        new_beam = []
        for prefix, score in beam:
            for i in range(C):  # for every state
                new_prefix = prefix + [i]
                new_score = score + log_y[t, i]
                
                new_beam.append((new_prefix, new_score)) #            
        # top beam_size
        new_beam.sort(key=lambda x: x[1], reverse=True)
        beam = new_beam[:beam_size] #选取top bema_size 概率的路径
    return beam

def get_str_list(output,target, converter, beam_width=0):
  if beam_width == 0: # gready searcj
    output = F.softmax(output,dim=2) # B T C -> B T C
    score, predicted = output.max(2) # B T C -> B T
    score_list = torch.prod(score, dim=1).cpu().numpy().tolist()
    pred_list = converter.decode(predicted)
  else: #beam search
    path, score_list = beam_search(output, beam_width)[0]
    path_list=[]
    path_list.append(path)
    pred_list = converter.decode(path_list)
  #cheat 
  pred_list = [_normalize_text(pred) for pred in pred_list]
  targ_list = [_normalize_text(targ) for targ in target]
  return pred_list, targ_list, score_list

def _lexicon_search(lexicon, word):
  edit_distances = []
  for lex_word in lexicon:
    edit_distances.append(editdistance.eval(_normalize_text(lex_word), _normalize_text(word)))
  edit_distances = np.asarray(edit_distances, dtype=np.int)
  argmin = np.argmin(edit_distances)
  return lexicon[argmin]


def Accuracy(output, target, converter):
  pred_list, target_list, score_list = get_str_list(output,target, converter)
  acc_list = [(pred == targ) for pred, targ in zip(pred_list, target_list)]
  accuracy = 1.0 * sum(acc_list) / len(acc_list)
  return pred_list, score_list, accuracy

def word_Accuracy(output, target, converter):

  output = F.softmax(output,dim=2) # B T C -> B T C
  score, predicted = output.max(2) # B T C -> B T
  score_list = torch.prod(score, dim=1).cpu().numpy().tolist()
  pred_list = converter.decode(predicted)
  total_len = 0
  correct = 0
  for pred, label in zip(pred_list, target):
    if ' ' in label: # for english sentence
      pred = pred.split(' ')
      pred_len = len(pred)
      label = label.split(' ')
      label_len = len(label)
    else: # for chinese sentence
      pred = list(pred)
      pred_len = len(pred)
      label = list(label)
      label_len = len(label)

    total_len += len(label)
    
    min_len = min(pred_len, label_len)
    pred = pred[:min_len]
    label = label[:min_len]
    if min_len == 0:
      correct+=0
    else:
      correct += sum([(p == l) for p, l in zip(pred, label)])

  accuracy = correct / total_len
  return pred_list, score_list, accuracy


def Accuracy_with_lexicon(output, target, dataset=None, file_names=None):
  pred_list, targ_list = get_str_list(output, target, dataset)
  accuracys = []

  # with no lexicon
  acc_list = [(pred == targ) for pred, targ in zip(pred_list, targ_list)]
  accuracy = 1.0 * sum(acc_list) / len(acc_list)
  accuracys.append(accuracy)

  # lexicon50
  if len(file_names) == 0 or len(dataset.lexicons50[file_names[0]]) == 0:
    accuracys.append(0)
  else:
    refined_pred_list = [_lexicon_search(dataset.lexicons50[file_name], pred) for file_name, pred in zip(file_names, pred_list)]
    acc_list = [(pred == targ) for pred, targ in zip(refined_pred_list, targ_list)]
    accuracy = 1.0 * sum(acc_list) / len(acc_list)
    accuracys.append(accuracy)

  # lexicon1k
  if len(file_names) == 0 or len(dataset.lexicons1k[file_names[0]]) == 0:
    accuracys.append(0)
  else:
    refined_pred_list = [_lexicon_search(dataset.lexicons1k[file_name], pred) for file_name, pred in zip(file_names, pred_list)]
    acc_list = [(pred == targ) for pred, targ in zip(refined_pred_list, targ_list)]
    accuracy = 1.0 * sum(acc_list) / len(acc_list)
    accuracys.append(accuracy)

  # lexiconfull
  if len(file_names) == 0 or len(dataset.lexiconsfull[file_names[0]]) == 0:
    accuracys.append(0)
  else:
    refined_pred_list = [_lexicon_search(dataset.lexiconsfull[file_name], pred) for file_name, pred in zip(file_names, pred_list)]
    acc_list = [(pred == targ) for pred, targ in zip(refined_pred_list, targ_list)]
    accuracy = 1.0 * sum(acc_list) / len(acc_list)
    accuracys.append(accuracy)    

  return accuracys


def EditDistance(output, target, converter):
  pred_list, score_list = get_str_list(output, converter)

  ed_list = [editdistance.eval(pred, targ) for pred, targ in zip(pred_list, target)]
  eds = sum(ed_list) / len(''.join(target))
  return pred_list, score_list, eds

def EditDistance_with_lexicon(output, target, dataset=None, file_names=None):
  pred_list, targ_list = get_str_list(output, target, dataset)
  eds = []

  # with no lexicon
  ed_list = [editdistance.eval(pred, targ) for pred, targ in zip(pred_list, targ_list)]
  ed = sum(ed_list)
  eds.append(ed)

  # lexicon50
  if len(file_names) == 0 or len(dataset.lexicons50[file_names[0]]) == 0:
    eds.append(0)
  else:
    refined_pred_list = [_lexicon_search(dataset.lexicons50[file_name], pred) for file_name, pred in zip(file_names, pred_list)]
    ed_list = [editdistance.eval(pred, targ) for pred, targ in zip(refined_pred_list, targ_list)]
    ed = sum(ed_list)
    eds.append(ed)

  # lexicon1k
  if len(file_names) == 0 or len(dataset.lexicons1k[file_names[0]]) == 0:
    eds.append(0)
  else:
    refined_pred_list = [_lexicon_search(dataset.lexicons1k[file_name], pred) for file_name, pred in zip(file_names, pred_list)]
    ed_list = [editdistance.eval(pred, targ) for pred, targ in zip(refined_pred_list, targ_list)]
    ed = sum(ed_list)
    eds.append(ed)

  # lexiconfull
  if len(file_names) == 0 or len(dataset.lexiconsfull[file_names[0]]) == 0:
    eds.append(0)
  else:
    refined_pred_list = [_lexicon_search(dataset.lexiconsfull[file_name], pred) for file_name, pred in zip(file_names, pred_list)]
    ed_list = [editdistance.eval(pred, targ) for pred, targ in zip(refined_pred_list, targ_list)]
    ed = sum(ed_list)
    eds.append(ed)

  return eds


def RecPostProcess(output, target, score, dataset=None):
  pred_list, targ_list = get_str_list(output, target, dataset) #id转char
  max_len_labels = output.size(1)
  score_list = []

  score = to_numpy(score)
  for i, pred in enumerate(pred_list):
    len_pred = len(pred) + 1 # eos should be included
    len_pred = min(max_len_labels, len_pred) # maybe the predicted string don't include a eos.
    score_i = score[i,:len_pred]
    score_i = math.exp(sum(map(math.log, score_i)))
    score_list.append(score_i)

  return pred_list, targ_list, score_list