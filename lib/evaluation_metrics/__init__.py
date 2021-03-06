from __future__ import absolute_import

from .metrics import Accuracy, EditDistance, RecPostProcess, Accuracy_with_lexicon, EditDistance_with_lexicon, word_Accuracy


__factory = {
  'accuracy': Accuracy,
  'word_accuracy':word_Accuracy,
  'editdistance': EditDistance,
  'accuracy_with_lexicon': Accuracy_with_lexicon,
  'editdistance_with_lexicon': EditDistance_with_lexicon,
}

def names():
  return sorted(__factory.keys())

def factory():
  return __factory