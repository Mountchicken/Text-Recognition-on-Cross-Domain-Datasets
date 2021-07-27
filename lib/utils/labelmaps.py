from __future__ import absolute_import
import torch

class CTCLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, character, max_len=25):
        # character (str): set of the possible characters. 0123456789abcdefghijklmnopqrstuvwxyz
        dict_character = list(character)

        self.dict = {}
        for i, char in enumerate(dict_character):
            # NOTE: 0 is reserved for 'CTCblank' token required by CTCLoss
            self.dict[char] = i + 1 #{'0':1,'1':2,'2':3....} 相当于是在构建字典了

        self.character = ['-'] + dict_character  # blank '-' token for CTCLoss (index 0)
        self.max_len = max_len
        #['[CTCblank]','0','1','2',....]
    def encode(self, text):
        """convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
            batch_max_length: max length of text label in the batch. 25 by default

        output:
            text: text index for CTCLoss. [batch_size, batch_max_length]
            length: length of each text. [batch_size]
        """
        length = [len(s) for s in text]

        # The index used for padding (=0) would not affect the CTC loss calculation.
        batch_text = torch.LongTensor(len(text), self.max_len).fill_(0) #将所有的标签都用0填充至长度为25，
        for i, t in enumerate(text):
            text = list(t)
            text = [self.dict[char] for char in text]
            batch_text[i][:len(text)] = torch.LongTensor(text)
        return (batch_text, torch.IntTensor(length))

    def decode(self, text_index):
        """ convert text-index into text-label. """
        texts = []
        for text in text_index:
            char_list = []
            for i in range(self.max_len):
                if text[i] != 0 and (not (i > 0 and text[i - 1] == text[i])):  # removing repeated characters and blank.
                    char_list.append(self.character[text[i]])
            text = ''.join(char_list)
            texts.append(text)
        return texts