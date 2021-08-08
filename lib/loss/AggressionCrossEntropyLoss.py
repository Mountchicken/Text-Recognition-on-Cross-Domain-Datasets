"""Pytorch implementation of Aggression Cross Entropy Loss"""
import torch
import torch.nn as nn

class ACE(nn.Module):
    def __init__(self):
        '''
        type: str, 1D or 2D
        '''
        self.type = type
    def forward(self, preds, labels):
        '''
        preds: softmax prediction, (B, T, num_classes+1) for 1D, (B, H, W, num_classes+1) for 2D
        labels: labels in ACE form
        '''
        batch_size = preds.shape[0]
        if len(preds.shape) == 4:
            T = preds.shape[1] * preds.shape[2]
        else:
            T = preds.shape[1]
        # reshape preds
        preds = preds.reshape(batch_size,T,-1)
        # compute num of blanks in label, labels[:,0] is the length of a label
        labels[:,0] = T -labels[:,0]
        # four formulas
        preds = torch.sum(dim=1)
        preds = preds / T
        labels = labels / T
        loss = (-torch.sum(torch.log(input)*labels)) / batch_size
        return loss
