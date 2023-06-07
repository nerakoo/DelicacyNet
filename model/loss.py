"""WMAE loss"""

import torch.nn as nn
import torch

class WmaeLoss(nn.Module):
    def __init__(self, weights, device):
        super(WmaeLoss, self).__init__()
        self.weights = weights
        self.device= device 
    def forward(self, pred, targets):
        n = targets.shape[0]
        pred = pred['pred_logits']
        pred = torch.argmax(pred, dim=2)
        loss = 0
        Sum = 0
        for i in range(n):
            loss += self.weights[int(targets[i])] * abs(pred[i]-targets[i])
            Sum += self.weights[int(targets[i])]

        return loss/Sum