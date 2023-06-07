"""Encoder"""

import torch.nn as nn
from transformer import *
from ASPP import *
import torch
import torchvision
import torch.nn.functional as F
import numpy as np
from torchvision.models import Inception3, MobileNetV2, ResNet


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, in_channels, atrous_rates, encoder_norm)

    def forward(self, x):
        x = self.encoder(x)
        return x

if __name__ == '__main__':
    model = Encoder()
    print(model)