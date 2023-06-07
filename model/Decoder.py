"""Decoder"""

import torch.nn as nn
from transformer import *

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(TransformerDecoderLayer)
    def forward(self, ):
        return x