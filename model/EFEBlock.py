"""Environment feature extraction modele"""
import os
import torch.nn as nn
import torch
from torch import Tensor
import torchvision
import torch.nn.functional as F
import numpy as np
from typing import Dict, List
from torchvision.models import Inception3, MobileNetV2, inception
from .Position_encoding import *
from utils.misc import NestedTensor

class EFEBlock(nn.Module):
    def __init__(self):
        super(EFEBlock, self).__init__()
        self.num_channels = 512
        # ResNet预训练模型
        submodule = torchvision.models.resnet50(pretrained=True)
        submodule = list(submodule.children())[:-2]
        self.ResNet = nn.Sequential(*submodule)
        # InceptionV3模型
        submodule = Inception3()
        submodule = list(submodule.children())[:-3]
        self.Inception3 = nn.Sequential(*submodule)
        # MobileNetV2模型
        submodule = MobileNetV2()
        submodule = list(submodule.children())[:-1]
        self.MobileNetV2 = nn.Sequential(*submodule)
        # Global Average Pooling
        self.GlobalAvgPooling = nn.AdaptiveAvgPool2d(8)
        
        self.resize = torch.nn.ConvTranspose2d(in_channels=1280, out_channels=2048, kernel_size=1, stride=1)
        self.conv1 = nn.Conv2d(2048, 512, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(1280, 512, kernel_size=1, stride=1)


    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        x_resnet = self.ResNet(x)
        x_resnet = self.conv1(x_resnet)
        #torch_resize = torchvision.transforms.Resize([299,299])
        #x_inception3 = torch_resize(x)
        #x_inception3 = self.Inception3(x_inception3)
        x_mobilenetv2 = self.MobileNetV2(x)
        x_mobilenetv2 = self.conv2(x_mobilenetv2)
        #x = torch.add(x_resnet, x_inception3)
        x = torch.add(x_resnet, x_mobilenetv2)
        x = torch.true_divide(x, torch.tensor([2]).to('cuda'))
        x = self.GlobalAvgPooling(x)
        
        m = tensor_list.mask
        assert m is not None
        mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
        out = NestedTensor(x, mask)
        return out

class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        x = self[0](tensor_list)
        pos = []
        pos.append(self[1](x).to(x.tensors.dtype))

        return x, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    backbone = EFEBlock()
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model

if __name__ == '__main__':
    model = EFEBlock()
    print(model)
