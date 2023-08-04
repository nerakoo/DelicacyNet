from typing import List
import cv2
import torch
from torch import nn
import matplotlib.pyplot as plt
import os
import numpy

class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

class DWConv(nn.Module):
    """DW conv"""

    def __init__(self, in_channel, out_channel, kernel_size, padding):
        super(DWConv, self).__init__()
        # deep conv
        self.depth_conv = nn.Conv2d(in_channel, in_channel, kernel_size=kernel_size, padding=padding, groups=in_channel)
        # every point conv
        self.point_conv = nn.Conv2d(in_channel, out_channel, kernel_size=1)

    def forward(self, x):
        return self.point_conv(self.depth_conv(x))


class Trans(nn.Module):
    """Trans Module"""
    def __init__(self, in_channel, out_channel, n_head=3, hidden_scale=1.5):
        super(Trans, self).__init__()

        self.n_groups = in_channel / n_head
        # assert self.n_groups % 1 == 0, "n_head err"
        self.n_groups = int(self.n_groups)
        self.n_head = n_head

        # DW conv
        self.dw_conv_list = nn.ModuleList()
        k = 3
        for n in range(n_head):
            self.dw_conv_list.append(DWConv(self.n_groups, self.n_groups, kernel_size=k, padding=k // 2))
            k += 2


        self.lw_conv_list = nn.ModuleList()
        k = 3
        hidden_channel = int(hidden_scale * self.n_head)
        for n in range(self.n_groups):
            self.lw_conv_list.append(nn.Conv2d(self.n_head, hidden_channel, kernel_size=k, padding=k // 2))
            k += 2

        self.conv1x1 = nn.Conv2d(hidden_channel, out_channel, kernel_size=1)

    def forward(self, x: List):
        x = [self.dw_conv_list[i](x[i]) for i in range(self.n_head)]
        x = torch.transpose(torch.stack(x, dim=1), 1, 2)
        x = torch.unbind(x, dim=1)
        x = torch.cat([self.lw_conv_list[i](x[i]) for i in range(len(x))])
        return self.conv1x1(x)



class EFE(nn.Module):
    def __init__(self, out_channel, n_head=4):
        super(EFE, self).__init__()

        in_channel = 4
        # linear layer
        self.linear = nn.Linear(224 * 224 * 3, 196)

        # convolutional layer
        self.conv_list = nn.ModuleList()

        # Number of channels per convolution
        self.c_conv = in_channel / n_head
        # assert self.c_conv % 1 == 0, "n_head err"
        self.c_conv = int(self.c_conv)
        self.n_head = n_head

        k = 1
        for n in range(n_head):
            self.conv_list.append(nn.Conv2d(self.c_conv, self.c_conv, kernel_size=k, padding=k // 2))
            k += 2

        self.trans = Trans(in_channel, out_channel * 4, n_head)

        self.gelu = nn.GELU()
        self.grn = GRN(out_channel * 4)
        self.l2 = nn.Linear(196 * 4, 196)

    def forward(self, x: torch.Tensor, visual=False):
        b, c, h, w = x.shape
        x = torch.reshape(x, (b, 1, -1))
        short_cut = self.linear(x)
        x = torch.reshape(short_cut, (b, 4, 7, 7))

        x = torch.chunk(x, self.n_head, dim=1)
        x = self.trans([self.conv_list[i](x[i]) for i in range(self.n_head)])

        if visual:
            self.visualization(x)

        x = self.gelu(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.grn(x)
        x = torch.reshape(x, (b, 1, -1))
        x = self.l2(x)
        x = x + short_cut


        return x


    def visualization(self, x):
        if not os.path.exists('feature'):
            os.mkdir('feature')
        for i in range(len(x)):
            feature = torch.reshape(x[i], (1, 28, 28))
            for c in range(len(feature)):
                plt.imshow(feature[c].cpu().detach().numpy())
                plt.savefig(f'feature/img{i}_feature{c}')
                print(f'save feature/img{i}_feature{c}')
                plt.show()



# x = torch.randn((3, 3, 224, 224), device='cuda')

# img = cv2.imread("1.webp")
# img = cv2.resize(img, (224, 224))
# img=numpy.transpose(img,(2,0,1))
# input=torch.from_numpy(img)
# input=torch.tensor(input,dtype=torch.float)
# input = input[None]

# import time
# start = time.time()
# EFE(4, 4)(input, True)
# print(time.time()-start)


