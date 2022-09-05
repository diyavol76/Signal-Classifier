import torch
import torch.nn as nn
import torch.nn.functional as F


class LoraModel(nn.Module):

    def __init__(self, conv_dim=64, n_res_blocks=6):
        super(LoraModel, self).__init__()
        self.conv1=nn.Conv2d()

    def forward(self, x):

        return x