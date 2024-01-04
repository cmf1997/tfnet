#!/usr/bin/env python
# -*- encoding: utf-8 -*-

'''
@File : simple_cnn.py
@Time : 2023/11/30 13:14:58
@Author : Cmf
@Version : 1.0
@Desc : None
'''

# here put the import lib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

from tfnet.data_utils import ACIDS
from tfnet.all_tfs import all_tfs
#from tfnet.modules import *

__all__ = ['SimpleCNN']

# code
class Network(nn.Module):
    def __init__(self, *, padding_idx=0, DNA_pad=10, tf_len=39, **kwargs):
        super(Network, self).__init__()
        self.DNA_pad, self.padding_idx, self.tf_len = DNA_pad, padding_idx, tf_len

    def forward(self, DNA_x, tf_x, *args, **kwargs):
        return DNA_x


class SimpleCNN_2d(Network):
    def __init__(self, *, emb_size, conv_num, conv_size, conv_off, linear_size, full_size, **kwargs):
        super(SimpleCNN_2d, self).__init__(**kwargs)

        in_channels = [int(emb_size)] + conv_num  # depend on embedding
        
        # ---------------------- nn.Conv2d size (1,8) for deepsea ---------------------- #
        self.conv = nn.ModuleList([nn.Conv2d(in_channel,out_channel,(1,9),(1,1),padding="same") for in_channel,out_channel in zip(in_channels[:-1],in_channels[1:])])
        self.conv_bn = nn.ModuleList([nn.BatchNorm2d(out_channel) for out_channel in in_channels[1:]])          
        self.conv_len = len(conv_num)

        #full_size_first = [4096] # [linear_size[-1] * len(all_tfs) * 1024(DNA_len + 2*DNA_pad - conv_off - 2 * conv_size + 1) / 4**len(self.max_pool) ]
        full_size = full_size + [len(all_tfs)]
        self.full_connect = nn.ModuleList([nn.Linear(in_s, out_s) for in_s, out_s in zip(full_size[:-1], full_size[1:])])
        self.full_connect_bn = nn.ModuleList([nn.BatchNorm1d(out_s) for out_s in full_size[1:]] )
        self.full_len = len(all_tfs)

        self.reset_parameters()

    def forward(self, DNA_x, tf_x, **kwargs):
        DNA_x = super(SimpleCNN_2d, self).forward(DNA_x, tf_x)

        DNA_x = torch.transpose(DNA_x,1,2)
        DNA_x = DNA_x.unsqueeze(2)

        # ---------------------- due to padding in dataset.py---------------------- #
        #DNA_x = DNA_x[:,:,:,10:DNA_x.shape[3]-10] # uncomment if DNA_X is True
        conv_out = DNA_x

        # ----------------do not apply conv off for same output dim then iconv  ----------------#
        conv_index = 0
        for conv, conv_bn in zip(self.conv, self.conv_bn):
            conv_index += 1
            conv_out = conv(conv_out)
            conv_out = conv_bn(conv_out)
            conv_out = nn.functional.relu(conv_out)
            if conv_index == self.conv_len:
                conv_out = nn.functional.dropout(conv_out,0.5)         
            else:
                conv_out = nn.functional.max_pool2d(conv_out,(1,4),(1,4))
                conv_out = nn.functional.dropout(conv_out,0.2)

        # ---------------- flatten and full connect ----------------#
        conv_out = torch.flatten(conv_out, start_dim = 1)

        full_index = 0
        for full, full_bn in zip(self.full_connect, self.full_connect_bn):
            full_index += 1
            conv_out = F.relu(full_bn(full(conv_out)))
            if full_index == self.full_len:
                conv_out = nn.functional.dropout(conv_out,0)
        return conv_out

    def reset_parameters(self):
        for conv, conv_bn in zip(self.conv, self.conv_bn):
            conv.reset_parameters()
            conv_bn.reset_parameters()
            nn.init.normal_(conv_bn.weight.data, mean=1.0, std=0.002)
        for full_connect, full_connect_bn in zip(self.full_connect, self.full_connect_bn):
            nn.init.trunc_normal_(full_connect.weight, std=0.02)
            nn.init.zeros_(full_connect.bias)
            full_connect_bn.reset_parameters()
            nn.init.normal_(full_connect_bn.weight.data, mean=1.0, std=0.002)