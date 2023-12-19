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


class SimpleCNN(Network):
    def __init__(self, *, emb_size, conv_num, conv_size, conv_off, linear_size, full_size, dropout=0.2, pooling=True, **kwargs):
        super(SimpleCNN, self).__init__(**kwargs)


        self.conv = nn.ModuleList(nn.Conv1d(int(emb_size), len(all_tfs), cs) for cn, cs in zip(conv_num, conv_size))        
        self.conv_bn = nn.ModuleList(nn.BatchNorm1d(len(all_tfs)) for cn in conv_num)

        self.conv_off = conv_off
        self.dropout = nn.Dropout(dropout)
        linear_size = [len(conv_num)*len(all_tfs)] + linear_size
        self.linear = nn.ModuleList([nn.Conv1d(in_s, out_s, 5, padding="same")
                                     for in_s, out_s in zip(linear_size[:-1], linear_size[1:])])
        
        self.linear_bn = nn.ModuleList([nn.BatchNorm1d(out_s) for out_s in linear_size[1:]])

        self.linear_s1 = nn.Conv1d(linear_size[-1],64,1)
        self.linear_bn_s1 = nn.BatchNorm1d(64)


        #full_size_first = [4096] # [linear_size[-1] * len(all_tfs) * 1024(DNA_len + 2*DNA_pad - conv_off - 2 * conv_size + 1) / 4**len(self.max_pool) ]
        full_size = full_size + [len(all_tfs)]
        self.full_connect = nn.ModuleList([nn.Linear(in_s, out_s) for in_s, out_s in zip(full_size[:-1], full_size[1:])])
        self.full_connect_bn = nn.ModuleList([nn.BatchNorm1d(out_s) for out_s in full_size[1:] ])

        self.reset_parameters()

    def forward(self, DNA_x, tf_x, pooling=None, **kwargs):
        DNA_x = super(SimpleCNN, self).forward(DNA_x, tf_x)
        DNA_x = torch.transpose(DNA_x,1,2)

        # ----------------do not apply conv off for same output dim then iconv  ----------------#
        conv_out = torch.cat([F.gelu(conv_bn(conv(DNA_x[:,:,off: DNA_x.shape[2] - off])))
                              for conv, conv_bn, off in zip(self.conv, self.conv_bn, self.conv_off)], dim=1)
        #conv_out = self.dropout(conv_out)

        #torch.Size([64, 145, 1024])
        conv_out = nn.functional.max_pool1d(conv_out,4,4)

        # ---------------------- covn1d tower with maxpool ---------------------- #
        linear_index = 0
        for linear, linear_bn in zip(self.linear, self.linear_bn):
            linear_index += 1
            conv_out = linear_bn(linear(conv_out))
            if linear_index == 1:
                #conv_out = F.gelu(nn.functional.max_pool1d(conv_out,2,2))
                conv_out = F.gelu(nn.functional.avg_pool1d(conv_out,4,4))
            else:
                #conv_out = F.gelu(nn.functional.max_pool1d(conv_out,2,2))
                conv_out = F.gelu(nn.functional.avg_pool1d(conv_out,4,4))

        # ---------------------- last conv1d with size 1  ---------------------- #
        conv_out = self.linear_bn_s1(self.linear_s1(conv_out))
        #conv_out = F.gelu(nn.functional.max_pool1d(conv_out,2,2))
        conv_out = F.gelu(nn.functional.avg_pool1d(conv_out,2,2))
        conv_out = self.dropout(conv_out)
        # ---------------- flatten and full connect ----------------#
        conv_out = torch.flatten(conv_out, start_dim = 1)

        full_index = 0
        for full, full_bn in zip(self.full_connect, self.full_connect_bn):
            full_index += 1
            conv_out = full_bn(F.gelu(full(conv_out)))
            if full_index == 1:
                conv_out = self.dropout(conv_out)
        #return torch.sigmoid(conv_out)
        return conv_out

    def reset_parameters(self):
        for conv, conv_bn in zip(self.conv, self.conv_bn):
            conv.reset_parameters()
            conv_bn.reset_parameters()
            nn.init.normal_(conv_bn.weight.data, mean=1.0, std=0.002)
        for linear, linear_bn in zip(self.linear, self.linear_bn):
            nn.init.trunc_normal_(linear.weight, std=0.02)
            nn.init.zeros_(linear.bias)
            linear_bn.reset_parameters()
            nn.init.normal_(linear_bn.weight.data, mean=1.0, std=0.002)
        for full_connect, full_connect_bn in zip(self.full_connect, self.full_connect_bn):
            #full_connect.reset_parameters()
            nn.init.trunc_normal_(full_connect.weight, std=0.02)
            nn.init.zeros_(full_connect.bias)
            full_connect_bn.reset_parameters()
            nn.init.normal_(full_connect_bn.weight.data, mean=1.0, std=0.002)
        nn.init.trunc_normal_(self.linear_s1.weight, std=0.02)
        nn.init.zeros_(self.linear_s1.bias)
        self.linear_bn_s1.reset_parameters()
        nn.init.normal_(self.linear_bn_s1.weight.data, mean=1.0, std=0.002)
        

