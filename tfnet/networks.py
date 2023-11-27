#!/usr/bin/env python
# -*- encoding: utf-8 -*-

'''
@File : networks.py
@Time : 2023/11/09 11:21:52
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
from tfnet.modules import *

__all__ = ['TFNet']


# code
class Network(nn.Module):
    def __init__(self, *, emb_size=6, vocab_size=len(ACIDS), padding_idx=0, DNA_pad=10, tf_len=39, **kwargs):
        super(Network, self).__init__()
        self.tf_emb = nn.Embedding(vocab_size, emb_size)
        self.DNA_pad, self.padding_idx, self.tf_len = DNA_pad, padding_idx, tf_len

    def forward(self, DNA_x, tf_x, *args, **kwargs):
        return DNA_x, self.tf_emb(tf_x)

    def reset_parameters(self):
        nn.init.uniform_(self.tf_emb.weight, -0.1, 0.1)

class TFNet(Network):
    def __init__(self, *, conv_num, conv_size, conv_off, linear_size, full_size, dropout=0.5, pooling=True, **kwargs):
        super(TFNet, self).__init__(**kwargs)
        self.conv = nn.ModuleList(IConv(cn, cs, self.tf_len) for cn, cs in zip(conv_num, conv_size))        
        self.conv_bn = nn.ModuleList(nn.BatchNorm2d(cn) for cn in conv_num)

        self.conv_off = conv_off
        self.dropout = nn.Dropout(dropout)

        linear_size = [sum(conv_num)] + linear_size
        self.linear = nn.ModuleList([nn.Conv1d(in_s, out_s, 1)
                                     for in_s, out_s in zip(linear_size[:-1], linear_size[1:])])
        self.linear_bn = nn.ModuleList([nn.BatchNorm1d(out_s) for out_s in linear_size[1:]])

        self.max_pool = nn.ModuleList([nn.MaxPool2d(kernel_size = 2, stride = 2) for i in range(4)])

        #full_size_first = [4096] # [linear_size[-1] * len(all_tfs) * 1024(DNA_len + 2*DNA_pad - conv_off - 2 * conv_size + 1) / 2**2**len(self.max_pool) ]
        full_size = full_size + [len(all_tfs)]
        self.full_connect = nn.ModuleList([nn.Linear(in_s, out_s) for in_s, out_s in zip(full_size[:-1], full_size[1:])])
        self.full_connect_bn = nn.ModuleList([nn.BatchNorm1d(out_s) for out_s in full_size[1:] ])

        self.reset_parameters()

    def forward(self, DNA_x, tf_x, pooling=None, **kwargs):
        DNA_x, tf_x = super(TFNet, self).forward(DNA_x, tf_x)

        # ---------------- apply conv off for same output dim then iconv  ----------------#
        conv_out = torch.cat([conv_bn(F.relu(conv(DNA_x[:, off: DNA_x.shape[1] - off], tf_x)))
                              for conv, conv_bn, off in zip(self.conv, self.conv_bn, self.conv_off)], dim=1)
        conv_out = self.dropout(conv_out)

        # ---------------- reduce dim 1 by conv1d  ----------------#
        conv_out_linear =[]
        for conv_1 in conv_out.unbind(dim=-1):
            for linear, linear_bn in zip(self.linear, self.linear_bn):
                conv_1 = linear_bn(F.relu(linear(conv_1)))
            conv_out_linear.append(conv_1)
        conv_out = torch.stack(conv_out_linear,dim=-1)
        # ---------------- reduce dim -1，-2 by maxpool 2d ----------------#
        conv_out_max_pool =[]
        for conv_1 in conv_out.unbind(dim=-1):
            for max_pool in self.max_pool:
                conv_1 = max_pool(conv_1)
            conv_out_max_pool.append(conv_1)
        conv_out = torch.stack(conv_out_max_pool,dim=-1)
        conv_out = conv_out.view(conv_out.shape[0], -1,conv_out.shape[-1])

        # ---------------- flatten and output ----------------#
        conv_out = torch.flatten(conv_out, start_dim = 1)
        #print("shape before full connect", conv_out.shape)
        for full, full_bn in zip(self.full_connect, self.full_connect_bn):
            #print("shape after full connect", full(conv_out).shape)
            conv_out = full_bn(F.relu(full(conv_out)))
            #print("shape after full connect bn", conv_out.shape)
            conv_out = self.dropout(conv_out)
        return torch.sigmoid(conv_out)

    def reset_parameters(self):
        super(TFNet, self).reset_parameters()
        for conv, conv_bn in zip(self.conv, self.conv_bn):
            conv.reset_parameters()
            conv_bn.reset_parameters()
            nn.init.normal_(conv_bn.weight.data, mean=1.0, std=0.002)
        for linear, linear_bn in zip(self.linear, self.linear_bn):
            #truncated_normal_(linear.weight, std=0.02)
            nn.init.trunc_normal_(linear.weight, std=0.02)
            nn.init.zeros_(linear.bias)
            linear_bn.reset_parameters()
            nn.init.normal_(linear_bn.weight.data, mean=1.0, std=0.002)
        for full_connect, full_connect_bn in zip(self.full_connect, self.full_connect_bn):
            full_connect.reset_parameters()
            #nn.init.trunc_normal_(full_connect.weight, std=0.02)
            full_connect_bn.reset_parameters()
            nn.init.normal_(full_connect_bn.weight.data, mean=1.0, std=0.002)
