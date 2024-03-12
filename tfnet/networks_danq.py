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
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

from tfnet.all_tfs import all_tfs

__all__ = ['Danq']

# code
class Network(nn.Module):
    def __init__(self, *, padding_idx=0, DNA_pad=10, tf_len=39, **kwargs):
        super(Network, self).__init__()
        self.DNA_pad, self.padding_idx, self.tf_len = DNA_pad, padding_idx, tf_len

    def forward(self, DNA_x, tf_x, *args, **kwargs):
        return DNA_x


class Danq(Network):
    def __init__(self, *, emb_size, linear_size, full_size, dropouts, **kwargs):
        super(Danq, self).__init__(**kwargs)

        in_channels = [int(emb_size)] + linear_size  # depend on embedding
        
        self.conv = nn.Conv1d(in_channels[0], in_channels[-1], 26, 1)

        self.rnn = nn.LSTM(in_channels[-1], in_channels[-1], num_layers=2, batch_first=True, bidirectional=True, dropout=0.5)

        full_size = full_size + [len(all_tfs)]
        self.full_connect = nn.ModuleList([nn.Linear(in_s, out_s) for in_s, out_s in zip(full_size[:-1], full_size[1:])])
        
        self.dropout = nn.ModuleList([nn.Dropout(dropout) for dropout in dropouts])

    def forward(self, DNA_x, tf_x, **kwargs):
        DNA_x = super(Danq, self).forward(DNA_x, tf_x)

        conv_out = torch.transpose(DNA_x,1,2)

        # ---------------- conv activate maxpool dropout ----------------#
        temp = self.dropout[0](F.max_pool1d(F.relu(self.conv(conv_out)), 13, 13))

        # ---------------------- bilstm ---------------------- #
        temp = torch.transpose(temp, 1, 2)
        temp, (h_n,h_c) = self.rnn(temp)

        # ---------------- flatten and full connect ----------------#
        temp = torch.flatten(temp, start_dim = 1)

        for index, full in enumerate(self.full_connect):
            temp = full(temp)
            if index != len(self.full_connect)-1:
                temp = F.relu(temp)
        return temp
    