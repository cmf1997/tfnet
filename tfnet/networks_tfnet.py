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
import torch.nn.functional as F
import torch.nn as nn
import pdb
import os



__all__ = ['TFNet']

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# code
class Network(nn.Module):
    def __init__(self, **kwargs):
        super(Network, self).__init__()

    def forward(self, DNA_x, **kwargs):
        return DNA_x
    


class TFNet(Network):
    def __init__(self, *, emb_size, linear_size, full_size, dropouts, all_tfs, **kwargs):
        super(TFNet, self).__init__(**kwargs)

        in_channels = [int(emb_size)] + linear_size

        #self.conv = nn.Conv1d(in_channels[0], in_channels[-1], 26, 1)
        #self.conv_bn = nn.BatchNorm1d(in_channels[-1])

        self.conv = nn.ModuleList([nn.Conv1d(in_channel,out_channel, 8, 1) for in_channel,out_channel in zip(in_channels[:-1],in_channels[1:])])
        self.conv_bn = nn.ModuleList([nn.BatchNorm1d(out_channel) for out_channel in in_channels[1:]])     
      
        self.multihead_attn = nn.MultiheadAttention(embed_dim=in_channels[-1], num_heads=4, dropout=0, batch_first=True)

        self.bidirectional = nn.LSTM(input_size=in_channels[-1], hidden_size=in_channels[-1], num_layers=1, batch_first=True, bidirectional=True)

        full_size = full_size + [len(all_tfs)]
        self.full_connect = nn.ModuleList([nn.Linear(in_s, out_s) for in_s, out_s in zip(full_size[:-1], full_size[1:])])

        self.dropout = nn.ModuleList([nn.Dropout(dropout) for dropout in dropouts])

        self.all_tfs = all_tfs

    def forward(self, DNA_x, **kwargs):

        temp = torch.transpose(DNA_x,1,2)

        # ---------------- conv relu maxpool drop ----------------#
        #temp = self.dropout[0](F.max_pool1d(F.relu(self.conv_bn(self.conv(DNA_x))), 13, 13))        
        for index, (conv, conv_bn) in enumerate(zip(self.conv, self.conv_bn)):
            temp = F.relu(conv_bn(conv(temp)))
            if index == len(self.conv)-1:
                temp = self.dropout[1](temp)
            else:
                temp = F.max_pool1d(temp,4,4)
                temp = self.dropout[0](temp)      

        # ---------------- multihead attention ----------------#
        temp = temp.permute(0, 2, 1)
        temp, _ = self.multihead_attn(temp, temp, temp)

        # ---------------- bilstm ----------------#
        temp, _ = self.bidirectional(temp)
        temp = self.dropout[1](temp)

        # ---------------- full connect ----------------#
        temp = torch.flatten(temp, start_dim = 1)

        for index, full in enumerate(self.full_connect):
            temp = full(temp)
            if index != len(self.full_connect)-1:
                temp = F.relu(temp)

        return temp