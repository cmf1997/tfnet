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
from tfnet.all_tfs import all_tfs
import pdb
import os



__all__ = ['TBiNet']

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# code
class Network(nn.Module):
    def __init__(self, **kwargs):
        super(Network, self).__init__()

    def forward(self, DNA_x, **kwargs):
        return DNA_x
    

class Lambda(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.mean(x, dim=2)

class attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.Linear = nn.Linear(320, 1)
        self.lamb = Lambda()

    def forward(self, x):
        source = x
        x = x.permute(0, 2, 1)  
        x = self.Linear(x) 
        x = x.permute(0, 2, 1) 
        x = F.softmax(x, dim=2) 
        x = x.permute(0, 2, 1)
        x = self.lamb(x)
        
        x = x.unsqueeze(dim=1)
       
        x = x.repeat(1, 320, 1)
        return source * x


class TBiNet(Network):
    def __init__(self, *, emb_size, linear_size, full_size, dropouts, **kwargs):
        super(TBiNet, self).__init__(**kwargs)

        in_channels = [int(emb_size)] + linear_size

        self.conv = nn.Conv1d(in_channels[0], in_channels[-1], 26, 1)
      
        self.attn = attention()

        self.bidirectional = nn.LSTM(input_size=320, hidden_size=320, num_layers=1, batch_first=True, bidirectional=True)

        full_size = full_size + [len(all_tfs)]
        self.full_connect = nn.ModuleList([nn.Linear(in_s, out_s) for in_s, out_s in zip(full_size[:-1], full_size[1:])])

        self.dropout = nn.ModuleList([nn.Dropout(dropout) for dropout in dropouts])

    def forward(self, DNA_x, tf_x, **kwargs):

        DNA_x = torch.transpose(DNA_x,1,2)

        # ---------------- conv relu maxpool drop ----------------#
        temp = self.dropout[0](F.max_pool1d(F.relu(self.conv(DNA_x)), 13, 13))        

        temp = self.attn(temp)
      
        temp = temp.permute(0, 2, 1)
        temp, _ = self.bidirectional(temp)
        temp = self.dropout[1](temp)
      
        temp = torch.flatten(temp, start_dim = 1)

        for index, full in enumerate(self.full_connect):
            temp = full(temp)
            if index != len(self.full_connect)-1:
                temp = F.relu(temp)

        return temp