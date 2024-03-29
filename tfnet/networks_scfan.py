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

__all__ = ['scFAN']

# code
class Network(nn.Module):
    def __init__(self, **kwargs):
        super(Network, self).__init__()

    def forward(self, DNA_x, **kwargs):
        return DNA_x


class scFAN(Network):
    def __init__(self, *, emb_size, linear_size, full_size, dropouts, all_tfs, **kwargs):
        super(scFAN, self).__init__(**kwargs)

        in_channels = [int(emb_size)] + linear_size  # depend on embedding
        
        # ---------------------- nn.Conv2d size (1,8) for deepsea ---------------------- #
        self.conv = nn.ModuleList([nn.Conv2d(in_channel,out_channel,(1,8),(1,1)) for in_channel,out_channel in zip(in_channels[:-1],in_channels[1:])])
        self.conv_bn = nn.ModuleList([nn.BatchNorm2d(out_channel) for out_channel in in_channels[1:]])          

        full_size = full_size + [len(all_tfs)]
        self.full_connect = nn.ModuleList([nn.Linear(in_s, out_s) for in_s, out_s in zip(full_size[:-1], full_size[1:])])
        self.full_connect_bn = nn.ModuleList([nn.BatchNorm1d(out_s) for out_s in full_size[1:]] )
        
        self.dropout = nn.ModuleList([nn.Dropout(dropout) for dropout in dropouts])

        self.all_tfs = all_tfs

        self.reset_parameters()

    def forward(self, DNA_x, **kwargs):

        DNA_x = torch.transpose(DNA_x,1,2)
        DNA_x = DNA_x.unsqueeze(2)

        # ---------------------- due to padding in dataset.py---------------------- #
        #DNA_x = DNA_x[:,:,:,10:DNA_x.shape[3]-10] # uncomment if DNA_X is True
        conv_out = DNA_x

        # ----------------do not apply conv off for same output dim then iconv  ----------------#
        for index, (conv, conv_bn) in enumerate(zip(self.conv, self.conv_bn)):
            conv_out = F.relu(conv_bn(conv(conv_out)))
            if index == len(self.conv)-1:
                conv_out = self.dropout[1](conv_out)
            else:
                conv_out = F.max_pool2d(conv_out,(1,4),(1,4))
                conv_out = self.dropout[0](conv_out)

        # ---------------- flatten and full connect ----------------#
        conv_out = torch.flatten(conv_out, start_dim = 1)

        for index, (full, full_bn) in enumerate(zip(self.full_connect, self.full_connect_bn)):
            #conv_out = F.relu(full_bn(full(conv_out)))
            conv_out = full(conv_out)
            if index != len(self.full_connect)-1:
                conv_out = F.relu(conv_out)
        return conv_out
    

    def reset_parameters(self):
        for conv, conv_bn in zip(self.conv, self.conv_bn):
            conv.reset_parameters()
            conv_bn.reset_parameters()
            nn.init.normal_(conv_bn.weight.data, mean=1.0, std=0.002)
        for full_connect, full_connect_bn in zip(self.full_connect, self.full_connect_bn):
            nn.init.trunc_normal_(full_connect.weight, std=0.02)
            #nn.init.kaiming_normal_(full_connect.weight)
            nn.init.zeros_(full_connect.bias)
            full_connect_bn.reset_parameters()
            nn.init.normal_(full_connect_bn.weight.data, mean=1.0, std=0.002)