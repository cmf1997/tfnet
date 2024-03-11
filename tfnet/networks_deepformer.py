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
from tfnet.all_tfs import all_tfs
from torch import nn, einsum
import pdb
import os



__all__ = ['DeepFormer']

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# code
class Network(nn.Module):
    def __init__(self, **kwargs):
        super(Network, self).__init__()

    def forward(self, DNA_x, **kwargs):
        return DNA_x
    

class Flow_Attention(nn.Module):
    def __init__(self, d_input, d_model, d_output, n_heads, drop_out=0.05, eps=5e-4):
        super(Flow_Attention, self).__init__()
        self.n_heads = n_heads
        self.query_projection = nn.Linear(d_input, d_model)
        self.key_projection = nn.Linear(d_input, d_model)
        self.value_projection = nn.Linear(d_input, d_model)
        self.out_projection = nn.Linear(d_model, d_output)
        self.dropout = nn.Dropout(drop_out)
        self.eps = eps

    def kernel_method(self, x):
        return torch.sigmoid(x)

    def dot_product(self, q, k, v):
        kv = torch.einsum("nhld,nhlm->nhdm", k, v)
        qkv = torch.einsum("nhld,nhdm->nhlm", q, kv)
        return qkv

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        queries = self.query_projection(queries).view(B, L, self.n_heads, -1)
        keys = self.key_projection(keys).view(B, S, self.n_heads, -1)
        values = self.value_projection(values).view(B, S, self.n_heads, -1)
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
      
        queries = self.kernel_method(queries)
        keys = self.kernel_method(keys)
        
       
        sink_incoming = 1.0 / (torch.einsum("nhld,nhd->nhl", queries + self.eps, keys.sum(dim=2) + self.eps))
        source_outgoing = 1.0 / (torch.einsum("nhld,nhd->nhl", keys + self.eps, queries.sum(dim=2) + self.eps))
        
        conserved_sink = torch.einsum("nhld,nhd->nhl", queries + self.eps,
                                      (keys * source_outgoing[:, :, :, None]).sum(dim=2) + self.eps)
        conserved_source = torch.einsum("nhld,nhd->nhl", keys + self.eps,
                                        (queries * sink_incoming[:, :, :, None]).sum(dim=2) + self.eps)
        conserved_source = torch.clamp(conserved_source, min=-1.0, max=1.0)  # for stability
       
        sink_allocation = torch.sigmoid(conserved_sink * (float(queries.shape[2]) / float(keys.shape[2])))
        source_competition = torch.softmax(conserved_source, dim=-1) * float(keys.shape[2])
        
        x = (self.dot_product(queries * sink_incoming[:, :, :, None],  # for value normalization
                              keys,
                              values * source_competition[:, :, :, None])  # competition
             * sink_allocation[:, :, :, None]).transpose(1, 2)  # allocation
       
        x = x.reshape(B, L, -1)
        x = self.out_projection(x)
        x = self.dropout(x)
        return x
        


class DeepFormer(Network):
    #def __init__(self, *, sequence_length, n_targets):
    def __init__(self, *, emb_size, linear_size, full_size, dropouts, **kwargs):
        super(DeepFormer, self).__init__(**kwargs)

        in_channels = [int(emb_size)] + linear_size
        self.conv = nn.ModuleList([nn.Conv1d(in_channel,out_channel, 8, 1) for in_channel,out_channel in zip(in_channels[:-1],in_channels[1:])])
        self.conv_bn = nn.ModuleList([nn.BatchNorm1d(out_channel) for out_channel in in_channels[1:]])     

        self.attn = Flow_Attention(54, 54, 54, 6)

        full_size = full_size + [len(all_tfs)]
        self.full_connect = nn.ModuleList([nn.Linear(in_s, out_s) for in_s, out_s in zip(full_size[:-1], full_size[1:])])
        self.full_connect_bn = nn.ModuleList([nn.BatchNorm1d(out_s) for out_s in full_size[1:]] )

        self.dropout = nn.ModuleList([nn.Dropout(dropout) for dropout in dropouts])

        self.reset_parameters()

    #def forward(self, DNA_x):
    def forward(self, DNA_x, tf_x, **kwargs):

        conv_out = torch.transpose(DNA_x,1,2)

        # ---------------- conv tower ----------------#
        for index, (conv, conv_bn) in enumerate(zip(self.conv, self.conv_bn)):
            conv_out = F.relu(conv_bn(conv(conv_out)))
            if index == len(self.conv)-1:
                conv_out = self.dropout[1](conv_out)
            else:
                conv_out = F.max_pool1d(conv_out,4,4)
                conv_out = self.dropout[0](conv_out)
    
        # ---------------- attention ----------------#
        attn_out = self.attn(conv_out, conv_out, conv_out)


        # ---------------- flatten and full connect ----------------#
        linear_out = torch.flatten(attn_out, start_dim = 1)

        for index, full in enumerate(self.full_connect):
            linear_out = full(linear_out)
            if index != len(self.full_connect)-1:
                linear_out = F.relu(linear_out)

        return linear_out
        
    def reset_parameters(self):
        for conv, conv_bn in zip(self.conv, self.conv_bn):
            conv.reset_parameters()
            conv_bn.reset_parameters()
            nn.init.normal_(conv_bn.weight.data, mean=1.0, std=0.002)
        for full_connect in self.full_connect:
            nn.init.trunc_normal_(full_connect.weight, std=0.02)
            nn.init.zeros_(full_connect.bias)