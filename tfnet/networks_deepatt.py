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
from torch import nn
import math
import pdb
import os



__all__ = ['DeepATT']

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
mps_device = torch.device("mps")

# code
class Network(nn.Module):
    def __init__(self, **kwargs):
        super(Network, self).__init__()

    def forward(self, DNA_x, **kwargs):
        return DNA_x
    

class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            q_d_model: int = 512,
            k_d_model: int = 512,
            v_d_model: int = 512,
            num_dimensions: int = 512,
            num_heads: int = 16,
            dropout_p: float = 0.1,
    ):
        super(MultiHeadAttention, self).__init__()
        assert num_dimensions % num_heads == 0, "num_dimensions % num_heads should be zero."
        self.num_dimensions = num_dimensions
        self.d_head = int(num_dimensions / num_heads)
        self.num_heads = num_heads
        self.sqrt_dim = math.sqrt(num_dimensions)

        self.wq = nn.Linear(q_d_model, num_dimensions)
        self.wk = nn.Linear(k_d_model, num_dimensions)
        self.wv = nn.Linear(v_d_model, num_dimensions)

        self.dropout = nn.Dropout(p=dropout_p)
        self.u_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        self.v_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        torch.nn.init.xavier_uniform_(self.u_bias)
        torch.nn.init.xavier_uniform_(self.v_bias)

        self.dense = nn.Linear(num_dimensions, num_dimensions)

    def forward(self, q, k, v, mask = None):
        batch_size = q.size(0)

        query = self.wq(q).view(batch_size, -1, self.num_heads, self.d_head)
        key = self.wk(k).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        value = self.wv(v).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)

        content_score = torch.matmul((query + self.u_bias).transpose(1, 2), key.transpose(2, 3))

        score = (content_score) / self.sqrt_dim

        if mask is not None:
            if mask.dtype == torch.bool:
                score.masked_fill_(mask, float('-inf'))
            else:
                score += mask

        attn = torch.softmax(score, -1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, value).transpose(1, 2)
        context = context.contiguous().view(batch_size, -1, self.num_dimensions)

        return self.dense(context), score
        

class DeepATT(Network):
    def __init__(self, *, emb_size, linear_size, full_size, dropouts, all_tfs, **kwargs):
        super(DeepATT, self).__init__(**kwargs)

        in_channels = [int(emb_size)] + linear_size
        self.conv = nn.Conv1d(in_channels[0], in_channels[-1], 30, 1)

        self.bidirectional = nn.LSTM(input_size=1024, hidden_size=512, num_layers=1, batch_first=True, bidirectional=True)

        self.category_encoding = torch.eye(len(all_tfs))[None, :, :]

        self.multi_head_attention = MultiHeadAttention(q_d_model=len(all_tfs), k_d_model=1024, v_d_model=1024,
                                                       num_dimensions=400, num_heads=4)

        self.full_connect = nn.ModuleList([nn.Linear(in_s, out_s) for in_s, out_s in zip(full_size[:-1], full_size[1:])])

        self.dropout = nn.ModuleList([nn.Dropout(dropout) for dropout in dropouts])

        self.all_tfs = all_tfs

    def forward(self, DNA_x, **kwargs):

        DNA_x = torch.transpose(DNA_x,1,2)

        batch_size = DNA_x.shape[0]

        # ---------------- conv relu maxpool drop ----------------#
        temp = self.dropout[0](F.max_pool1d(F.relu(self.conv(DNA_x)), 4, 4))
        temp = temp.transpose(1, 2)
    
        # ---------------- bilstm ----------------#
        temp, _ = self.bidirectional(temp)

        # ---------------------- attention ---------------------- #
        query = torch.tile(self.category_encoding, dims=[batch_size, 1, 1]).to(mps_device)
        temp, _ = self.multi_head_attention(q=query, k=temp, v=temp)

        temp = self.dropout[0](temp)

        # ---------------- flatten and full connect ----------------#
        for index, full in enumerate(self.full_connect):
            temp = full(temp)

        return temp.reshape([-1, len(self.all_tfs)])