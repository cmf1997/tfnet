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



__all__ = ['TFNet3']

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# code
class Network(nn.Module):
    def __init__(self, **kwargs):
        super(Network, self).__init__()

    def forward(self, DNA_x, **kwargs):
        return DNA_x
    

class BahdanauAttention(nn.Module):
	def __init__(self,in_features, hidden_units,out_features):
		super(BahdanauAttention,self).__init__()
		self.W1 = nn.Linear(in_features=in_features,out_features=hidden_units)
		self.W2 = nn.Linear(in_features=in_features,out_features=hidden_units)
		self.V = nn.Linear(in_features=hidden_units, out_features=out_features)

	def forward(self, hidden_states, values):
		hidden_with_time_axis = torch.unsqueeze(hidden_states,dim=1)
		score  = self.V(nn.Tanh()(self.W1(values)+self.W2(hidden_with_time_axis)))
		attention_weights = nn.Softmax(dim=1)(score)
		values = torch.transpose(values,1,2)   
		context_vector = torch.matmul(values,attention_weights)
		context_vector = torch.transpose(context_vector,1,2)
		return context_vector, attention_weights


class attention_tbinet(nn.Module):
    def __init__(self):
        super().__init__()
        self.Linear = nn.Linear(320, 1)

    def forward(self, x):
        source = x
        x = x.permute(0, 2, 1)  
        x = self.Linear(x) 
        x = x.permute(0, 2, 1) 
        x = F.softmax(x, dim=2) 
        x = x.permute(0, 2, 1)
        x = torch.mean(x, dim=2)
        
        x = x.unsqueeze(dim=1)
       
        x = x.repeat(1, 320, 1)
        return source * x
    

class TFNet3(Network):
    def __init__(self, *, emb_size, linear_size, full_size, dropouts, all_tfs, **kwargs):
        super(TFNet3, self).__init__(**kwargs)

        in_channels = [int(emb_size)] + linear_size

        #self.conv = nn.Conv1d(in_channels[0], in_channels[-1], 26, 1)
        #self.conv_bn = nn.BatchNorm1d(in_channels[-1])

        self.conv = nn.ModuleList([nn.Conv1d(in_channel,out_channel, 8, 1) for in_channel,out_channel in zip(in_channels[:-1],in_channels[1:])])
        self.conv_bn = nn.ModuleList([nn.BatchNorm1d(out_channel) for out_channel in in_channels[1:]])     
      
        self.multihead_attn = nn.MultiheadAttention(embed_dim=in_channels[-1], num_heads=4, dropout=0, batch_first=True)

        #self.attn = attention_tbinet()
        self.bidirectional = nn.LSTM(input_size=in_channels[-1], hidden_size=in_channels[-1], num_layers=1, batch_first=True, bidirectional=True)

        self.attn2 = BahdanauAttention(in_features=in_channels[-1]*2, hidden_units=in_channels[-1]*2, out_features = len(all_tfs))

        #full_size = full_size + [len(all_tfs)]
        #self.full_connect = nn.ModuleList([nn.Linear(in_s, out_s) for in_s, out_s in zip(full_size[:-1], full_size[1:])])
        for i in range(len(all_tfs)):
            setattr(self, "FC%d" %i, nn.Sequential(
                                        nn.Linear(in_features=in_channels[-1]*2,out_features=64),
                                        nn.ReLU(),
                                        nn.Linear(in_features=64,out_features=1),
                                        ))

        self.dropout = nn.ModuleList([nn.Dropout(dropout) for dropout in dropouts])
        self.all_tfs = all_tfs

    def forward(self, DNA_x, **kwargs):

        temp = torch.transpose(DNA_x,1,2)
        batch_size = temp.shape[0]

        # ---------------- conv relu maxpool drop ----------------#
        #temp = self.dropout[0](F.max_pool1d(F.relu(self.conv_bn(self.conv(DNA_x))), 13, 13))        
        for index, (conv, conv_bn) in enumerate(zip(self.conv, self.conv_bn)):
            temp = F.relu(conv_bn(conv(temp)))
            if index == len(self.conv)-1:
                temp = self.dropout[1](temp)
            else:
                temp = F.max_pool1d(temp,4,4)
                temp = self.dropout[0](temp)      

        temp = temp.permute(0, 2, 1)

        # ---------------- bilstm ----------------#
        temp, (h_n, c_n) = self.bidirectional(temp)
        #temp = self.dropout[0](temp)
        h_n = h_n.view(batch_size, temp.shape[-1])

        # ---------------- attention ----------------#
        temp, attention_weights = self.attn2(h_n, temp)
        #temp = self.dropout[0](temp)


        # ---------------- full connect ----------------#
        ''''
        temp = torch.flatten(temp, start_dim = 1)

        for index, full in enumerate(self.full_connect):
            temp = full(temp)
            if index != len(self.full_connect)-1:
                temp = F.relu(temp)
        '''
        outs = []
        for i in range(len(self.all_tfs)):
            FClayer = getattr(self, "FC%d"%i)
            y = FClayer(temp[:,i,:])
            y = torch.squeeze(y, dim=-1)
            outs.append(y)

        return torch.stack(outs,1)