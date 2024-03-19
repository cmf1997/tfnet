import torch
import torch.nn.functional as F
import torch.nn as nn
from tfnet.all_tfs import all_tfs
import pdb
import os



class BahdanauAttention(nn.Module):
	def __init__(self, in_features, hidden_units, out_features):
		super(BahdanauAttention,self).__init__()
		self.W1 = nn.Linear(in_features=in_features,out_features=hidden_units)
		self.W2 = nn.Linear(in_features=in_features,out_features=hidden_units)
		self.V = nn.Linear(in_features=hidden_units, out_features=out_features)

	def forward(self, hidden_states, values):
		hidden_with_time_axis = torch.unsqueeze(hidden_states,dim=1)
		score  = self.V(nn.Tanh()(self.W1(values)+self.W2(hidden_with_time_axis)))
		attention_weights = nn.Softmax(dim=1)(score)
		values = torch.transpose(values,1,2)   # transpose to make it suitable for matrix multiplication
		#print(attention_weights.shape,values.shape)
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
    

temp = torch.rand(2,8,1024)

batch_size = temp.shape[0]

in_channels = [8, 64, 128, 256]

conv_list = nn.ModuleList([nn.Conv1d(in_channel,out_channel, 8, 1) for in_channel,out_channel in zip(in_channels[:-1],in_channels[1:])])

for index, conv in enumerate(conv_list):
    temp = F.relu(conv(temp))
    if index == len(conv_list)-1:
          pass
    else:
        temp = F.max_pool1d(temp,4,4)

temp = temp.permute(0, 2, 1)

bidirectional = nn.LSTM(input_size=in_channels[-1], hidden_size=in_channels[-1], num_layers=1, batch_first=True, bidirectional=True)

temp, (h_n, c_n) = bidirectional(temp)
h_n = h_n.view(batch_size, temp.shape[-1])
#multihead_attn = nn.MultiheadAttention(embed_dim=in_channels[-1], num_heads=4, dropout=0, batch_first=True)
attn2 = BahdanauAttention(in_features=in_channels[-1]*2, hidden_units=10, out_features = len(all_tfs))

temp, attention_weights = attn2(h_n, temp)


test_layer = nn.Linear(in_features=8,out_features=1)
sigmoid = nn.Sigmoid()

test_temp = torch.rand(2,len(all_tfs),8)


