#!/usr/bin/env python
# -*- encoding: utf-8 -*-

'''
@File : datasets.py
@Time : 2023/11/09 11:19:37
@Author : Cmf
@Version : 1.0
@Desc : None
'''

# here put the import lib
import numpy as np
import torch

from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from tfnet.data_utils import ACIDS
from tfnet.all_tfs import all_tfs
import re
import pdb


__all__ = ["TFBindDataset"]


# code
class TFBindDataset(Dataset):
    def __init__(self, data_list, DNA_len=1024, DNA_pad=10, tf_len=39, DNA_N = False, padding_idx=0):
        pdb.set_trace()
        #self.tf_names, self.DNA_x, self.tf_x, self.targets = [], [], [], []
        self.DNA_x, self.tf_x, self.targets = [], [], []
        #for tf_name, DNA_seq, tf_seq, score in tqdm(data_list, leave=False):
        #for DNA_seq, bind_list, all_tfs_seq in tqdm(data_list, leave=False):    
        for DNA_seq, atac_signal, bind_list, all_tfs_seq in tqdm(data_list, leave=False):   
            #self.tf_names.append(tf_name)
            # one-hot encode for DNA input
            if DNA_N:
                d = {'a':0, 'A':0, 'g':1, 'G':1, 'c':2, 'C':2, 't':3, 'T':3, 'N':4, 'n':4}
                DNA_seq = DNA_pad*"N" + DNA_seq + DNA_pad*"N"     # for DNA pad to set conv1d output same dim
                mat = np.zeros((len(DNA_seq),5))
                #mat = np.zeros((len(DNA_seq),4))
                for i in range(len(DNA_seq)):
                    mat[i,d[DNA_seq[i]]] = 1
                DNA_x = mat[:DNA_len + DNA_pad*2, :5]
            else: 
                if len(DNA_seq) == len(re.findall('[atcg]', DNA_seq.lower())):
                    d = {'a':0, 'A':0, 'g':1, 'G':1, 'c':2, 'C':2, 't':3, 'T':3}
                    mat = np.zeros((len(DNA_seq),4))
                    for i in range(len(DNA_seq)):
                        mat[i,d[DNA_seq[i]]] = 1
                        DNA_x = mat[:DNA_len, :4]

            DNA_x = torch.tensor(DNA_x, dtype=torch.float32)
            
            # ---------------------- atac_signal need padding like DNA_x ---------------------- #
            if DNA_N:
                atac_signal = [0 for i in range(DNA_pad)] + atac_signal + [0 for i in range(DNA_pad)]
            atac_signal = np.expand_dims(atac_signal,axis=-1)
            atac_signal = torch.tensor(atac_signal, dtype=torch.float32)
            DNA_x = torch.cat([DNA_x, atac_signal],dim=1)
            
            #self.DNA_x.append(DNA_x)
            #assert self.DNA_x[-1].shape[1] == DNA_len + DNA_pad * 2
            tf_x = []
            for tf_seq in all_tfs_seq:
                tf_x.append([ACIDS.index(x if x in ACIDS else "-") for x in tf_seq])
                assert len(tf_seq) == tf_len
            #self.tf_x.append([ACIDS.index(x if x in ACIDS else "-") for x in tf_seq])


            #If bind_list has constant values (e.g., all zeros or all ones), the correlation coefficient becomes undefined, pcc resulting in nan and won't save model
            if (1 in bind_list) and (0 in bind_list):
                self.tf_x.append(tf_x)
                self.targets.append(bind_list)
                self.DNA_x.append(DNA_x)
                #pdb.set_trace()

                assert self.DNA_x[-1].shape[0] == DNA_len + DNA_pad * 2

        #self.DNA_x, self.tf_x = np.asarray(self.DNA_x), np.asarray(self.tf_x)
        #self.DNA_x = np.asarray(self.DNA_x, dtype=np.float32)

        self.tf_x = torch.tensor(self.tf_x, dtype=torch.long)
        #self.tf_x = torch.tensor(self.tf_x, dtype=torch.float32)
        
        self.targets = np.asarray(self.targets, dtype=np.float32)
    def __getitem__(self, item):
        return (self.DNA_x[item], self.tf_x[item]), self.targets[item]
    def __len__(self):
        return len(self.DNA_x)
