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
import pysam
import pyBigWig
import pdb


__all__ = ["TFBindDataset"]


# code
class TFBindDataset(Dataset):
    def __init__(self, data_list, genome_fasta_file, bw_file, DNA_len=1024, DNA_pad=10, tf_len=39, padding_idx=0, target_len=200, DNA_N = True):
        #self.tf_names, self.DNA_x, self.tf_x, self.targets = [], [], [], []
        self.DNA_x, self.tf_x, self.targets = [], [], []
        self.data_list = data_list 
        self.DNA_N = DNA_N
        self.DNA_pad = DNA_pad
        self.DNA_len = DNA_len
        self.tf_len = tf_len
        self.genome_window_size = DNA_len
        self.genome_fasta = pysam.Fastafile(genome_fasta_file)
        self.bigwig_data = {}
        for index, single_bw_file in enumerate(bw_file):
            self.bigwig_data[index] = pyBigWig.open(single_bw_file)

    def __getitem__(self, item):
        for chr, start, stop, bind_list, all_tfs_seq in tqdm(self.data_list, leave=False):   

            #med = int((start + stop) / 2)
            #start = med - int(self.genome_window_size) / 2
            #stop = med + int(self.genome_window_size) / 2
            start = int(start)
            stop = int(stop)
            DNA_seq = self.genome_fasta.fetch(chr, start, stop)

            if self.DNA_N:
                d = {'a':0, 'A':0, 'g':1, 'G':1, 'c':2, 'C':2, 't':3, 'T':3, 'N':4, 'n':4}
                DNA_seq = self.DNA_pad*"N" + DNA_seq + self.DNA_pad*"N"     # for DNA pad to set conv1d output same dim
                mat = np.zeros((len(DNA_seq),5))
                for i in range(len(DNA_seq)):
                    mat[i,d[DNA_seq[i]]] = 1
                DNA_x = mat[:self.DNA_len + self.DNA_pad*2, :5]
            else: 
                d = {'a':0, 'A':0, 'g':1, 'G':1, 'c':2, 'C':2, 't':3, 'T':3}
                mat = np.zeros((len(DNA_seq),4))
                for i in range(len(DNA_seq)):
                    mat[i,d[DNA_seq[i]]] = 1
                DNA_x = mat[:self.DNA_len, :4]
            DNA_x = torch.tensor(DNA_x, dtype=torch.float32)
            # ---------------------- bw_list need padding like DNA_x ---------------------- #
            bigwig_signal = {}
            for index in range(len(self.bigwig_data)):
                bigwig_signal[index] = np.array(self.bigwig_data[index].values(chr,start,stop))
                bigwig_signal[index][np.isnan(bigwig_signal[index])] = 0

            for i in range(len(bigwig_signal)):
                if self.DNA_N:
                    bigwig_signal[i] = [0 for i in range(self.DNA_pad)] + [j for j in bigwig_signal[i]] + [0 for i in range(self.DNA_pad)]
                else:
                    bigwig_signal[i] = bigwig_signal[i]

                bigwig_signal[i] = np.expand_dims(bigwig_signal[i],axis=-1)
                #bigwig_signal_rc = bigwig_signal[i][::-1,:].copy()
                bigwig_signal[i] = torch.tensor(bigwig_signal[i], dtype=torch.float32)
                #bigwig_signal_rc = torch.tensor(bigwig_signal_rc, dtype=torch.float32)
                DNA_x = torch.cat([DNA_x, bigwig_signal[i]],dim=1)
                #DNA_x = torch.cat([DNA_x, bigwig_signal_rc],dim=1)
            
            tf_x = []
            for tf_seq in all_tfs_seq:
                tf_x.append([ACIDS.index(x if x in ACIDS else "-") for x in tf_seq])
                assert len(tf_seq) == self.tf_len
            #self.tf_x.append([ACIDS.index(x if x in ACIDS else "-") for x in tf_seq])
            

            #If bind_list has constant values (e.g., all zeros or all ones), the correlation coefficient becomes undefined, pcc resulting in nan and won't save model
            #if (1 in bind_list) and (0 in bind_list):
            self.tf_x.append(tf_x)
            self.targets.append(bind_list)
            self.DNA_x.append(DNA_x)


        self.tf_xx = torch.tensor(self.tf_x, dtype=torch.long)
        #self.tf_x = torch.tensor(self.tf_x, dtype=torch.float32)
        
        self.targets_x = np.asarray(self.targets, dtype=np.float32)


        return (self.DNA_x[item], self.tf_xx[item]), self.targets_x[item]
    def __len__(self):
        return len(self.data_list)
