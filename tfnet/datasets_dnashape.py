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
from tfnet.generate_dnashape import seq_to_shape_dict
import re
import pysam
import pyBigWig
import pandas as pd
import pdb


__all__ = ["TFBindDataset"]


# code
class TFBindDataset(Dataset):
    def __init__(self, data_list, genome_fasta_file, bw_file, DNA_len=1024, DNA_pad=10, tf_len=39, padding_idx=0, target_len=200):
        self.data_list = data_list
        self.DNA_x, self.tf_x = [], []
        self.genome_fasta = pysam.Fastafile(genome_fasta_file)
        self.bigwig_data = {}
        for index, single_bw_file in enumerate(bw_file):
            self.bigwig_data[index] = pyBigWig.open(single_bw_file)

        self.DNA_pad = DNA_pad
        self.DNA_len = DNA_len
        self.tf_len = tf_len
        self.bind_list = [ i[-1] for i in data_list]
        self.bind_list = np.asarray(self.bind_list, dtype=np.float32)

        #self.dnashape = pd.read_csv("./tfnet/dnashape5.2.csv", header=0, index_col=0) # for dnashape5
        self.dnashape = pd.read_csv("./tfnet/dnashape14.2.csv", header=0, index_col=0) # for dnashape14
        #if normalize:
        #    self.dnashape = self.dnashape.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
        self.dnashape_dict =self.dnashape.apply(lambda x: x.dropna().tolist(), axis=1).to_dict()   


    def __getitem__(self, idx):
        chr, start, stop, bind_list = self.data_list[idx]

        bind_list = np.asarray(bind_list, dtype=np.float32)
        start = int(start)
        stop = int(stop)
        # ---------------------- shift ---------------------- #
        '''
        # disable because may cause N in seq result in error in dnashape
        shift = np.random.randint(-20, 20+1)
        start += shift
        stop += shift
        '''

        DNA_seq = self.genome_fasta.fetch(chr, start, stop)
        d = {'a':0, 'A':0, 'g':1, 'G':1, 'c':2, 'C':2, 't':3, 'T':3}
        mat = np.zeros((len(DNA_seq),4))
        for i in range(len(DNA_seq)):
            if len(re.findall('[atcg]', DNA_seq[i].lower())) != 0:  # no one hot for n
                mat[i,d[DNA_seq[i]]] = 1
        DNA_x = mat[:self.DNA_len, :4]
        DNA_x = torch.tensor(DNA_x, dtype=torch.float32)
        
        # ---------------------- DNA shape info ---------------------- #
        DNA_shape_unpad = torch.tensor(seq_to_shape_dict(DNA_seq, self.dnashape_dict), dtype=torch.float32) # much faster than seq_to_shape5
        #zero_padding = torch.zeros(2, 5) # for dnashape5
        zero_padding = torch.zeros(2, 14) # for dnashape14
        DNA_shape = torch.cat([zero_padding, DNA_shape_unpad, zero_padding], dim=0)
        DNA_x = torch.cat([DNA_x, DNA_shape],dim=1)
        
        # ---------------------- bw_list need padding like DNA_x ---------------------- #
        bigwig_signals = []
        bigwig_signals_rc = []
        
        for index in range(len(self.bigwig_data)):
            bigwig_signal = np.array(self.bigwig_data[index].values(chr,start,stop))
            bigwig_signal[np.isnan(bigwig_signal)] = 0

            # ---------------------- place mappability first, chromatin second ---------------------- #
            bigwig_signals.append(bigwig_signal)
            bigwig_signals_rc.append(bigwig_signal[::-1].copy())

        # ---------------- concatenate rc, comment to abort----------------#
        bigwig_signals.extend(bigwig_signals_rc)
        for i in range(len(bigwig_signals)):
            bigwig_signal = bigwig_signals[i]

            bigwig_signal = np.expand_dims(bigwig_signal, axis=-1)
            bigwig_signal = torch.tensor(bigwig_signal, dtype=torch.float32)

            DNA_x = torch.cat([DNA_x, bigwig_signal],dim=1)

        return DNA_x, bind_list
    
    def __len__(self):
        return len(self.data_list)
