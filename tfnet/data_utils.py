#!/usr/bin/env python
# -*- encoding: utf-8 -*-

'''
@File : data_utils.py
@Time : 2023/11/09 11:19:13
@Author : Cmf
@Version : 1.0
@Desc : None
'''

# here put the import lib
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tfnet.all_tfs import all_tfs
import re
import ast
import gzip
import pysam
import pdb

__all__ = ['ACIDS', 'get_tf_name_seq', 'get_data', 'get_data_lazy', 'get_binding_data', 'calculate_class_weights_dict','get_seq2logo_data', 'set_DNA_len','get_model_parameters']

ACIDS = '0-ACDEFGHIKLMNPQRSTVWY'

set_DNA_len = 1024

# code
def get_tf_name_seq(tf_name_seq_file):
    tf_name_seq = {}
    with open(tf_name_seq_file) as fp:
        for line in fp:
            tf_name, tf_seq = line.split()
            tf_name_seq[tf_name] = tf_seq
    return tf_name_seq


def get_data(data_file, tf_name_seq, DNA_N = True):
    data_list = []
    all_tfs_seq = []
    for tf_name in all_tfs:
         all_tfs_seq.append(tf_name_seq[tf_name])
    with gzip.open(data_file, 'rt') as fp:
        for line in fp:
            bw_list = []
            #DNA_seq, bind_list,  = line.split()
            # ---------------------- process multiple bigwig file ---------------------- #
            DNA_seq, bw_signal, bind_list  = line.split('\t')

            bind_list = [float(i) for i in bind_list.split(',')]

            # ---------------------- encounter w r in dna seq ---------------------- #
            if DNA_N:
                if len(DNA_seq) == set_DNA_len and len(DNA_seq) == len(re.findall('[atcgn]', DNA_seq.lower())):
                    #data_list.append((DNA_seq, bind_list, all_tfs_seq))
                    bw_signal = ast.literal_eval(bw_signal)
                    bw_signal = np.array(bw_signal)
                    for i in range(bw_signal.shape[0]):
                        bw_list.append([float(i) for i in bw_signal[i].split(",")])
                    data_list.append((DNA_seq, bw_list, bind_list, all_tfs_seq))

            else:
                if len(DNA_seq) == set_DNA_len and len(DNA_seq) == len(re.findall('[atcg]', DNA_seq.lower())):
                    #data_list.append((DNA_seq, bind_list, all_tfs_seq))                        
                    bw_signal = ast.literal_eval(bw_signal)
                    bw_signal = np.array(bw_signal)
                    for i in range(bw_signal.shape[0]):
                        bw_list.append([float(i) for i in bw_signal[i].split(",")])      
                    data_list.append((DNA_seq, bw_list, bind_list, all_tfs_seq))   
    return data_list

def get_data_lazy(data_file, tf_name_seq, genome_fasta_file, DNA_N = True):
    data_list = []
    all_tfs_seq = []
    for tf_name in all_tfs:
         all_tfs_seq.append(tf_name_seq[tf_name])

    genome_fasta = pysam.Fastafile(genome_fasta_file)
    with gzip.open(data_file, 'rt') as fp:
        for line in fp:
            #DNA_seq, bind_list,  = line.split()
            # ---------------------- process multiple bigwig file ---------------------- #
            chr, start, stop, bind_list  = line.split('\t')
            start = int(start)
            stop = int(stop)

            DNA_seq = genome_fasta.fetch(chr, start, stop)

            bind_list = [float(i) for i in bind_list.split(',')]

            # ---------------------- encounter w r in dna seq ---------------------- #
            if DNA_N:
                if len(DNA_seq) == set_DNA_len and len(DNA_seq) == len(re.findall('[atcgn]', DNA_seq.lower())):
                    data_list.append((chr, start, stop, bind_list, all_tfs_seq))

            else:
                if len(DNA_seq) == set_DNA_len and len(DNA_seq) == len(re.findall('[atcg]', DNA_seq.lower())):   
                    data_list.append((chr, start, stop, bind_list, all_tfs_seq))
    return data_list

def calculate_class_weights_dict(data_file):
    y_train = np.loadtxt(data_file,dtype=str)
    true_label = [ y_train[i][-1] for i in range(y_train.shape[0])]
    bind_list = []
    for i in range(len(true_label)):
        bind_list.append([float(j) for j in true_label[i].split(',')])
    bind_list = np.array(bind_list)
    num_labels = bind_list.shape[1]
    class_weights_dict = {}

    # Calculate class weights for each binary label independently
    for label in range(num_labels):
        classes = np.unique(bind_list[:, label])
        class_weights = compute_class_weight(class_weight='balanced', classes = classes, y=bind_list[:, label])
        class_weights_dict[label] = {cls: weight for cls, weight in zip(classes, class_weights)}
    
    return class_weights_dict


def get_binding_data(data_file, tf_name_seq, peptide_pad=3, core_len=9):
    data_list = []
    with open(data_file) as fp:
        for line in fp:
            pdb, mhc_name, mhc_seq, peptide_seq, core = line.split()
            assert len(core) == core_len
            data_list.append(((pdb, mhc_name, core), peptide_seq, tf_name_seq[mhc_name], 0.0))
    return data_list


def get_seq2logo_data(data_file, mhc_name, mhc_seq):
    with open(data_file) as fp:
        return [(mhc_name, line.strip(), mhc_seq, 0.0) for line in fp]
    

def get_model_parameters(model):
    params = list(model.parameters())
    k = 0
    for i in params:
        l = 1
        for j in i.size():
            l*= j
        k += l
    print("total:" + str(k))
