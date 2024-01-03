#!/usr/bin/env python
# -*- encoding: utf-8 -*-

'''
@File : analysis_bind_result.py
@Time : 2024/01/03 16:25:36
@Author : Cmf
@Version : 1.0
@Desc : None
'''

# here put the import lib
from tfnet.all_tfs import all_tfs
import gzip
import numpy as np
import seaborn as sns


# code
sns.set_theme()


# ---------------------- tfs dict ---------------------- #
all_tfs_dict = {}
for index, tf in enumerate(all_tfs):
    all_tfs_dict[tf] = index


# ---------------------- co-occurrence matrix ---------------------- #
with gzip.open('/Users/cmf/Downloads/TFNet-multi-tf/data/tf_chip/lazy/data_train_mini.txt.gz', 'rt') as fp:
    #lines = fp.read().splitlines()
    #bind_list = [ i.split('\t')[-1] for i in lines 
    labels_array = []
    for line in fp:
        chr, start, stop, bind_list  = line.split('\t')
        bind_list = [float(i) for i in bind_list.split(',')]
        labels_array.append(bind_list)

labels_array = np.array(labels_array)

num_labels = labels_array.shape[1]
num_observations = labels_array.shape[0]
co_occurrence_matrix = np.zeros((num_labels, num_labels), dtype=int)

for i in range(num_observations):
    label_indices = np.where(labels_array[i] == 1)[0]
    for j in range(len(label_indices)):
        for k in range(j + 1, len(label_indices)):
            co_occurrence_matrix[label_indices[j], label_indices[k]] += 1
            co_occurrence_matrix[label_indices[k], label_indices[j]] += 1

g = sns.clustermap(co_occurrence_matrix, center=0, cmap="vlag",
                   dendrogram_ratio=(0, .2),
                   cbar_pos=(.02, .32, .03, .2),
                   linewidths=.75, figsize=(12, 13))
g.savefig('test.pdf')

co_occurrence_matrix_2 = np.dot(labels_array.T,labels_array)
row, col = np.diag_indices_from(co_occurrence_matrix_2)
co_occurrence_matrix_2[row,col] = 0

g = sns.clustermap(co_occurrence_matrix_2, center=0, cmap="vlag",
                   dendrogram_ratio=(0, .2),
                   cbar_pos=(.02, .32, .03, .2),
                   linewidths=.75, figsize=(12, 13))
g.savefig('test2.pdf')