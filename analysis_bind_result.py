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
co_occurrence_matrix = np.dot(labels_array.T,labels_array)
row, col = np.diag_indices_from(co_occurrence_matrix)
co_occurrence_matrix[row,col] = 0

g = sns.clustermap(co_occurrence_matrix, center=0, cmap="vlag",
                   dendrogram_ratio=(0, .2),
                   cbar_pos=(.02, .32, .03, .2),
                   linewidths=.75, figsize=(12, 13))
g.savefig('results/co_occurrence_matrix_model.pdf')


# ---------------------- network ---------------------- #
import networkx as nx
import matplotlib.pyplot as plt


num_labels = labels_array.shape[1]
num_observations = labels_array.shape[0]

# Create a graph
G = nx.Graph()

# Add nodes and edges based on co-occurrence matrix
for i in range(num_labels):
    for j in range(i + 1, num_labels):
        weight = co_occurrence_matrix[i, j]
        if weight > 0:
            G.add_edge(i, j, weight=weight)

# Visualize the graph
plt.clear()
#pos = nx.spring_layout(G)  # You can use different layout algorithms
pos = nx.nx_pydot.pydot_layout(G)
nx.draw(G, pos, with_labels=True, font_weight='bold', node_size=500, node_color='skyblue', font_color='black', edge_color='gray', width=2, alpha=0.7)
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

plt.show()
