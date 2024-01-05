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
import networkx as nx
import matplotlib.pyplot as plt
import sys
import pandas as pd
from matplotlib.colors import ListedColormap
import matplotlib.gridspec
import pdb


# code
sns.set_theme()
sys.setrecursionlimit(3000)


# ---------------------- tfs dict ---------------------- #
all_tfs_dict = {}
for index, tf in enumerate(all_tfs):
    all_tfs_dict[tf] = index


# ---------------------- read model_bind list ---------------------- #
with gzip.open('/Users/cmf/Downloads/TFNet-multi-tf/data/tf_chip/lazy/data_train_mini.txt.gz', 'rt') as fp:
    #lines = fp.read().splitlines()
    #bind_list = [ i.split('\t')[-1] for i in lines 
    labels_array = []
    for line in fp:
        chr, start, stop, bind_list  = line.split('\t')
        bind_list = [float(i) for i in bind_list.split(',')]
        labels_array.append(bind_list)
labels_array = np.array(labels_array)


# ---------------------- distribution of each tf ---------------------- #
category_labels = [ i for i in all_tfs_dict]
distribution_df = pd.DataFrame({
    'tf': [ i for i in all_tfs_dict],
    'Count': labels_array.sum(axis=0)
})

distribution_bar = sns.barplot(distribution_df, y='tf', x='count', hue = "tf", orient="y")
distribution_bar.set_xlabel("")
distribution_bar.set_ylabel("")
distribution_bar.figure.savefig('results/distribution_matrix_model.pdf')


# ---------------------- correlation matrix ---------------------- #
def correlation_matrix(bind_array, category_labels):
    correlation_df = pd.DataFrame(bind_array, columns=category_labels)
    correlation_df = correlation_df[correlation_df.sum(axis=1) != 0]
    correlation_matrix = correlation_df.corr()
    sns.clustermap(correlation_matrix, center=0, cmap="vlag",
                   dendrogram_ratio=(0, .2),
                   #cbar_pos=(.02, .32, .03, .2),
                   cbar_pos=None,
                   linewidths=.75, figsize=(8, 8))


#fig, (ax1, ax2) = plt.subplots(1,2, figsize = (12, 6))
#fig, axes = plt.subplots(1,2, figsize = (12, 6))
correlation_matrix(motif_list, category_labels)
plt.title("Correlation Matrix of Motif bind")
plt.savefig('results/correlation_matrix_motif.pdf')
plt.close()

correlation_matrix(labels_array, category_labels)
plt.title("Correlation Matrix of Model bind")
plt.savefig('results/correlation_matrix_model.pdf')
plt.close()


# ---------------------- co-occurrence matrix for model_bind list ---------------------- #
def co_occurrence_matrix(bind_array, category_labels):
    co_occurrence_matrix = np.dot(bind_array.T,bind_array)
    row, col = np.diag_indices_from(co_occurrence_matrix)
    co_occurrence_matrix[row,col] = 0
    matrix_df = pd.DataFrame(co_occurrence_matrix, index=category_labels, columns=category_labels)
    sns.clustermap(matrix_df, center=0, cmap="vlag",
                    dendrogram_ratio=(0, .2),
                    cbar_pos=(.02, .32, .03, .2),
                    linewidths=.75, figsize=(12, 13))
    
co_occurrence_matrix(motif_list, category_labels)
plt.title("Correlation Matrix of Motif bind")
plt.savefig('results/co_occurrence_matrix_motif.pdf')
plt.close()

co_occurrence_matrix(labels_array, category_labels)
plt.title("Correlation Matrix of Model bind")
plt.savefig('results/co_occurrence_matrix_model.pdf')
plt.close()    


# ---------------------- network ---------------------- #
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



# ---------------------- motif and model list ---------------------- #
bind_list = np.array(labels_array)
motif_list = bind_list.copy()
np.random.shuffle(motif_list)

bind_list_add = np.add(bind_list, 3)
motif_list_add = np.add(motif_list, 1)
merge_list = np.multiply(bind_list_add, motif_list_add)

# for 0 in bind_list_add, 0 in motif_list_add resulting 3 indicating no bind for model and motif
# for 1 in bind_list_add, 0 in motif_list_add resulting 4 indicating bind for model and no bind for motif
# for 0 in bind_list_add, 1 in motif_list_add resulting 6 indicating no bind for model and bind for motif
# for 1 in bind_list_add, 1 in motif_list_add resulting 8 indicating bind for model and motif

category_labels = [ i for i in all_tfs_dict]
matrix_df = pd.DataFrame(merge_list, columns=category_labels)

matrix_df = matrix_df.mask(matrix_df == 3, 0)
matrix_df = matrix_df.mask(matrix_df == 4, 1)
matrix_df = matrix_df.mask(matrix_df == 6, 2)
matrix_df = matrix_df.mask(matrix_df == 8, 3)
matrix_df = matrix_df[matrix_df.sum(axis=1) != 0]

cmap_dict = {0: '#FFFFFF', 1: '#DC0000FF', 2: '#00468BFF', 3: '#FFFFFF'}
cmap = ListedColormap([cmap_dict[i] for i in range(len(cmap_dict))])
ax = sns.clustermap(data=matrix_df, method="weighted", metric="euclidean", cmap=cmap, vmin=-0.5, vmax=3.5, dendrogram_ratio=(0, 0.1), yticklabels=False, xticklabels=True)
plt.tight_layout()
ax.savefig('results/multiply_matrix.pdf')



# ---------------------- co-occurrence matrix for motif and model list ---------------------- #
merge_list
