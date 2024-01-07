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
from pathlib import Path
import csv
import pdb


# code
sns.set_theme()
sys.setrecursionlimit(3000)
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)

# ---------------------- tfs dict ---------------------- #
all_tfs_dict = {}
for index, tf in enumerate(all_tfs):
    all_tfs_dict[tf] = index
category_labels = [ i for i in all_tfs_dict]

# ---------------------- read model_binds and motif_binds list ---------------------- #
with gzip.open('/Users/cmf/Downloads/tfnet/data/tf_chip/lazy/data_train_mini.txt.gz', 'rt') as fp:
    #lines = fp.read().splitlines()
    #model_binds = [ i.split('\t')[-1] for i in lines 
    model_binds = []
    for line in fp:
        chr, start, stop, bind_list  = line.split('\t')
        bind_list = [float(i) for i in bind_list.split(',')]
        model_binds.append(bind_list)


model_binds = np.array(model_binds)
# ---------------- pseduo motif_binds ----------------#
motif_binds = np.random.randint(0,2,(model_binds.shape[0], model_binds.shape[1]))



# ---------------------- distribution of each tf for motif_bind and model_bind ---------------------- #
def distribution_plot(bind_array, category_labels):
    distribution_df = pd.DataFrame({
        'tf': category_labels,
        'Count': bind_array.sum(axis=0)
    })
    distribution_bar = sns.barplot(distribution_df, y='tf', x='Count', hue = "tf", orient="y", width=0.8, gap=0)
    distribution_bar.set_xlabel("")
    distribution_bar.set_ylabel("")


#fig, axes = plt.subplots(1,2, figsize = (10, 4))
f = plt.figure() 
f.set_figwidth(10) 
f.set_figheight(4)
plt.subplot(1, 2, 1)
distribution_plot(motif_binds, category_labels)
plt.title("tf distribution of Motif bind")
plt.yticks(fontsize=10)

plt.subplot(1, 2, 2)
distribution_plot(model_binds, category_labels)
plt.title("tf distribution of Model bind")
plt.yticks(fontsize=10)
plt.savefig('results/distribution_matrix.pdf')


# ---------------------- correlation matrix for motif_bind and model_bind ---------------------- #
def correlation_matrix(bind_array, category_labels):
    correlation_df = pd.DataFrame(bind_array, columns=category_labels)
    correlation_df = correlation_df[correlation_df.sum(axis=1) != 0]
    correlation_matrix = correlation_df.corr()
    sns.clustermap(correlation_matrix, center=0, cmap="vlag",
                   dendrogram_ratio=(0, .2),
                   cbar_pos=(.02, .32, .03, .2),
                   linewidths=.75, figsize=(8, 8))


#fig, (ax1, ax2) = plt.subplots(1,2, figsize = (12, 6))
#fig, axes = plt.subplots(1,2, figsize = (12, 6))
correlation_matrix(motif_binds, category_labels)
plt.title("Correlation Matrix of Motif bind")
plt.savefig('results/correlation_matrix_motif.pdf')
plt.close()

correlation_matrix(model_binds, category_labels)
plt.title("Correlation Matrix of Model bind")
plt.savefig('results/correlation_matrix_model.pdf')
plt.close()


# ---------------- Pairwise Scatter for motif_bind and model_bind ----------------#
data = {
    'Label1': np.random.randint(0, 2, 100),
    'Label2': np.random.randint(0, 2, 100),
    'Label3': np.random.randint(0, 2, 100),
    'Label4': np.random.randint(0, 2, 100),
}

df = pd.DataFrame(data)

# Add a continuous variable for demonstration
df['ContinuousVar'] = np.random.randn(100)

# Create a pairplot
sns.pairplot(df, hue="Label1", markers=["o", "s"], diag_kind="kde")
plt.suptitle("Pairwise Scatter Plots with KDE Diagonals")

# Show the plot
plt.show()


# ---------------------- co-occurrence matrix for motif_bind and model_bind ---------------------- #
def co_occurrence_matrix(bind_array, category_labels):
    co_occurrence_matrix = np.dot(bind_array.T,bind_array)
    row, col = np.diag_indices_from(co_occurrence_matrix)
    co_occurrence_matrix[row,col] = 0
    matrix_df = pd.DataFrame(co_occurrence_matrix, index=category_labels, columns=category_labels)
    sns.clustermap(matrix_df, center=0, cmap="vlag",
                    dendrogram_ratio=(0, .2),
                    cbar_pos=(.02, .32, .03, .2),
                    linewidths=.75, figsize=(12, 13))
    
co_occurrence_matrix(motif_binds, category_labels)
plt.title("Correlation Matrix of Motif bind")
plt.savefig('results/co_occurrence_matrix_motif.pdf')
plt.close()

co_occurrence_matrix(model_binds, category_labels)
plt.title("Correlation Matrix of Model bind")
plt.savefig('results/co_occurrence_matrix_model.pdf')
plt.close()    


# ---------------------- network ---------------------- #
num_labels = model_binds.shape[1]
num_observations = model_binds.shape[0]

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



# ---------------------- difference of motif and model bind ---------------------- #
model_binds_add = np.add(model_binds, 3)
motif_binds_add = np.add(motif_binds, 1)
merge_lists = np.multiply(model_binds_add, motif_binds_add)
# for 0 in model_binds_add, 0 in motif_binds_add resulting 3 indicating no bind for model and motif
# for 1 in model_binds_add, 0 in motif_binds_add resulting 4 indicating bind for model and no bind for motif
# for 0 in model_binds_add, 1 in motif_binds_add resulting 6 indicating no bind for model and bind for motif
# for 1 in model_binds_add, 1 in motif_binds_add resulting 8 indicating bind for model and motif
merge_binds = pd.DataFrame(merge_lists, columns=category_labels)

merge_binds = merge_binds.mask(merge_binds == 3, 0)
merge_binds = merge_binds.mask(merge_binds == 4, 1)
merge_binds = merge_binds.mask(merge_binds == 6, 2)
merge_binds = merge_binds.mask(merge_binds == 8, 3)
merge_binds = merge_binds[merge_binds.sum(axis=1) != 0]

#cmap_dict = {0: '#FFFFFF', 1: '#DC0000FF', 2: '#00468BFF', 3: '#FFFFFF'}
cmap_dict = {0: '#FFFFFF', 1: '#DC0000FF', 2: '#00468BFF', 3: 'black'}
# inspect the relation between 1 2 and 3
# convert 1,2,3 to 1 and plot cluster map, and color based on original value
cmap = ListedColormap([cmap_dict[i] for i in range(len(cmap_dict))])
ax = sns.clustermap(data=merge_bind, method="weighted", metric="euclidean", cmap=cmap, vmin=-0.5, vmax=3.5, dendrogram_ratio=(0, 0.1), yticklabels=False, xticklabels=True)
plt.tight_layout()
ax.savefig('results/multiply_matrix.pdf')

# ---------------- Set values greater than 1 to 1 for clustering and color for original value ----------------#
row_col_cluster = sns.clustermap(merge_bind.applymap(lambda x: 1 if x > 1 else x), method='average', row_cluster=True, col_cluster=True)
plt.close()
merge_bind_reordered = merge_bind.iloc[row_col_cluster.dendrogram_row.reordered_ind, row_col_cluster.dendrogram_col.reordered_ind]
sns.heatmap(merge_bind_reordered, cmap=cmap)
plt.savefig('results/multiply_matrix_2.pdf')



# ---------------------- co-occurrence matrix for merge_bind ---------------------- #
merge_binds



# ---------------- network for merge_bind ----------------#
# iterate each tf each value pairwise
def pairwise_net(merge_bind, category_labels):
    if Path("results/network_merge_bind.bed").exists():
        print(f"error: result file exists")
    else:
        with open("results/network_merge_bind.bed", "w") as fp:
            writer = csv.writer(fp, delimiter="\t")
            for i in range(len(merge_bind)):
                for j in range(len(merge_bind[i])):
                    tf_name_1 = category_labels[j]
                    if merge_bind[i][j] == 1:
                        tf_name_1 = "Model_" + tf_name_1
                    elif merge_bind[i][j] == 2:
                        tf_name_1 = "Motif_" + tf_name_1
                    elif merge_bind[i][j] == 3:
                        tf_name_1 = "Both_" + tf_name_1
                    else:
                        continue # dut to no bind
                    for k in range(len(category_labels)):
                        if k == j:
                            continue # due to encounter self
                        else:
                            tf_name_2 = category_labels[k]
                        if merge_bind[i][k] == 1:
                            tf_name_2 = "Model_" + tf_name_2
                        elif merge_bind[i][k] == 2:
                            tf_name_2 = "Model_" + tf_name_2
                        elif merge_bind[i][k] == 3:
                            tf_name_2 = "Both_" + tf_name_2
                        else:
                            continue # dut to no bind
                        writer.writerow([tf_name_1, tf_name_2])

pairwise_net(np.array(merge_binds), category_labels)


pairwise_net_df = pd.read_csv("results/network_merge_bind.bed", header=None, sep='\t', names=['target','object'])
pairwise_net_count_df = pairwise_net_df.value_counts(subset=['target', 'object']).to_frame().reset_index()

G = nx.Graph()
for index, row in pairwise_net_count_df.iterrows():
        G.add_edge(row['target'], row['object'], weight=row['count'])

pos = nx.nx_pydot.pydot_layout(G)
nx.draw(G, pos, with_labels=True, font_weight='bold', node_size=300, node_color='skyblue', font_color='black', edge_color='gray', width=2, alpha=0.7)
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

plt.show()
