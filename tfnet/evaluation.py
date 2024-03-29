#!/usr/bin/env python
# -*- encoding: utf-8 -*-

'''
@File : evaluation.py
@Time : 2023/11/09 11:20:10
@Author : Cmf
@Version : 1.0
@Desc : None
'''

# here put the import lib
import csv
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_recall_curve, average_precision_score, auc
from logzero import logger
import pdb
import warnings
warnings.filterwarnings('error')  

__all__ = ['CUTOFF', 'get_mean_auc', 'get_auc', 'get_recall', 'get_f1', 'get_precision', 'get_mean_precision', 'get_mean_recall', 'get_mean_aupr', 'get_mean_f1', 'get_mean_accuracy_score', 'get_balanced_accuracy_score', 'get_mean_balanced_accuracy_score','get_label_ranking_average_precision_score', 'get_group_metrics', 'output_eval', 'output_predict']

CUTOFF = 0.8


# code
def get_mean_auc(targets, scores):
    auc_scores = []
    #return roc_auc_score(targets >= CUTOFF, scores)
    for i in range(targets.shape[1]):
        auc = roc_auc_score(targets[:, i], scores[:, i] )
        auc_scores.append(auc)
    return np.mean(auc_scores)

def get_auc(targets, scores):
    auc_scores = []
    #return roc_auc_score(targets >= CUTOFF, scores)
    for i in range(targets.shape[1]):
        auc = roc_auc_score(targets[:, i], scores[:, i] )
        auc_scores.append(auc)
    return auc_scores


def get_recall(targets, scores, cutoff = CUTOFF):
    recall_list = []
    for i in range(targets.shape[1]):
        recall = recall_score(targets[:, i], scores[:, i]> cutoff, zero_division=1.0)
        recall_list.append(recall)
    return recall_list


def get_mean_recall(targets, scores, cutoff = CUTOFF):
    recall_list = get_recall(targets, scores, cutoff)
    return np.mean(recall_list)    


def get_precision(targets, scores, cutoff = CUTOFF):
    precision_list = []
    for i in range(targets.shape[1]):
        precision = precision_score(targets[:, i], scores[:, i]> cutoff, zero_division=1.0)
        precision_list.append(precision)
    return precision_list


def get_mean_precision(targets, scores, cutoff = CUTOFF):
    precision_list = get_precision(targets, scores, cutoff = cutoff)
    return np.mean(precision_list)


def get_mean_aupr(targets, scores):
    aupr_list = []
    for i in range(targets.shape[1]):
        precision, recall, thresholds = precision_recall_curve(targets[:, i], scores[:, i])
        #average_precision = average_precision_score(targets[i, :], scores[i, :])
        auc_precision_recall = auc(recall, precision)
        aupr_list.append(auc_precision_recall)
    return np.mean(np.array(aupr_list, dtype=float))  


def get_aupr(targets, scores):
    aupr_list = []
    for i in range(targets.shape[1]):
        precision, recall, thresholds = precision_recall_curve(targets[:, i], scores[:, i])
        #average_precision = average_precision_score(targets[i, :], scores[i, :])
        auc_precision_recall = auc(recall, precision)
        aupr_list.append(auc_precision_recall)
    return aupr_list


def get_label_ranking_average_precision_score(targets, scores):
    return label_ranking_average_precision_score(targets, scores)


def get_mean_f1(targets, scores, cutoff=CUTOFF):
    return f1_score(targets, scores > cutoff, average='macro', zero_division=1.0)


def get_f1(targets, scores, cutoff=CUTOFF):
    f1_list = []
    for i in range(targets.shape[1]):
        f1_list.append(f1_score(targets[:, i], scores[:, i] > cutoff, zero_division=1.0))
    return f1_list


def get_mean_accuracy_score(targets, scores, axis = 0, cutoff=CUTOFF):
    accuracy_score_list = []
    if axis == 0:
        for i in range(targets.shape[0]):
            accuracy = accuracy_score(targets[i, :], scores[i, :]> cutoff)
            accuracy_score_list.append(accuracy)
    elif axis == 1:
        for i in range(targets.shape[1]):
            accuracy = accuracy_score(targets[:, i], scores[:, i]> cutoff)
            accuracy_score_list.append(accuracy)
    return np.mean(np.array(accuracy_score_list, dtype=float))


def get_balanced_accuracy_score(targets, scores, axis = 0, cutoff=CUTOFF):
    accuracy_score_list = []
    if axis == 0:
        for i in range(targets.shape[0]):
            accuracy = balanced_accuracy_score(targets[i, :], scores[i, :]> cutoff)
            accuracy_score_list.append(accuracy)
    elif axis == 1:
        for i in range(targets.shape[1]):
            accuracy = balanced_accuracy_score(targets[:, i], scores[:, i]> cutoff)
            accuracy_score_list.append(accuracy)        
    return accuracy_score_list

#try :
#    balanced_accuracy_score(targets[i, :], scores[i, :]> CUTOFF)   # due to all 0 in y_true
#except Warning as e:
#        pdb.set_trace()


def get_mean_balanced_accuracy_score(targets, scores, axis = 0, cutoff=CUTOFF):
    accuracy_score_list = get_balanced_accuracy_score(targets, scores, axis = axis, cutoff=cutoff)
    return np.mean(accuracy_score_list)


def output_eval(chrs, starts, stops, targets_lists, scores_lists, all_tfs, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    eval_out_path = output_path.with_suffix('.eval.tsv')

    metrics = []
    metrics.append(get_mean_auc(targets_lists, scores_lists))
    metrics.append(get_mean_aupr(targets_lists, scores_lists))
    metrics.append(get_mean_recall(targets_lists, scores_lists))
    metrics.append(get_mean_f1(targets_lists, scores_lists))
    metrics.append(get_label_ranking_average_precision_score(targets_lists, scores_lists))
    metrics.append(get_mean_accuracy_score(targets_lists, scores_lists))
    metrics.append(get_mean_balanced_accuracy_score(targets_lists, scores_lists))


    # ---------------------- plot ---------------------- #
    plot_data = pd.DataFrame({
        "TF_name" : all_tfs,
        "AUC" : get_auc(targets_lists, scores_lists),
        "AUPR" : get_aupr(targets_lists, scores_lists),
        "RECALL" : get_recall(targets_lists, scores_lists),
        "F1" : get_f1(targets_lists, scores_lists)
        }
    )

    plot_data.to_csv(output_path.with_suffix('.eval.repl.tsv'), sep='\t')

    #sns.set(rc={'figure.figsize':(7,4)})
    rel_plot = sns.scatterplot(data=plot_data, x="AUC", y="AUPR", hue="RECALL", size="F1", sizes=(50,200))
    sns.move_legend(rel_plot, "upper left", bbox_to_anchor=(1, 0.75))
    fig = rel_plot.get_figure()
    fig.savefig(output_path.with_suffix('.eval.repl.pdf')) 


    fig, axes = plt.subplots(2, 2)
    xlabel = all_tfs
    sns.barplot(data=plot_data, x='TF_name', y='AUC', ax=axes[0,0])
    axes[0,0].tick_params(axis='x', labelrotation=45)
    axes[0,0].set_xticklabels(xlabel, fontsize=4)
    axes[0,0].set(xlabel='')

    sns.barplot(data=plot_data, x='TF_name', y='AUPR', ax=axes[0,1])
    axes[0,1].tick_params(axis='x', labelrotation=45)
    axes[0,1].set_xticklabels(xlabel, fontsize=4)
    axes[0,1].set(xlabel='')

    sns.barplot(data=plot_data, x='TF_name', y='RECALL', ax=axes[1,0])
    axes[1,0].tick_params(axis='x', labelrotation=45)
    axes[1,0].set_xticklabels(xlabel, fontsize=4)
    axes[1,0].set(xlabel='')

    sns.barplot(data=plot_data, x='TF_name', y='F1', ax=axes[1,1])
    axes[1,1].tick_params(axis='x', labelrotation=45)
    axes[1,1].set_xticklabels(xlabel, fontsize=4)
    axes[1,1].set(xlabel='')

    fig.savefig(output_path.with_suffix('.eval.box.pdf')) 
    # ---------------------- section ---------------------- #
    ori_scores_lists = scores_lists
    ori_scores_lists = np.split(ori_scores_lists,ori_scores_lists.shape[0], axis=0)
    ori_scores_lists = [i.flatten().tolist() for i in ori_scores_lists]

    scores_lists = np.where(scores_lists > CUTOFF, 1, 0)
    scores_lists = np.split(scores_lists,scores_lists.shape[0], axis=0)
    scores_lists = [i.flatten().tolist() for i in scores_lists]

    targets_lists = [ list(map(int,i.tolist())) for i in targets_lists ]

    with open(eval_out_path, 'w') as fp:
        writer = csv.writer(fp, delimiter="\t")
        #writer.writerow(['chr', 'start', 'stop', 'targets', 'predict'])
        for chr, start, stop, targets_list, ori_scores_list, scores_list in zip(chrs, starts, stops, targets_lists, ori_scores_lists, scores_lists):
            writer.writerow([chr, start, stop, targets_list, ori_scores_list, scores_list])
    logger.info(
            f'mean_auc: {metrics[0]:.5f}  '
            f'aupr: {metrics[1]:.5f}  '
            f'recall score: {metrics[2]:.5f}  '
            f'f1 score: {metrics[3]:.5f}  '
            f'lrap: {metrics[4]:.5f}  '
            f'accuracy: {metrics[5]:.5f}  '
            f'balanced accuracy: {metrics[6]:.5f}'
            )
    logger.info(f'Eval Completed')


def output_predict(chrs, starts, stops, scores_lists, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predict_out_path = output_path.with_suffix('.predict.tsv')

    scores_lists = np.where(scores_lists > CUTOFF, 1, 0)
    scores_lists = np.split(scores_lists,scores_lists.shape[0], axis=0)
    scores_lists = [i.flatten().tolist() for i in scores_lists]

    with open(predict_out_path, 'w') as fp:
        writer = csv.writer(fp, delimiter="\t")
        writer.writerow(['chr', 'start', 'stop', 'predict'])
        for chr, start, stop, scores_list in zip(chrs, starts, stops, scores_lists):
            writer.writerow([chr, start, stop, scores_list])
    logger.info(f'Predicting Completed')