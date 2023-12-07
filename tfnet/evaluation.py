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

from pathlib import Path
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from tfnet.all_tfs import all_tfs
from logzero import logger

__all__ = ['CUTOFF', 'get_mean_auc', 'get_mean_pcc', 'get_mean_f1', 'get_mean_accuracy_score', 'get_mean_balanced_accuracy_score','get_label_ranking_average_precision_score', 'get_group_metrics', 'output_res']

CUTOFF = 0.8


# code
def get_mean_auc(targets, scores):
    auc_scores = []
    #return roc_auc_score(targets >= CUTOFF, scores)
    for i in range(targets.shape[1]):
        auc = roc_auc_score(targets[:, i], scores[:, i] > CUTOFF)
        auc_scores.append(auc)
    return np.mean(auc_scores)


def get_mean_pcc(targets, scores):
    pcc_list = []
    for i in range(targets.shape[0]):
        pcc = np.corrcoef(targets[i, :], scores[i, :])
        pcc_list.append(pcc)
    return np.mean(np.array(pcc_list, dtype=float))


def get_label_ranking_average_precision_score(targets, scores):
    return label_ranking_average_precision_score(targets, scores)


def get_mean_f1(targets, scores):
    return f1_score(targets, scores > CUTOFF, average='samples', zero_division=1.0)


def get_mean_accuracy_score(targets, scores):
    accuracy_score_list = []
    for i in range(targets.shape[0]):
        accuracy = accuracy_score(targets[i, :], scores[i, :]> CUTOFF)
        accuracy_score_list.append(accuracy)
    return np.mean(np.array(accuracy_score_list, dtype=float))


def get_mean_balanced_accuracy_score(targets, scores):
    accuracy_score_list = []
    for i in range(targets.shape[0]):
        accuracy = balanced_accuracy_score(targets[i, :], scores[i, :]> CUTOFF)
        accuracy_score_list.append(accuracy)
    return np.mean(np.array(accuracy_score_list, dtype=float))


def output_res(DNA_seqs, targets_lists, scores_lists, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    eval_out_path = output_path.with_suffix('.tsv')

    metrics = []
    metrics.append(get_f1(targets_lists, scores_lists))
    metrics.append(get_pcc(targets_lists, scores_lists))
    metrics.append(get_label_ranking_average_precision_score(targets_lists, scores_lists))

    #print("f1 score ",get_f1(targets_lists, scores_lists))
    #print("pcc ",get_pcc(targets_lists, scores_lists))
    #print("lrap ",get_label_ranking_average_precision_score(targets_lists, scores_lists))

    scores_lists = np.where(scores_lists > CUTOFF, 1, 0)
    scores_lists = np.split(scores_lists,scores_lists.shape[0], axis=0)
    scores_lists = [i.flatten().tolist() for i in scores_lists]

    with open(eval_out_path, 'w') as fp:
        writer = csv.writer(fp,delimiter="\t")
        writer.writerow(['DNA_seq', 'targets', 'predict'])
        for DNA_seq, targets_list,  scores_list in zip(DNA_seqs, targets_lists, scores_lists):
            writer.writerow([DNA_seq, all_tfs, targets_list, scores_list])
    logger.info(f'f1 score: {metrics[0]:3f} PCC: {metrics[1]:3f} lrap: {metrics[2]:3f}')
    logger.info(f'Complete')

