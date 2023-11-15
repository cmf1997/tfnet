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
from tfnet.all_tfs import all_tfs
from logzero import logger

__all__ = ['CUTOFF', 'get_auc', 'get_pcc', 'get_f1', 'get_label_ranking_average_precision_score', 'get_srcc', 'get_group_metrics', 'output_res']

CUTOFF = 0.5


# code
def get_auc(targets, scores):
    #return roc_auc_score(targets >= CUTOFF, scores)
    return roc_auc_score(targets, scores)

# If one of the variables has constant values (e.g., all zeros or all ones), the correlation coefficient becomes undefined, resulting in nan.
def get_pcc(targets, scores):
    return np.corrcoef(targets, scores)[0, 1]


def get_label_ranking_average_precision_score(targets, scores):
    return label_ranking_average_precision_score(targets, scores)


def get_f1(targets, scores):
    return f1_score(targets, scores > CUTOFF, average = "samples")


def get_srcc(targets, scores):
    return spearmanr(targets, scores)[0]


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

