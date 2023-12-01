#!/usr/bin/env python
# -*- encoding: utf-8 -*-

'''
@File : baseline_zero.py
@Time : 2023/11/30 09:58:39
@Author : Cmf
@Version : 1.0
@Desc : due to the sparse of tf binding list, make a baseline model with all predict to 0
'''

# here put the import lib
from ruamel.yaml import YAML
import numpy as np
from pathlib import Path
from tfnet.data_utils import set_DNA_len
from tfnet.evaluation import get_mean_pcc, get_mean_f1, get_label_ranking_average_precision_score, get_mean_accuracy_score, get_mean_auc, get_mean_balanced_accuracy_score
from logzero import logger
import sys

np.seterr(divide='ignore', invalid='ignore')

def get_data_for_baseline(data_file):
    data_list = []
    all_tfs_seq = []
    with open(data_file) as fp:
        for line in fp:
            DNA_seq, atac_signal, bind_list  = line.split()
            bind_list = [float(i) for i in bind_list.split(',')]
            if len(DNA_seq) == set_DNA_len:
                data_list.append((DNA_seq, bind_list))
    return data_list


def generate_zero_sequence(length):
    return list(np.zeros(length))


def main():
    data_cnf = sys.argv[1]
    # read test data
    yaml = YAML(typ='safe')
    data_cnf = yaml.load(Path(data_cnf))
    test_data = get_data_for_baseline(data_cnf['test'])
    bind_list = [i[1] for i in test_data]
    bind_list = np.array(bind_list)

    zero_sequences = [generate_zero_sequence(len(sample)) for sample in bind_list]
    zero_sequences = np.array(zero_sequences)

    metrics = []
    metrics.append(get_mean_f1(bind_list, zero_sequences))
    metrics.append(get_mean_pcc(bind_list, zero_sequences))
    metrics.append(get_label_ranking_average_precision_score(bind_list, zero_sequences))
    metrics.append(get_mean_accuracy_score(bind_list, zero_sequences))
    metrics.append(get_mean_balanced_accuracy_score(bind_list, binary_sequences))
    metrics.append(get_mean_auc(bind_list, zero_sequences))

    # calculate matrics include pcc f1 lrsp
    logger.info("baseline zero model prediction")
    logger.info(f'f1 score: {metrics[0]:5f}   PCC: {metrics[1]:5f}   lrap: {metrics[2]:5f}   accuracy: {metrics[3]:5f}   balanced_accuracy: {metrics[4]:5f}   AUC: {metrics[5]:5f}')
if __name__ == '__main__':
    main()