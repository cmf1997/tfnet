import ast
import numpy as np
import pdb
from logzero import logger
from tfnet.backup.evaluation import get_mean_auc, get_mean_f1, get_label_ranking_average_precision_score, get_mean_accuracy_score, get_mean_balanced_accuracy_score, get_mean_recall, get_mean_aupr
import warnings

warnings.filterwarnings("ignore",category=UserWarning)
np.seterr(divide='ignore', invalid='ignore')

def read_predict(predict_file):
    with open(predict_file, 'r') as fp:
        chr_list = []
        start_list = []
        stop_list = []
        targets_list = []
        ori_predict_list = []
        predict_list = []
        for line in fp:
            chr, start, stop, targets, ori_predict, predict = line.split('\t')
            targets_list.append(ast.literal_eval(targets))
            ori_predict_list.append(ast.literal_eval(ori_predict))
            predict_list.append(ast.literal_eval(predict))
    targets_array = np.array(targets_list)
    ori_predict_array = np.array(ori_predict_list)
    predict_array = np.array(predict_list)
    return targets_array,ori_predict_array,predict_array


#targets_array_g, ori_predict_array_g ,predict_array_g = read_predict("../TFNet_shared_G/results/SimpleCNN_2d.eval.tsv")
#targets_array_h, ori_predict_array_h ,predict_array_h = read_predict("../TFNet_shared_G/results/SimpleCNN_2d.eval.tsv")
#targets_array_h, ori_predict_array_h ,predict_array_h = read_predict("../TFNet_shared_H/results/SimpleCNN_2d.eval.tsv")




targets_array_g, ori_predict_array_g ,predict_array_g = read_predict("../TFNET-MULTI-TF/results/SimpleCNN_2d.eval.tsv")
targets_array_h, ori_predict_array_h ,predict_array_h = read_predict("../TFNET-MULTI-TF/results/SimpleCNN_2d.eval.tsv")

merge_predict = np.divide(np.add(ori_predict_array_g,ori_predict_array_h),2)

#pdb.set_trace()

mean_auc = get_mean_auc(targets_array_g, merge_predict)
f1_score = get_mean_f1(targets_array_g, merge_predict)
recall_score = get_mean_recall(targets_array_g, merge_predict)
aupr = get_mean_aupr(targets_array_g, merge_predict)
lrap = get_label_ranking_average_precision_score(targets_array_g, merge_predict)
accuracy = get_mean_accuracy_score(targets_array_g, merge_predict)
balanced_accuracy = get_mean_balanced_accuracy_score(targets_array_g, merge_predict)
logger.info(
                        f'mean_auc: {mean_auc:.5f}  '
                        f'aupr: {aupr:.5f}  '
                        f'recall score: {recall_score:.5f}  '
                        f'f1 score: {f1_score:.5f}  '
                        f'lrap: {lrap:.5f}  '
                        f'accuracy: {accuracy:.5f}  '
                        f'balanced accuracy: {balanced_accuracy:.5f}'
                        )
