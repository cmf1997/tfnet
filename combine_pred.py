import ast
import numpy as np
import pdb
from logzero import logger
from tfnet.all_tfs import all_tfs
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tfnet.evaluation import get_mean_auc, get_auc, get_mean_f1, get_f1, get_label_ranking_average_precision_score, get_mean_accuracy_score, get_mean_balanced_accuracy_score, get_mean_recall, get_recall, get_mean_aupr, get_aupr, get_precision, get_mean_precision
import warnings
import sys


sys.setrecursionlimit(65520)
sns.set_theme(style="ticks")
warnings.filterwarnings("ignore",category=UserWarning)
np.seterr(divide='ignore', invalid='ignore')
palette = ['#DC143C', '#4169E1','#ff69b4']

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



# ---------------------- merge_predict ---------------------- #
# ---------------------- merge_predict ---------------------- #
'''
#targets_array_g, ori_predict_array_g ,predict_array_g = read_predict("../TFNet_shared_G/results/SimpleCNN_2d.eval.tsv")
#targets_array_h, ori_predict_array_h ,predict_array_h = read_predict("../TFNet_shared_H/results/SimpleCNN_2d.eval.tsv")
targets_array_g, ori_predict_array_g ,predict_array_g = read_predict("/Users/cmf/Downloads/TFNet-multi-tf/results/SimpleCNN_2d_G.eval.tsv")
targets_array_h, ori_predict_array_h ,predict_array_h = read_predict("/Users/cmf/Downloads/TFNet-multi-tf/results/SimpleCNN_2d_H.eval.tsv")

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


auc_data = pd.DataFrame({"tf_name" : all_tfs,
                         "GM12878":get_auc(targets_array_g, ori_predict_array_g),
                         "H1ESC" : get_auc(targets_array_g, ori_predict_array_h),
                         "Merge" : get_auc(targets_array_g, merge_predict),
                         "type" : "AUC"
                         }).melt(id_vars=['tf_name','type'],var_name="Model", value_name="value")

aupr_data = pd.DataFrame({"tf_name" : all_tfs,
                         "GM12878":get_aupr(targets_array_g, ori_predict_array_g),
                         "H1ESC" : get_aupr(targets_array_g, ori_predict_array_h),
                         "Merge" : get_aupr(targets_array_g, merge_predict),
                         "type" : "AUPR"
                         }).melt(id_vars=['tf_name','type'],var_name="Model", value_name="value")

recall_data = pd.DataFrame({"tf_name" : all_tfs,
                            "GM12878":get_recall(targets_array_g, ori_predict_array_g),
                            "H1ESC" : get_recall(targets_array_g, ori_predict_array_h),
                            "Merge" : get_recall(targets_array_g, merge_predict),
                            "type" : "RECALL"
                            }).melt(id_vars=['tf_name','type'],var_name="Model", value_name="value")

f1_data = pd.DataFrame({"tf_name" : all_tfs,
                        "GM12878":get_f1(targets_array_g, ori_predict_array_g),
                        "H1ESC" : get_f1(targets_array_g, ori_predict_array_h),
                        "Merge" : get_f1(targets_array_g, merge_predict),
                        "type" : "F1"
                        }).melt(id_vars=['tf_name','type'],var_name="Model", value_name="value")

eval_data = pd.concat([auc_data, aupr_data, recall_data, f1_data], ignore_index=True)

sns.relplot(
    data=eval_data,
    x="tf_name", y="value",
    hue="Model",
    kind='line', 
    col="type", col_wrap=2,
    palette=palette,
    height=5,
    aspect=1.8,
    lw=3.5, alpha = 0.7
)
plt.savefig("results/predict_K562.eval.all.pdf")

sns.relplot(
    data=eval_data[eval_data['type']== "AUC"],
    x="tf_name", y="value",
    hue="Model",
    kind='line', 
    #col="type", col_wrap=2,
    palette=palette,
    height=5,
    aspect=1.4,
    lw=3.5, alpha = 0.7
)
plt.title("Cross Cell Line Prediction - K562 test dataset")
plt.tick_params(axis='x', labelrotation=45)
plt.xlabel("")
plt.ylabel("AUC")
plt.savefig("results/predict_K562.eval.AUC.pdf")


# ---------------------- plot auc of model G and H for eval G and H test dataset ---------------------- #
G_eval_G = pd.read_csv('results/G_eval_G_SimpleCNN_2d.eval.repl.tsv', sep='\t', index_col=0, header=0)
G_eval_G['Model'] = 'GM12878'
G_eval_H = pd.read_csv('results/G_eval_H_SimpleCNN_2d.eval.repl.tsv', sep='\t', index_col=0, header=0)
G_eval_H['Model'] = 'GM12878'
H_eval_G = pd.read_csv('results/H_eval_G_SimpleCNN_2d.eval.repl.tsv', sep='\t', index_col=0, header=0)
H_eval_G['Model'] = 'H1ESC'
H_eval_H = pd.read_csv('results/H_eval_H_SimpleCNN_2d.eval.repl.tsv', sep='\t', index_col=0, header=0)
H_eval_H['Model'] = 'H1ESC'

eval_G = pd.concat([G_eval_G[['TF_name', 'AUC', 'Model']], H_eval_G[['TF_name', 'AUC', 'Model']]])
eval_H = pd.concat([G_eval_H[['TF_name', 'AUC', 'Model']], H_eval_H[['TF_name', 'AUC', 'Model']]])

sns.relplot(
    data=eval_G,
    x="TF_name", y="AUC",
    hue="Model",
    kind='line', 
    #col="type", col_wrap=2,
    palette=palette[:2],
    height=5,
    aspect=1.4,
    lw=3.5, alpha = 0.7
)
plt.title("Cross Cell Line Prediction - GM12878 test dataset")
plt.tick_params(axis='x', labelrotation=45)
plt.xlabel("")
plt.ylabel("AUC")
plt.savefig("results/predict_GM12878.eval.AUC.pdf")


sns.relplot(
    data=eval_H,
    x="TF_name", y="AUC",
    hue="Model",
    kind='line', 
    #col="type", col_wrap=2,
    palette=palette[:2],
    height=5,
    aspect=1.4,
    lw=3.5, alpha = 0.7
)
plt.title("Cross Cell Line Prediction - H1ESC test dataset")
plt.tick_params(axis='x', labelrotation=45)
plt.xlabel("")
plt.ylabel("AUC")
plt.savefig("results/predict_H1ESC.eval.AUC.pdf")
'''




# ---------------------- classweight ---------------------- #
# ---------------------- classweight ---------------------- #
'''
pre_cutoffs = [0.4, 0.45, 0.5, 0.55, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]

targets_array_weight, ori_predict_array_weight ,predict_array_weight = read_predict("/Users/cmf/Downloads/TFNet-multi-tf/results/classweight/weight/SimpleCNN_2d.eval.tsv")
targets_array_noweight, ori_predict_array_noweight ,predict_array_noweight = read_predict("/Users/cmf/Downloads/TFNet-multi-tf/results/classweight/noweight/SimpleCNN_2d.eval.tsv")

auc_weight = get_auc(targets_array_weight, ori_predict_array_weight)
auc_noweight = get_auc(targets_array_noweight, ori_predict_array_noweight)

aupr_weight = get_aupr(targets_array_weight, ori_predict_array_weight)
aupr_noweight = get_aupr(targets_array_noweight, ori_predict_array_noweight)

cutoffs = []
recall_weight = []
recall_noweight = []
precision_weight = []
precision_noweight = []
f1_weight = []
f1_noweight = []
balanced_accuracy_weight = []
balanced_accuracy_noweight = []

for cutoff in pre_cutoffs:
    cutoffs.append(cutoff)
    recall_weight.append(get_mean_recall(targets_array_weight, ori_predict_array_weight, cutoff=cutoff))
    recall_noweight.append(get_mean_recall(targets_array_noweight, ori_predict_array_noweight, cutoff=cutoff))
    precision_weight.append(get_mean_precision(targets_array_weight, ori_predict_array_weight, cutoff=cutoff))
    precision_noweight.append(get_mean_precision(targets_array_noweight, ori_predict_array_noweight, cutoff=cutoff))
    f1_weight.append(get_mean_f1(targets_array_weight, ori_predict_array_weight, cutoff=cutoff))
    f1_noweight.append(get_mean_f1(targets_array_noweight, ori_predict_array_noweight, cutoff=cutoff))
    balanced_accuracy_weight.append(get_mean_balanced_accuracy_score(targets_array_weight, ori_predict_array_weight, axis = 0, cutoff=cutoff))
    balanced_accuracy_noweight.append(get_mean_balanced_accuracy_score(targets_array_noweight, ori_predict_array_noweight, axis = 0, cutoff=cutoff))

cutoff_data = pd.DataFrame({"CUTOFF" : cutoffs,
                        "RECALL_classweight":recall_weight,
                        "RECALL" : recall_noweight,
                        "PRECISION_classweight" : precision_weight,
                        "PRECISION" : precision_noweight,
                        "F1_classweight": f1_weight,
                        "F1" : f1_noweight,
                        "BALANCE_ACCURACY_classweight" : balanced_accuracy_weight,
                        "BALANCE_ACCURACY" : balanced_accuracy_noweight
                        })

cutoff_data = cutoff_data.melt(id_vars=['CUTOFF'],var_name="Model", value_name="value")
cutoff_data['type'] = [ i.split("_")[0] for i in cutoff_data['Model'].tolist() ]

sns.relplot(
    data=cutoff_data,
    x="CUTOFF", y="value",
    hue="Model",
    kind='line', 
    col="type", col_wrap=2,
    palette=palette[:2],
    height=4,
    aspect=1,
    lw=6, alpha = 0.7,
    facet_kws={'sharey': False, 'sharex': False}
)
plt.title("for each prediction    for each TFs")
plt.xlabel("CUTOFF")
plt.ylabel("Value")
plt.savefig("results/eval.GM12878.compare.classweight.pdf")
'''


# ---------------------- pseudosequences tfnet ---------------------- #
# ---------------------- pseudosequences tfnet ---------------------- #
pre_cutoffs = [0.4, 0.45, 0.5, 0.55, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]

# ---------------------- load data ---------------------- #
targets_array_tfnet, ori_predict_array_tfnet ,predict_array_tfnet = read_predict("/Users/cmf/Downloads/TFNet-multi-tf/results/tfnet/TFNet.eval.tsv")
targets_array_weight, ori_predict_array_weight ,predict_array_weight = read_predict("/Users/cmf/Downloads/TFNet-multi-tf/results/classweight/weight/SimpleCNN_2d.eval.tsv")
targets_array_noweight, ori_predict_array_noweight ,predict_array_noweight = read_predict("/Users/cmf/Downloads/TFNet-multi-tf/results/classweight/noweight/SimpleCNN_2d.eval.tsv")

# ---------------------- auc aupr ---------------------- #
auc_tfnet = get_auc(targets_array_tfnet, ori_predict_array_tfnet)
auc_weight = get_auc(targets_array_weight, ori_predict_array_weight)
auc_noweight = get_auc(targets_array_noweight, ori_predict_array_noweight)

aupr_weight = get_aupr(targets_array_tfnet, ori_predict_array_tfnet)
aupr_weight = get_aupr(targets_array_weight, ori_predict_array_weight)
aupr_noweight = get_aupr(targets_array_noweight, ori_predict_array_noweight)


# ---------------------- eval ---------------------- #
cutoffs = []

recall_tfnet = []
recall_weight = []
recall_noweight = []

precision_tfnet = []
precision_weight = []
precision_noweight = []

f1_tfnet = []
f1_weight = []
f1_noweight = []

balanced_accuracy_tfnet = []
balanced_accuracy_weight = []
balanced_accuracy_noweight = []

for cutoff in pre_cutoffs:
    cutoffs.append(cutoff)

    recall_tfnet.append(get_mean_recall(targets_array_tfnet, ori_predict_array_tfnet, cutoff=cutoff))
    recall_weight.append(get_mean_recall(targets_array_weight, ori_predict_array_weight, cutoff=cutoff))
    recall_noweight.append(get_mean_recall(targets_array_noweight, ori_predict_array_noweight, cutoff=cutoff))

    precision_tfnet.append(get_mean_precision(targets_array_tfnet, ori_predict_array_tfnet, cutoff=cutoff))
    precision_weight.append(get_mean_precision(targets_array_weight, ori_predict_array_weight, cutoff=cutoff))
    precision_noweight.append(get_mean_precision(targets_array_noweight, ori_predict_array_noweight, cutoff=cutoff))

    f1_tfnet.append(get_mean_f1(targets_array_tfnet, ori_predict_array_tfnet, cutoff=cutoff))
    f1_weight.append(get_mean_f1(targets_array_weight, ori_predict_array_weight, cutoff=cutoff))
    f1_noweight.append(get_mean_f1(targets_array_noweight, ori_predict_array_noweight, cutoff=cutoff))

    balanced_accuracy_tfnet.append(get_mean_balanced_accuracy_score(targets_array_tfnet, ori_predict_array_tfnet, axis = 0, cutoff=cutoff))
    balanced_accuracy_weight.append(get_mean_balanced_accuracy_score(targets_array_weight, ori_predict_array_weight, axis = 0, cutoff=cutoff))
    balanced_accuracy_noweight.append(get_mean_balanced_accuracy_score(targets_array_noweight, ori_predict_array_noweight, axis = 0, cutoff=cutoff))

cutoff_data = pd.DataFrame({"CUTOFF" : cutoffs,
                        "RECALL_tfnet" : recall_tfnet,
                        "RECALL_classweight":recall_weight,
                        "RECALL" : recall_noweight,

                        "PRECISION_tfnet" : precision_tfnet,
                        "PRECISION_classweight" : precision_weight,
                        "PRECISION" : precision_noweight,

                        "F1_tfnet" : f1_tfnet,
                        "F1_classweight": f1_weight,
                        "F1" : f1_noweight,

                        "BALANCE_ACCURACY_tfnet" : balanced_accuracy_tfnet,
                        "BALANCE_ACCURACY_classweight" : balanced_accuracy_weight,
                        "BALANCE_ACCURACY" : balanced_accuracy_noweight
                        })

cutoff_data = cutoff_data.melt(id_vars=['CUTOFF'],var_name="Model", value_name="value")
cutoff_data['type'] = [ i.split("_")[0] for i in cutoff_data['Model'].tolist() ]

# ---------------------- plot ---------------------- #
sns.relplot(
    data=cutoff_data,
    x="CUTOFF", y="value",
    hue="Model",
    kind='line', 
    col="type", col_wrap=2,
    palette=palette[:3],
    height=4,
    aspect=1,
    lw=6, alpha = 0.7,
    facet_kws={'sharey': False, 'sharex': False}
)
plt.title("pseudosequence TFNet")
plt.xlabel("CUTOFF")
plt.ylabel("Value")
plt.savefig("results/eval.GM12878.compare.tfnet.pdf")
