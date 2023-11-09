#!/bin/bash

#SBATCH --job-name=predict-TFNet
#SBATCH --partition=dgx2
#SBATCH -n 1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --mail-type=end
#SBATCH --mail-user=562590731@qq.com
#SBATCH --output=%j.out
#SBATCH --error=%j.err

#conda activate deepMHC-cmf

python main.py -d configure/data.yaml -m configure/deepmhcii.yaml --mode eval -n 2
#python main.py -d configure/dataset.yaml -m configure/deepmhcii.yaml --mode cmf_auc -n 10
