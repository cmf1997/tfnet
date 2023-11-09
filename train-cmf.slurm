#!/bin/bash

#SBATCH --job-name=train_model
#SBATCH --partition=dgx2
#SBATCH -n 1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --mail-type=end
#SBATCH --mail-user=mufeng_cai@163.com
#SBATCH --output=%j.out
#SBATCH --error=%j.err

#conda activate deepMHC-cmf

#python main.py -d configure/dataset.yaml -m configure/deepmhcii.yaml --mode cmf-B.1 -n 10          
python main.py -d configure/data.yaml -m configure/deepmhcii.yaml --mode train -n 5
