#!/bin/bash

#SBATCH --job-name=train-model
#SBATCH --partition=cpu
#SBATCH -n 10
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --exclusive
#SBATCH --mail-user=mufeng_cai@163.com

#source activate deepMHC-cmf
    
python main.py -d configure/data.yaml -m configure/deepmhcii.yaml --mode train -n 5
