#!/bin/bash

#SBATCH --job-name=predict-mhcii
#SBATCH --partition=cpu
#SBATCH -n 40
#SBATCH --ntasks-per-node=40
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --exclusive
#SBATCH --mail-user=562590731@qq.com

#source activate deepMHC-cmf
    
python main.py -d configure/dataset.yaml -m configure/deepmhcii.yaml --mode cmf_del_repair -n 10 
