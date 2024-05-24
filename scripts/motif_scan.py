#!/usr/bin/env python
# -*- encoding: utf-8 -*-

'''
@File : motif_scan.py
@Time : 2023/11/22 11:37:51
@Author : Cmf
@Version : 1.0
@Desc : 
all tf motifs were downloaded from https://jaspar.elixir.no/downloads/
'''

# here put the import lib
import click
from tfnet.all_tfs import all_tfs
import gzip
import pysam
import numpy as np
from ruamel.yaml import YAML
from pathlib import Path
import os
import pandas as pd
import seaborn as sns
import pdb
# code

# ---------------------- data ---------------------- #
def tfs_dict(all_tfs):
    all_tfs_dict = {}
    for index, tf in enumerate(all_tfs):
        all_tfs_dict[tf] = index


def get_data_bed(data_file, genome_fasta_file):
    genome_fasta = pysam.Fastafile(genome_fasta_file)
    data_list = []
    with gzip.open(data_file, 'rt') as fp:
        for line in fp:
            chr, start, stop, bind_list  = line.split('\t')
            start = int(start)
            stop = int(stop)
            DNA_seq = genome_fasta.fetch(chr, start, stop)
            data_list.append((chr, start, stop, DNA_seq, bind_list))
    genome_fasta.close()
    return data_list, len(data_list)


def generate_fasta(data_list):
    DNA_seqs = [ i[3] for i in data_list]
    with open('predict_dna_seq.fa','w') as fp:
        for index, DNA_seq in enumerate(DNA_seqs):
            fp.write(">{}\n".format(index))
            fp.write("{}\n".format(DNA_seq))


# ---------------------- cat all meme file to one ---------------------- #
# ---------------------- run fimo of meme suite ---------------------- #
# /Users/cmf/meme/bin/fimo --qv-thresh --thresh 1e-2 --o data/fimo_out data/all_tfs.meme data/predict_dna_seq.fa
# ---------------------- process fimo out to a binary list indicating the ocurrency of specific tf ---------------------- #
def parse_fimo_out(fimo_out_file, data_len, cutoff):
    all_tfs_dict = {}
    for index, tf in enumerate(all_tfs):
        all_tfs_dict[tf] = index
    fimo_out = pd.read_csv(fimo_out_file, delimiter='\t', header=0, comment='#')
    fimo_out = fimo_out[fimo_out['q-value'] < cutoff]
    sequence_names = set(fimo_out['sequence_name'])
    motif_targets=[]
    for sequence_name in range(data_len):
        if sequence_name in sequence_names:
            single_motif_target = np.zeros(len(all_tfs), dtype=int)
            single_fimo_out = fimo_out[fimo_out['sequence_name']==sequence_name]
            bind_tfs = set(single_fimo_out['motif_alt_id'])
            for bind_tf in bind_tfs:
                index = all_tfs_dict[bind_tf]
                single_motif_target[index] = 1
            motif_targets.append(single_motif_target)
        else:
            motif_targets.append(np.zeros(len(all_tfs), dtype=int))
    return motif_targets


def plot_co_occurrence_matrix(motif_array):
    motif_array = np.array(motif_array)
    co_occurrence_matrix = np.dot(motif_array.T,motif_array)
    row, col = np.diag_indices_from(co_occurrence_matrix)
    co_occurrence_matrix[row,col] = 0
    g = sns.clustermap(co_occurrence_matrix, center=0, cmap="vlag",
                   dendrogram_ratio=(0, .2),
                   cbar_pos=(.02, .32, .03, .2),
                   linewidths=.75, figsize=(12, 13))
    g.savefig('results/co_occurrence_matrix_motif.pdf')


def write_motif_result(data_file, motif_targets):
    with gzip.open(data_file, 'rt') as fp:
        lines = fp.read().splitlines()
    filepath = f'{Path(data_file).stem}_with_motif{Path(data_file).suffix}'
    with open(filepath, "w") as fp:
        for line, motif_target in zip(lines, motif_targets):
            fp.write('\t'.join([line, motif_target]) )


@click.command()
@click.option('-d', '--data-cnf', type=click.Path(exists=True))
@click.option('-m', '--model-cnf', type=click.Path(exists=True))
def main():
    yaml = YAML(typ='safe')
    data_cnf = yaml.load(Path(data_cnf))
    data_file = ''
    genome_fasta_file = data_cnf['genome_fasta_file']
    data_list, data_len = get_data_bed(data_file, genome_fasta_file)
    generate_fasta(data_list)
    fimo_path = '/Users/cmf/meme/bin/fimo'
    os.system("{} --qv-thresh --thresh 1e-4 --o data/fimo_out data/all_tfs.meme data/predict_dna_seq.fa".format(fimo_path))
    fimo_out_file = 'data/fimo_out/fimo.tsv'

    cutoff = 1
    motif_targets = parse_fimo_out(fimo_out_file, data_len, cutoff)
    plot_co_occurrence_matrix(motif_targets)
    write_motif_result(data_file, motif_targets)
    
if __name__ == '__main__':
    main()