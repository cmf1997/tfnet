#!/usr/bin/env python
# -*- encoding: utf-8 -*-

'''
@File : preprocess_chip_data.py
@Time : 2023/11/13 15:20:52
@Author : Cmf
@Version : 1.0
@Desc : preprocess chip-seq data, input as a datafolder, output as a interval and multiple binary labels
blacklist were downloaded from https://github.com/Boyle-Lab/Blacklist/blob/master/lists/hg19-blacklist.v2.bed.gz
test data were downloaded from https://www.synapse.org/#!Synapse:syn6131484/wiki/402033
'''

# here put the import lib

# code
from ruamel.yaml import YAML
from pybedtools import BedTool, Interval
import pybedtools
import numpy as np
import parmap
import click
from pathlib import Path
import csv
from tfnet.all_tfs import all_tfs
import os
from Bio import SeqIO
import pyBigWig
import random

import pdb

def get_genome_bed(genome_sizes_file):
    genome_sizes_info = np.loadtxt(genome_sizes_file, dtype=str)
    chroms = list(genome_sizes_info[:,0])
    chroms_sizes = list(genome_sizes_info[:,1].astype(int))
    genome_bed = []
    for chrom, chrom_size in zip(chroms, chroms_sizes):
        genome_bed.append(Interval(chrom, 0, chrom_size))
    genome_bed = BedTool(genome_bed)
    genome_bed = genome_bed.sort()
    return chroms, chroms_sizes, genome_bed


def make_blacklist(blacklist_file, genome_sizes_file, genome_window_size):
    blacklist = BedTool(blacklist_file)
    blacklist = blacklist.slop(g=genome_sizes_file, b=genome_window_size)
    # Add ends of the chromosomes to the blacklist
    genome_sizes_info = np.loadtxt(genome_sizes_file, dtype=str)
    chroms = list(genome_sizes_info[:,0])
    chroms_sizes = list(genome_sizes_info[:,1].astype(int))
    blacklist2 = []
    for chrom, size in zip(chroms, chroms_sizes):
        blacklist2.append(Interval(chrom, 0, genome_window_size))
        blacklist2.append(Interval(chrom, size - genome_window_size, size))
    blacklist2 = BedTool(blacklist2)
    blacklist = blacklist.cat(blacklist2)
    return blacklist


def get_chip_beds(input_dir):
    chip_info_file = input_dir + '/chip.txt'
    chip_info = np.loadtxt(chip_info_file, dtype=str)
    if len(chip_info.shape) == 1:
        chip_info = np.reshape(chip_info, (-1,len(chip_info)))
    tfs = list(chip_info[:, 1])
    chip_bed_files = [input_dir + '/' + i for i in chip_info[:,0]]
    chip_beds = [BedTool(chip_bed_file) for chip_bed_file in chip_bed_files]
    # ---------------------- Sorting BED files ---------------------- # 
    chip_beds = [chip_bed.sort() for chip_bed in chip_beds]
    if len(chip_beds) > 1:
        merged_chip_bed = BedTool.cat(*chip_beds)
        merged_chip_bed = merged_chip_bed.sort()
    else:
        merged_chip_bed = chip_beds[0]
    return tfs, chip_beds, merged_chip_bed


def intersect_count(chip_bed, windows_file):
    windows = BedTool(windows_file)
    #chip_bedgraph = windows.intersect(chip_bed, wa=True, c=True, f=1.0*(genome_window_size/2+1)/genome_window_size, sorted=True)
    chip_bedgraph = windows.intersect(chip_bed, wa=True, c=True, F=1, sorted=True)
    bed_counts = [i.count for i in chip_bedgraph]
    return bed_counts


def load_chip_multiTask(input_dir, genome_sizes_file, genome_window_size, genome_window_step, blacklist):
    tfs, chip_beds, merged_chip_bed = get_chip_beds(input_dir)
    # ---------------------- Removing peaks outside of X chromosome and autosomes ---------------------- #
    chroms, chroms_sizes, genome_bed = get_genome_bed(genome_sizes_file)
    merged_chip_bed = merged_chip_bed.intersect(genome_bed, u=True, sorted=True)

    genome_windows = BedTool().window_maker(g=genome_sizes_file, w=genome_window_size,
                                            s=genome_window_step)
    
    genome_windows = genome_windows.sort()

    # ---------------------- Extracting windows that overlap at least one ChIP interval ---------------------- #
    positive_windows = genome_windows.intersect(merged_chip_bed, u=True, F=1, sorted=True)
    
    # ---------------------- Removing windows that overlap a blacklisted region ---------------------- #
    positive_windows = positive_windows.intersect(blacklist, wa=True, v=True, sorted=True)

    # ---------------------- Generate targets ---------------------- #
    y_positive = parmap.map(intersect_count, chip_beds, positive_windows.fn)
    y_positive = np.array(y_positive, dtype=bool).T
    print('Positive matrix sparsity', (~y_positive).sum()*1.0/np.prod(y_positive.shape))
    merged_chip_slop_bed = merged_chip_bed.slop(g=genome_sizes_file, b=genome_window_size)
    # ---------------------- gather negative windows from the genome that do not overlap with a blacklisted or ChIP region ---------------------- #
    nonnegative_regions_bed = merged_chip_slop_bed.cat(blacklist)

    negative_windows = genome_windows.intersect(nonnegative_regions_bed, wa=True, v=True, sorted=True, output='data/tf_chip/negative_windows.bed')

    return tfs, positive_windows, y_positive, negative_windows


def chroms_filter(feature, chroms):
    if feature.chrom in chroms:
        return True
    return False


def write_result(filename, tfs_bind_datas, result_filefolder):
    with open(result_filefolder + filename +'.bed', 'w') as output_file:
        writer = csv.writer(output_file, delimiter="\t")
        for chrom, start, stop, atac_signal, target_array in tfs_bind_datas:
            writer.writerow([chrom, start, stop, atac_signal, target_array])


def make_pos_features_multiTask(genome_sizes_file, positive_windows, y_positive, valid_chroms, test_chroms, genome_fasta_file, atac_data, result_filefolder):
    chroms, chroms_sizes, genome_bed = get_genome_bed(genome_sizes_file)
    train_chroms = chroms
    for chrom in valid_chroms + test_chroms:
        train_chroms.remove(chrom)

    positive_data_train = []
    positive_data_valid = []
    positive_data_test = []

    # ---------------- Splitting positive windows into training, validation, and testing sets ----------------# 
    for positive_window, target_array in zip(positive_windows, y_positive):
        # ---------------------- check the name of chrom, pass if chr1 chr2 ... pdb if chr19_gl000208_random ... ---------------------- #
        if len(positive_window.chrom) > 8:
            pdb.set_trace()
        chrom = positive_window.chrom
        start = int(positive_window.start)
        stop = int(positive_window.stop)

        target_array = np.array(target_array, dtype=int)
        target_array = np.array(target_array, dtype=str)
        target_array = ','.join(target_array)

        atac_signal = atac_data.values(chrom,start,stop)
        atac_signal = np.array(atac_signal, dtype=str)
        atac_signal = ','.join(atac_signal)

        #window_DNA_seq = BedTool([positive_window]).getfasta(fi=genome_fasta_file)
        #read_seq = open(window_DNA_seq.seqfn).read().split('\n')[1]

        if chrom in test_chroms:
            #positive_data_test.append((read_seq, target_array))
            positive_data_test.append((chrom, start, stop, atac_signal, target_array))
        elif chrom in valid_chroms:
            #positive_data_valid.append((read_seq, target_array))
            positive_data_valid.append((chrom, start, stop, atac_signal, target_array))
        else:
            #positive_data_train.append((read_seq, target_array))
            positive_data_train.append((chrom, start, stop, atac_signal, target_array))

    write_result('pos_data_test',positive_data_test, result_filefolder)
    write_result('pos_data_valid',positive_data_valid, result_filefolder)
    write_result('pos_data_train',positive_data_train, result_filefolder)


def make_neg_features_multiTask(genome_sizes_file, negative_windows, valid_chroms, test_chroms, genome_fasta_file, atac_data, result_filefolder):
    chroms, chroms_sizes, genome_bed = get_genome_bed(genome_sizes_file)
    train_chroms = chroms
    for chrom in valid_chroms + test_chroms:
        train_chroms.remove(chrom)

    negative_data_train = []
    negative_data_valid = []
    negative_data_test = []

    target_array = [0 for i in range(len(all_tfs))]
    target_array = np.array(target_array, dtype=str)
    target_array = ','.join(target_array)

    for negative_window in negative_windows:
        # ---------------------- check the name of chrom, pass if chr1 chr2 ... pdb if chr19_gl000208_random ... ---------------------- #
        if len(negative_window.chrom) > 8:
            pdb.set_trace()
        chrom = negative_window.chrom
        start = int(negative_window.start)
        stop = int(negative_window.stop)

        atac_signal = atac_data.values(chrom,start,stop)
        atac_signal = np.array(atac_signal, dtype=str)
        atac_signal = ','.join(atac_signal)

        #window_DNA_seq = BedTool([Interval(chrom, start, stop)]).getfasta(fi=genome_fasta_file)
        #read_seq = open(window_DNA_seq.seqfn).read().split('\n')[1]

        if chrom in test_chroms:
            #negative_data_test.append((read_seq, target_array))
            negative_data_test.append((chrom, start, stop, atac_signal, target_array))
        elif chrom in valid_chroms:
            #negative_data_valid.append((read_seq, target_array))
            negative_data_valid.append((chrom, start, stop, atac_signal, target_array))
        else:
            #negative_data_train.append((read_seq, target_array))
            negative_data_train.append((chrom, start, stop, atac_signal, target_array))

    # ---------------------- constrain the size of negative file ---------------------- #
    negative_data_test = random.sample(negative_data_test, 1000)
    negative_data_valid = random.sample(negative_data_valid, 1000)
    negative_data_train = random.sample(negative_data_train, 10000)

    write_result('neg_data_test',negative_data_test, result_filefolder)
    write_result('neg_data_valid',negative_data_valid, result_filefolder)
    write_result('neg_data_train',negative_data_train, result_filefolder)


@click.command()
@click.option('-d', '--data-cnf', type=click.Path(exists=True))
@click.option('-m', '--model-cnf', type=click.Path(exists=True))
def main(data_cnf, model_cnf):
    yaml = YAML(typ='safe')
    data_cnf, model_cnf = yaml.load(Path(data_cnf)), yaml.load(Path(model_cnf))

    input_dir = data_cnf['input_dir']

    atac_file = data_cnf['atac_file']
    atac_data = pyBigWig.open(atac_file)
    # ---------------------- consider adjust the value of atac signal to 0-1 by dividing the max ---------------------- #
    atac_signal_max = atac_data.header()['maxVal']

    genome_window_size = model_cnf['padding']['DNA_len']
    genome_window_step = data_cnf['genome_window_step']

    genome_sizes_file = data_cnf['genome_sizes_file']
    blacklist_file = data_cnf['blacklist_file']

    result_filefolder = input_dir
    valid_chroms = data_cnf['valid_chroms']
    test_chroms = data_cnf['test_chroms']

    genome_fasta_file = data_cnf['genome_fasta_file']

    pybedtools.set_tempdir('/Users/cmf/Downloads/tmp')

    blacklist = make_blacklist(blacklist_file, genome_sizes_file, genome_window_size)
    tfs, positive_windows, y_positive, negative_windows = load_chip_multiTask(input_dir,genome_sizes_file, genome_window_size, genome_window_step, blacklist)
    make_pos_features_multiTask(genome_sizes_file, positive_windows, y_positive, valid_chroms, test_chroms, genome_fasta_file, atac_data, result_filefolder)
    make_neg_features_multiTask(genome_sizes_file, negative_windows, valid_chroms, test_chroms, genome_fasta_file, atac_data, result_filefolder)

    # ---------------------- use seqkit to grep fasta due to low effiency of pybedtools getfasta---------------------- #
    os.system("seqkit subseq --bed {}pos_data_test.bed {} > {}pos_data_test.fa".format(result_filefolder, genome_fasta_file, result_filefolder))
    os.system("seqkit subseq --bed {}neg_data_test.bed {} > {}neg_data_test.fa".format(result_filefolder, genome_fasta_file, result_filefolder))

    os.system("seqkit subseq --bed {}pos_data_valid.bed {} > {}pos_data_valid.fa".format(result_filefolder, genome_fasta_file, result_filefolder))
    os.system("seqkit subseq --bed {}neg_data_valid.bed {} > {}neg_data_valid.fa".format(result_filefolder, genome_fasta_file, result_filefolder))

    os.system("seqkit subseq --bed {}pos_data_train.bed {} > {}pos_data_train.fa".format(result_filefolder, genome_fasta_file, result_filefolder))
    os.system("seqkit subseq --bed {}neg_data_train.bed {} > {}neg_data_train.fa".format(result_filefolder, genome_fasta_file, result_filefolder))


    for file_name in ['pos_data_test', 'neg_data_test', 'pos_data_valid', 'neg_data_valid', 'pos_data_train', 'neg_data_train']:
        data_fasta = SeqIO.parse(result_filefolder + str(file_name) + ".fa", "fasta")
        #data_bed = np.loadtxt(result_filefolder + str(file_name) + '.bed', dtype=str)
        #assert len(list(data_fasta)) == data_bed.shape[0]
        with open(result_filefolder + file_name + '.txt',"w") as fp :
            writer = csv.writer(fp, delimiter="\t")
            #for fasta, target_array in zip(data_fasta, data_bed):
            for fasta in data_fasta:
                if str(fasta.seq).count("N") > 100 or str(fasta.seq).count("n") > 100 :
                    print(f"sequence contain multifpe N, pass")
                
                else:
                    writer.writerow([str(fasta.seq), fasta.description.split('.')[1]])

if __name__ == '__main__':
    main()