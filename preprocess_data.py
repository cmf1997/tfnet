#!/usr/bin/env python
# -*- encoding: utf-8 -*-

'''
@File : preprocess_chip_data.py
@Time : 2023/11/13 15:20:52
@Author : Cmf
@Version : 1.0
@Desc : preprocess atac and chip-seq data, output as a DNA sequence,  Chromatin accessibility and multiple binary labels
blacklist were downloaded from https://github.com/Boyle-Lab/Blacklist/blob/master/lists/hg19-blacklist.v2.bed.gz
test data were downloaded from https://www.synapse.org/#!Synapse:syn6131484/wiki/402033
EP300,TAF1 were removed due to absent of DBD
'''

# here put the import lib
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
import pyBigWig
import pysam
import re
import random

import pdb

# code
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


def intersect_count(chip_bed, windows_file, target_window_size):
    windows = BedTool(windows_file)
    chip_bedgraph = windows.intersect(chip_bed, wa=True, c=True, f=1.0*(target_window_size/2+1)/target_window_size, sorted=True)
    bed_counts = [i.count for i in chip_bedgraph]
    return bed_counts


def load_chip_multiTask(input_dir, genome_sizes_file, target_window_size, genome_window_size, genome_window_step, blacklist):
    tfs, chip_beds, merged_chip_bed = get_chip_beds(input_dir)
    # ---------------------- Removing peaks outside of X chromosome and autosomes ---------------------- #
    chroms, chroms_sizes, genome_bed = get_genome_bed(genome_sizes_file)
    merged_chip_bed = merged_chip_bed.intersect(genome_bed, u=True, sorted=True)

    genome_windows = BedTool().window_maker(g=genome_sizes_file, w=target_window_size,
                                            s=genome_window_step)
    
    genome_windows = genome_windows.sort()

    # ---------------------- Extracting windows that overlap at least one ChIP interval ---------------------- #
    positive_windows = genome_windows.intersect(merged_chip_bed, u=True, f=1.0*(target_window_size/2+1)/target_window_size, sorted=True) # make sure same f parameter with function intersect_count
    
    # ---------------------- Removing windows that overlap a blacklisted region ---------------------- #
    positive_windows = positive_windows.intersect(blacklist, wa=True, v=True, sorted=True)

    # ---------------------- Generate targets ---------------------- #
    y_positive = parmap.map(intersect_count, chip_beds, positive_windows.fn, target_window_size)
    y_positive = np.array(y_positive, dtype=bool).T
    print('Positive matrix sparsity', (~y_positive).sum()*1.0/np.prod(y_positive.shape))
    merged_chip_slop_bed = merged_chip_bed.slop(g=genome_sizes_file, b=genome_window_size)
    # ---------------------- gather negative windows from the genome that do not overlap with a blacklisted or ChIP region ---------------------- #
    nonnegative_regions_bed = merged_chip_slop_bed.cat(blacklist)

    negative_windows = genome_windows.intersect(nonnegative_regions_bed, wa=True, v=True, sorted=True)


    return tfs, positive_windows, y_positive, negative_windows,nonnegative_regions_bed


def chroms_filter(feature, chroms):
    if feature.chrom in chroms:
        return True
    return False


def write_single_result(filename, tfs_bind_data, result_filefolder):
    with open(result_filefolder + filename +'.txt', 'a') as output_file:
        writer = csv.writer(output_file, delimiter="\t")
        window_fasta, bw_signal, target_array = tfs_bind_data
        bw_signal = [bw_signal[index] for index in range(len(bw_signal))]
        writer.writerow([window_fasta, bw_signal, target_array])


def make_pos_features_multiTask(genome_sizes_file, genome_window_size, positive_windows, y_positive, valid_chroms, test_chroms, genome_fasta_file, bigwig_data, result_filefolder):
    chroms, chroms_sizes, genome_bed = get_genome_bed(genome_sizes_file)
    train_chroms = chroms
    for chrom in valid_chroms + test_chroms:
        train_chroms.remove(chrom)

    # ---------------- Splitting positive windows into training, validation, and testing sets ----------------# 
    genome_fasta = pysam.Fastafile(genome_fasta_file)
    for positive_window, target_array in zip(positive_windows, y_positive):
        # ---------------------- check the name of chrom, pass if chr1 chr2 ... pdb if chr19_gl000208_random ... ---------------------- #
        if len(positive_window.chrom) > 8:
            pdb.set_trace()
        chrom = positive_window.chrom
        start = int(positive_window.start)
        stop = int(positive_window.stop)
        med = int((start + stop) / 2)
        start = med - int(genome_window_size) / 2
        stop = med + int(genome_window_size) / 2

        # ---------------------- sequences ---------------------- #
        window_fasta = genome_fasta.fetch(chrom, start, stop)
        if len(re.findall('[atcgn]', window_fasta.lower())) != len(window_fasta):
            continue
        
        # ---------------------- targets ---------------------- #
        target_array = np.array(target_array, dtype=int)
        target_array = np.array(target_array, dtype=str)
        target_array = ','.join(target_array)


        # ---------------------- bigwig signal ---------------------- #
        mappability_signal = {}
        for index in range(len(bigwig_data)):
            mappability_signal[index] = np.array(bigwig_data[index].values(chrom,start,stop))
            mappability_signal[index][np.isnan(mappability_signal[index])] = 0
            mappability_signal[index] = ','.join(map(str,mappability_signal[index]))

        # ---------------------- write pos data ---------------------- #
        if chrom in test_chroms:
            positive_data_test = [window_fasta, mappability_signal, target_array]
            write_single_result('pos_data_test',positive_data_test, result_filefolder)
        elif chrom in valid_chroms:
            positive_data_valid = [window_fasta, mappability_signal, target_array]
            write_single_result('pos_data_valid',positive_data_valid, result_filefolder)
        else:
            positive_data_train = [window_fasta, mappability_signal, target_array]
            write_single_result('pos_data_train',positive_data_train, result_filefolder)
    

    genome_fasta.close()



def subset_chroms(chroms, bed):
    result = bed.filter(chroms_filter, chroms).saveas()
    return BedTool(result.fn)



def make_neg_features_multiTask(genome_sizes_file, genome_window_size, positive_windows, nonnegative_regions_bed, valid_chroms, test_chroms, genome_fasta_file, bigwig_data, result_filefolder):
    chroms, chroms_sizes, genome_bed = get_genome_bed(genome_sizes_file)
    train_chroms = chroms
    for chrom in valid_chroms + test_chroms:
        train_chroms.remove(chrom)
    
    genome_bed_train, genome_bed_valid, genome_bed_test = [subset_chroms(chroms_set, genome_bed) for chroms_set in (train_chroms, valid_chroms, test_chroms)]
    # ---------------------- target array is composite of a list of 0 ---------------------- #
    target_array = [0 for i in range(len(all_tfs))]
    target_array = np.array(target_array, dtype=str)
    target_array = ','.join(target_array)

    genome_fasta = pysam.Fastafile(genome_fasta_file)

    # ---------------------- use shuffle to generate negative windows ---------------------- #
    positive_windows_train = subset_chroms(train_chroms, positive_windows)
    positive_windows_valid = subset_chroms(valid_chroms, positive_windows)
    positive_windows_test = subset_chroms(test_chroms, positive_windows)

    #negative_windows_train = BedTool.cat(*(epochs*[positive_windows]), postmerge=False)
    #negative_windows_train = negative_windows_train.shuffle(g=genome_sizes_file,
    negative_windows_train = positive_windows_train.shuffle(g=genome_sizes_file,
                                                            incl=genome_bed_train.fn,
                                                            excl=nonnegative_regions_bed.fn,
                                                            noOverlapping=False,
                                                            seed=np.random.randint(-214783648, 2147483647))


    negative_windows_valid = positive_windows_valid.shuffle(g=genome_sizes_file,
                                                            incl=genome_bed_valid.fn,
                                                            excl=nonnegative_regions_bed.fn,
                                                            noOverlapping=False,
                                                            seed=np.random.randint(-214783648, 2147483647))
    
    negative_windows_test = positive_windows_test.shuffle(g=genome_sizes_file,
                                                            incl=genome_bed_test.fn,
                                                            excl=nonnegative_regions_bed.fn,
                                                            noOverlapping=False,
                                                            seed=np.random.randint(-214783648, 2147483647))
    
    negative_windows = BedTool.cat(negative_windows_train, negative_windows_valid, negative_windows_test, postmerge=False)
    for negative_window in negative_windows:
        if len(negative_window.chrom) > 8:
            pdb.set_trace()
        chrom = negative_window.chrom
        start = int(negative_window.start)
        stop = int(negative_window.stop)
        med = int((start + stop) / 2)
        start = int(med - genome_window_size / 2)
        stop = int(med + genome_window_size / 2)

        window_fasta = genome_fasta.fetch(chrom, start, stop)
        if len(re.findall('[atcgn]', window_fasta.lower())) != len(window_fasta):
            continue

        continue_out = False

        # ---------------------- bigwig signal ---------------------- #
        mappability_signal = {}
        for index in range(len(bigwig_data)):
            try:
                bigwig_data[index].values(chrom,start,stop)
            except RuntimeError:
                continue_out = True
                break
            else:
                mappability_signal[index] = np.array(bigwig_data[index].values(chrom,start,stop))
                mappability_signal[index][np.isnan(mappability_signal[index])] = 0
                mappability_signal[index] = ','.join(map(str,mappability_signal[index]))
        if continue_out:
            continue

        if chrom in test_chroms:
            negative_data_test = [window_fasta, mappability_signal, target_array]
            write_single_result('neg_data_test',negative_data_test, result_filefolder)
        elif chrom in valid_chroms:
            negative_data_valid = [window_fasta, mappability_signal, target_array]
            write_single_result('neg_data_valid',negative_data_valid, result_filefolder)
        else:
            negative_data_train = [window_fasta, mappability_signal, target_array]
            write_single_result('neg_data_train',negative_data_train, result_filefolder)

    genome_fasta.close()




@click.command()
@click.option('-d', '--data-cnf', type=click.Path(exists=True))
@click.option('-m', '--model-cnf', type=click.Path(exists=True))
def main(data_cnf, model_cnf):
    yaml = YAML(typ='safe')
    data_cnf, model_cnf = yaml.load(Path(data_cnf)), yaml.load(Path(model_cnf))

    input_dir = data_cnf['input_dir']


    bw_file = data_cnf['bigwig_file']
    bigwig_data = {}
    for index, single_bw_file in enumerate(bw_file):
        bigwig_data[index] = pyBigWig.open(single_bw_file)
        

    
    genome_window_size = model_cnf['padding']['DNA_len']
    target_window_size = data_cnf['target_window_size']
    genome_window_step = data_cnf['genome_window_step']

    genome_sizes_file = data_cnf['genome_sizes_file']
    blacklist_file = data_cnf['blacklist_file']

    result_filefolder = input_dir
    valid_chroms = data_cnf['valid_chroms']
    test_chroms = data_cnf['test_chroms']

    if os.path.exists(result_filefolder + 'pos_data_train.txt') or os.path.exists(result_filefolder + 'pos_data_test.txt') or os.path.exists(result_filefolder + 'pos_data_valid.txt'):
        print(f'#-------------------------------------------------#')
        print(f'result data already exsit, delete before regenerate')
        exit()

    genome_fasta_file = data_cnf['genome_fasta_file']

    pybedtools.set_tempdir('/Users/cmf/Downloads/tmp')

    blacklist = make_blacklist(blacklist_file, genome_sizes_file, genome_window_size)
    tfs, positive_windows, y_positive, _ ,nonnegative_regions_bed= load_chip_multiTask(input_dir,genome_sizes_file, target_window_size, genome_window_size, genome_window_step, blacklist)
    
    # ---------------------- random sample the negative data to match the size of positive ---------------------- #
    #os.system("shuf -n {} data/tf_chip/negative_windows.bed > data/tf_chip/shuf_negative_windows.bed".format(100000))
    #negative_windows = BedTool("data/tf_chip/shuf_negative_windows.bed") 

    make_pos_features_multiTask(genome_sizes_file, genome_window_size, positive_windows, y_positive, valid_chroms, test_chroms, genome_fasta_file, bigwig_data, result_filefolder)
    make_neg_features_multiTask(genome_sizes_file, genome_window_size, positive_windows, nonnegative_regions_bed, valid_chroms, test_chroms, genome_fasta_file, bigwig_data, result_filefolder)

    os.system("cat {}pos_data_train.txt {}neg_data_train.txt > {}data_train.txt".format(result_filefolder,result_filefolder,result_filefolder))
    os.system("cat {}pos_data_valid.txt {}neg_data_valid.txt > {}data_valid.txt".format(result_filefolder,result_filefolder,result_filefolder))
    os.system("cat {}pos_data_test.txt {}neg_data_test.txt > {}data_test.txt".format(result_filefolder,result_filefolder,result_filefolder))


if __name__ == '__main__':
    main()