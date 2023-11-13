#!/usr/bin/env python
# -*- encoding: utf-8 -*-

'''
@File : preprocess_chip_data.py
@Time : 2023/11/13 15:20:52
@Author : Cmf
@Version : 1.0
@Desc : preprocess chip-seq data, input as a datafolder, output as a interval and multiple binary labels
'''

# here put the import lib

# code
from ruamel.yaml import YAML
from pybedtools import BedTool, Interval
import pybedtools
import numpy as np
import parmap
import itertools
import pdb

genome_sizes_file = '/Users/cmf/Downloads/project_tf_dl/tfnet/data/tf_chip/hg19.chrom.sizes.reduced'
#genome_fasta_file = 'resources/hg19.fa'
blacklist_file = '/Users/cmf/Downloads/project_tf_dl/tfnet/data/tf_chip/filter_blacklist.bed'
genome_window_size = 1024
genome_window_step = 100
L = 1024


def get_genome_bed():
    genome_sizes_info = np.loadtxt(genome_sizes_file, dtype=str)
    chroms = list(genome_sizes_info[:,0])
    chroms_sizes = list(genome_sizes_info[:,1].astype(int))
    genome_bed = []
    for chrom, chrom_size in zip(chroms, chroms_sizes):
        genome_bed.append(Interval(chrom, 0, chrom_size))
    genome_bed = BedTool(genome_bed)
    genome_bed = genome_bed.sort()
    return chroms, chroms_sizes, genome_bed


def make_blacklist():
    genome_sizes_info = np.loadtxt(genome_sizes_file, dtype=str)
    avai_chr = list(genome_sizes_info[:,0])
    blacklist = BedTool(blacklist_file)
    blacklist = blacklist.slop(g=genome_sizes_file, b=L)
    # Add ends of the chromosomes to the blacklist
    
    chroms = list(genome_sizes_info[:,0])
    chroms_sizes = list(genome_sizes_info[:,1].astype(int))
    blacklist2 = []
    for chrom, size in zip(chroms, chroms_sizes):
        blacklist2.append(Interval(chrom, 0, L))
        blacklist2.append(Interval(chrom, size - L, size))
    blacklist2 = BedTool(blacklist2)
    blacklist = blacklist.cat(blacklist2)
    blacklist = blacklist.sort()

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
    chip_bedgraph = windows.intersect(chip_bed, wa=True, c=True, f=1.0*(genome_window_size/2+1)/genome_window_size, sorted=True)
    bed_counts = [i.count for i in chip_bedgraph]
    return bed_counts

def load_chip_multiTask(input_dir):
    tfs, chip_beds, merged_chip_bed = get_chip_beds(input_dir)
    # ---------------------- Removing peaks outside of X chromosome and autosomes ---------------------- #
    chroms, chroms_sizes, genome_bed = get_genome_bed()
    merged_chip_bed = merged_chip_bed.intersect(genome_bed, u=True, sorted=True)

    genome_windows = BedTool().window_maker(g=genome_sizes_file, w=genome_window_size,
                                            s=genome_window_step)
    
    genome_windows = genome_windows.sort()

    # ---------------------- Extracting windows that overlap at least one ChIP interval ---------------------- #
    positive_windows = genome_windows.intersect(merged_chip_bed, u=True, f=1.0*(genome_window_size/2+1)/genome_window_size, sorted=True)

    # Exclude all windows that overlap a blacklisted region
    blacklist = make_blacklist()
    
    # ---------------------- Removing windows that overlap a blacklisted region ---------------------- #
    positive_windows = positive_windows.intersect(blacklist, wa=True, v=True, sorted=True)

    # ---------------------- Generate targets ---------------------- #
    y_positive = parmap.map(intersect_count, chip_beds, positive_windows.fn)
    y_positive = np.array(y_positive, dtype=bool).T
    #print 'Positive matrix sparsity', (~y_positive).sum()*1.0/np.prod(y_positive.shape)
    merged_chip_slop_bed = merged_chip_bed.slop(g=genome_sizes_file, b=genome_window_size)
    # Later we want to gather negative windows from the genome that do not overlap
    # with a blacklisted or ChIP region
    nonnegative_regions_bed = merged_chip_slop_bed.cat(blacklist)
    pdb.set_trace()
    return tfs, positive_windows, y_positive, nonnegative_regions_bed

def subset_chroms(chroms, bed):
    result = bed.filter(chroms_filter, chroms).saveas()
    return BedTool(result.fn)

def make_features_multiTask(positive_windows, y_positive, nonnegative_regions_bed, bigwig_files, bigwig_names, genome, epochs, valid_chroms, test_chroms):
    chroms, chroms_sizes, genome_bed = get_genome_bed()
    train_chroms = chroms
    for chrom in valid_chroms + test_chroms:
        train_chroms.remove(chrom)
    genome_bed_train, genome_bed_valid, genome_bed_test = \
        [subset_chroms(chroms_set, genome_bed) for chroms_set in
         (train_chroms, valid_chroms, test_chroms)]

    positive_windows_train = []
    positive_windows_valid = []
    positive_windows_test = []
    positive_data_train = []
    positive_data_valid = []
    positive_data_test = []
    
    # ---------------- Splitting positive windows into training, validation, and testing sets ----------------# 
    for positive_window, target_array in itertools.izip(positive_windows, y_positive):
        if len(positive_window.chrom) > 8:
            pdb.set_trace()
        chrom = positive_window.chrom
        start = int(positive_window.start)
        stop = int(positive_window.stop)
        if chrom in test_chroms:
            positive_windows_test.append(positive_window)
            positive_data_test.append((chrom, start, stop, [], target_array))
        elif chrom in valid_chroms:
            positive_windows_valid.append(positive_window)
            positive_data_valid.append((chrom, start, stop, [], target_array))
        else:
            positive_windows_train.append(positive_window)
            positive_data_train.append((chrom, start, stop, [], target_array))
    
    positive_windows_train = BedTool(positive_windows_train)
    positive_windows_valid = BedTool(positive_windows_valid)
    positive_windows_test = BedTool(positive_windows_test)

    pdb.set_trace()

    #negative_windows_train = BedTool.cat(*(epochs*[positive_windows]), postmerge=False)
    #negative_windows_train = BedTool.cat(*(10*[positive_windows]), postmerge=False)
    #pdb.set_trace()

    # Train
#    negative_targets = np.zeros(y_positive.shape[1])
#    negative_data_train = [(window.chrom, window.start, window.stop, shift_size, bigwig_files, [], negative_targets)
#                           for window in negative_windows_train]

    # Validation
#    negative_data_valid = [(window.chrom, window.start, window.stop, shift_size, bigwig_files, [], negative_targets)
#                           for window in negative_windows_valid]
    
    # Test
#    negative_data_test = [(window.chrom, window.start, window.stop, shift_size, bigwig_files, [], negative_targets)
#                           for window in negative_windows_test]

#    num_positive_train_windows = len(positive_data_train)
    
#    data_valid = negative_data_valid + positive_data_valid
#    data_test = negative_data_test + positive_data_test

#    data_train = []
#    for i in range(epochs):
#        epoch_data = []
#        epoch_data.extend(positive_data_train)
#        epoch_data.extend(negative_data_train[i*num_positive_train_windows:(i+1)*num_positive_train_windows])
#        np.random.shuffle(epoch_data)
#        data_train.extend(epoch_data)

#    bigwig_rc_order = get_bigwig_rc_order(bigwig_names)
#    datagen_train = DataIterator(data_train, genome, batch_size, L, bigwig_rc_order)
#    datagen_valid = DataIterator(data_valid, genome, batch_size, L, bigwig_rc_order)
#    datagen_test = DataIterator(data_test, genome, batch_size, L, bigwig_rc_order)

#    return datagen_train, datagen_valid, datagen_test, data_valid,data_test

def main():
    yaml = YAML(typ='safe')
    #data_cnf, model_cnf = yaml.load(Path(data_cnf)), yaml.load(Path(model_cnf))

    pybedtools.set_tempdir('/Users/cmf/Downloads/project_tf_dl/tfnet/data/tf_chip/tmp')
    input_dir = '/Users/cmf/Downloads/project_tf_dl/tfnet/data/tf_chip/'
    tfs, positive_windows, y_positive, nonnegative_regions_bed = load_chip_multiTask(input_dir)
    make_features_multiTask()
    pdb.set_trace()



if __name__ == '__main__':
    main()