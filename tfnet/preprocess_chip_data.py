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
import pdb

#genome_sizes_file = '/Users/cmf/Downloads/TFNet-multi-tf/data/tf_chip/hg19.autoX.chrom.sizes'
genome_sizes_file = '/Users/cmf/Downloads/TFNet-multi-tf/data/tf_chip/hg19.chrom.sizes.reduced'
#genome_fasta_file = 'resources/hg19.fa'
blacklist_file = '/Users/cmf/Downloads/TFNet-multi-tf/data/tf_chip/blacklist.bed.gz'
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
    else:
        merged_chip_bed = chip_beds[0]
    return tfs, chip_beds, merged_chip_bed


def load_chip_multiTask(input_dir):
    tfs, chip_beds, merged_chip_bed = get_chip_beds(input_dir)
    # ---------------------- Removing peaks outside of X chromosome and autosomes ---------------------- #
    chroms, chroms_sizes, genome_bed = get_genome_bed()
    merged_chip_bed = merged_chip_bed.intersect(genome_bed, u=True, sorted=True)

    genome_windows = BedTool().window_maker(g=genome_sizes_file, w=genome_window_size,
                                            s=genome_window_step)

    # ---------------------- Extracting windows that overlap at least one ChIP interval ---------------------- #
    positive_windows = genome_windows.intersect(merged_chip_bed, u=True, f=1.0*(genome_window_size/2+1)/genome_window_size, sorted=True)

    # Exclude all windows that overlap a blacklisted region
    blacklist = make_blacklist()
    
    # ---------------------- Removing windows that overlap a blacklisted region ---------------------- #
    positive_windows = positive_windows.intersect(blacklist, wa=True, v=True, sorted=True)

    num_positive_windows = positive_windows.count()
    # ---------------------- Generate targets ---------------------- #
    y_positive = parmap.map(intersect_count, chip_beds, positive_windows.fn)
    y_positive = np.array(y_positive, dtype=bool).T
    #print 'Positive matrix sparsity', (~y_positive).sum()*1.0/np.prod(y_positive.shape)
    merged_chip_slop_bed = merged_chip_bed.slop(g=genome_sizes_file, b=genome_window_size)
    # Later we want to gather negative windows from the genome that do not overlap
    # with a blacklisted or ChIP region
    nonnegative_regions_bed = merged_chip_slop_bed.cat(blacklist)
    return tfs, positive_windows, y_positive, nonnegative_regions_bed

def make_features_multiTask(positive_windows, y_positive, nonnegative_regions_bed, 
                            bigwig_files, bigwig_names, genome, epochs, valid_chroms, test_chroms):
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
    
    import pdb
    print 'Splitting positive windows into training, validation, and testing sets'
    for positive_window, target_array in itertools.izip(positive_windows, y_positive):
        if len(positive_window.chrom) > 8:
            pdb.set_trace()
        chrom = positive_window.chrom
        start = int(positive_window.start)
        stop = int(positive_window.stop)
        if chrom in test_chroms:
            positive_windows_test.append(positive_window)
            positive_data_test.append((chrom, start, stop, shift_size, bigwig_files, [], target_array))
        elif chrom in valid_chroms:
            positive_windows_valid.append(positive_window)
            positive_data_valid.append((chrom, start, stop, shift_size, bigwig_files, [], target_array))
        else:
            positive_windows_train.append(positive_window)
            positive_data_train.append((chrom, start, stop, shift_size, bigwig_files, [], target_array))
    
    positive_windows_train = BedTool(positive_windows_train)
    positive_windows_valid = BedTool(positive_windows_valid)
    positive_windows_test = BedTool(positive_windows_test)

    import pdb
    print 'Getting negative training examples'
    negative_windows_train = BedTool.cat(*(epochs*[positive_windows]), postmerge=False)
    #negative_windows_train = BedTool.cat(*(10*[positive_windows]), postmerge=False)
    #pdb.set_trace()
    negative_windows_train = negative_windows_train.shuffle(g=genome_sizes_file,
                                                            incl=genome_bed_train.fn,
                                                            excl=nonnegative_regions_bed.fn,
                                                            noOverlapping=False,
                                                            seed=np.random.randint(-214783648, 2147483647))
                                                            #seed=np.random.randint(-21478364, 21474836))
    print 'Getting negative validation examples'
    negative_windows_valid = positive_windows_valid.shuffle(g=genome_sizes_file,
                                                            incl=genome_bed_valid.fn,
                                                            excl=nonnegative_regions_bed.fn,
                                                            noOverlapping=False,
                                                            seed=np.random.randint(-214783648, 2147483647))
                                                            #seed=np.random.randint(-21478364, 21474836))
    print 'Getting negative testing examples'
    negative_windows_test = positive_windows_test.shuffle(g=genome_sizes_file,
                                                            incl=genome_bed_test.fn,
                                                            excl=nonnegative_regions_bed.fn,
                                                            noOverlapping=False,
                                                            seed=np.random.randint(-214783648, 2147483647))
                                                            #seed=np.random.randint(-21478364, 21474836))

    # Train
    print 'Extracting data from negative training BEDs'
    negative_targets = np.zeros(y_positive.shape[1])
    negative_data_train = [(window.chrom, window.start, window.stop, shift_size, bigwig_files, [], negative_targets)
                           for window in negative_windows_train]

    # Validation
    print 'Extracting data from negative validation BEDs'
    negative_data_valid = [(window.chrom, window.start, window.stop, shift_size, bigwig_files, [], negative_targets)
                           for window in negative_windows_valid]
    
    # Test
    print 'Extracting data from negative testing BEDs'
    negative_data_test = [(window.chrom, window.start, window.stop, shift_size, bigwig_files, [], negative_targets)
                           for window in negative_windows_test]

    num_positive_train_windows = len(positive_data_train)
    
    data_valid = negative_data_valid + positive_data_valid
    data_test = negative_data_test + positive_data_test

    print 'Shuffling training data'
    data_train = []
    for i in xrange(epochs):
        epoch_data = []
        epoch_data.extend(positive_data_train)
        epoch_data.extend(negative_data_train[i*num_positive_train_windows:(i+1)*num_positive_train_windows])
        np.random.shuffle(epoch_data)
        data_train.extend(epoch_data)

    print 'Generating data iterators'
    bigwig_rc_order = get_bigwig_rc_order(bigwig_names)
    datagen_train = DataIterator(data_train, genome, batch_size, L, bigwig_rc_order)
    datagen_valid = DataIterator(data_valid, genome, batch_size, L, bigwig_rc_order)
    datagen_test = DataIterator(data_test, genome, batch_size, L, bigwig_rc_order)

    print len(datagen_train), 'training samples'
    print len(datagen_valid), 'validation samples'
    print len(datagen_test), 'test samples'
    return datagen_train, datagen_valid, datagen_test, data_valid,data_test

def main():
    yaml = YAML(typ='safe')
    #data_cnf, model_cnf = yaml.load(Path(data_cnf)), yaml.load(Path(model_cnf))

    pybedtools.set_tempdir('/Users/cmf/Downloads/TFNet-multi-tf/data/tmp')
    input_dir = '/Users/cmf/Downloads/TFNet-multi-tf/data/tf_chip/'
    tfs, positive_windows, y_positive, nonnegative_regions_bed = load_chip_multiTask(input_dir)
    pdb.set_trace()



if __name__ == '__main__':
    main()