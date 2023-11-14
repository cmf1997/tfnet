from pybedtools import BedTool, Interval
import pybedtools
import numpy as np

input_dir = '/Users/cmf/Downloads/TFNet-multi-tf/data/tf_chip/'
tfs, chip_beds, merged_chip_bed = get_chip_beds(input_dir)

genome_sizes_file = '/Users/cmf/Downloads/TFNet-multi-tf/data/tf_chip/hg19.chrom.sizes.reduced'
L = 1000
blacklist_file = '/Users/cmf/Downloads/TFNet-multi-tf/data/tf_chip/filter_blacklist.bed'
blacklist = BedTool(blacklist_file)
blacklist = blacklist.slop(g=genome_sizes_file, b=L)






def get_genome_bed(genome_sizes_file):
    genome_sizes_info = np.loadtxt(genome_sizes_file, dtype=str)
    chroms = list(genome_sizes_info[:,0])
    chroms_sizes = list(genome_sizes_info[:,1].astype(int))
    genome_bed = []
    for chrom, chrom_size in zip(chroms, chroms_sizes):
        genome_bed.append(Interval(chrom, 0, chrom_size))
    genome_bed = BedTool(genome_bed)
    return chroms, chroms_sizes, genome_bed


def get_genome_windows(genome_sizes_file, genome_window_size, genome_window_step):
    genome_windows = BedTool().window_maker(g=genome_sizes_file, w=genome_window_size,s=genome_window_step)
    return genome_windows

bedtools makewindows -g /Users/cmf/Downloads/TFNet-multi-tf/data/tf_chip/hg19.chrom.sizes.reduced -w 1024 -s 100 > genome_windows.bed
bedtools intersect -a genome_windows.bed -b GM12878_BATF.bed -wa -wb > annotated_windows.bed

list.extend()