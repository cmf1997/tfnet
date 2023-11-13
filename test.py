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

bedtools grep -i -header -f avai_chr.txt -fi blacklist.bed > filter_black.bed




