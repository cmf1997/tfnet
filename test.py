from pybedtools import BedTool, Interval
import pybedtools
import numpy as np
from tfnet.preprocess_chip_data import get_chip_beds
from tfnet.preprocess_chip_data import get_genome_bed

#input_dir = '/Users/cmf/Downloads/TFNet-multi-tf/data/tf_chip/'
input_dir = '/Users/cmf/Downloads/project_tf_dl/tfnet/data/tf_chip/'
tfs, chip_beds, merged_chip_bed = get_chip_beds(input_dir)

genome_sizes_file = '/Users/cmf/Downloads/project_tf_dl/tfnet/data/tf_chip/hg19.chrom.sizes.reduced'
L = 1000
blacklist_file = '/Users/cmf/Downloads/project_tf_dl/tfnet/data/tf_chip/filter_blacklist.bed'
blacklist = BedTool(blacklist_file)
blacklist = blacklist.slop(g=genome_sizes_file, b=L)



chroms, chroms_sizes, genome_bed = get_genome_bed(genome_sizes_file)
merged_chip_bed = merged_chip_bed.intersect(genome_bed, u=True, sorted=True)

