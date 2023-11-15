from pybedtools import BedTool, Interval
import pybedtools
import numpy as np
from preprocess_chip_data import get_chip_beds
from preprocess_chip_data import get_genome_bed

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




cat GM12878_BATF.bed GM12878_BMI1.bed GM12878_CBFB.bed GM12878_SMAD1.bed GM12878_TBX21.bed > five_tf_cat.bed
bedtools sort -i five_tf_cat.bed > five_tf_cat.sort.bed
bedtools makewindows -g hg19.chrom.sizes.reduced -w 1024 -s 100 > hg19_windows.bed
bedtools intersect -a hg19_windows.bed -b five_tf_cat.sort.bed -u -F 1 -sorted > positive_window.bed


bedtools intersect -a positive_window.bed -b filter_blacklist.bed -v -wa -sorted > positive_window_filter.bed

bedtools intersect -a positive_window_filter.bed -b GM12878_BATF.bed -F 1 -c -wa > GM12878_BATF.bedgraph
bedtools intersect -a positive_window_filter.bed -b GM12878_TBX21.bed -F 1 -c -wa > GM12878_TBX21.bedgraph

awk -F '\t' '$4 > 0 {print $0}' GM12878_BATF.bedgraph | wc -l 
awk -F '\t' '$4 > 0 {print $0}' GM12878_TBX21.bedgraph | wc -l 

targets_array = []
with open('/Users/cmf/Downloads/TFNet-multi-tf/tfnet/pos_data_test.txt') as fp :
    for line in fp:
        print(line)
        chrom, start, stop, target_array = line.split()
        target_array = [float(i) for i in target_array]
        targets_array.append(target_array)
    print(targets_array[0])



window_DNA_seq = BedTool([Interval('chr1', 10, 1000)])
window_DNA_seq = window_DNA_seq.getfasta(fi='/Users/cmf/Downloads/TFNet-multi-tf/data/genome/genome.fa')
read_seq = open(window_DNA_seq.seqfn).read().split('\n')[1]


window_DNA_seq.sequence(fi='/Users/cmf/Downloads/TFNet-multi-tf/data/genome/genome.fa')
genome_fasta = pybedtools

a.getfasta(fi='/Users/cmf/Downloads/TFNet-multi-tf/data/genome/genome.fa')

fasta = pybedtools.example_filename('test.fa')
a = a.sequence(fi=fasta)


a = pybedtools.BedTool("""chr1 1 10
                       chr1 50 55""", from_string=True)
fasta = pybedtools.example_filename('test.fa')
a = a.sequence(fi=fasta)