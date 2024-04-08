#!/bin/bash

# $1 for da_peaks.csv output by R script
# $2 for Genome Version (grch38 or grch37)

filename=`basename $1 .csv`
fc=0.5
qvalue=0.05

liftOver=/lustre/home/acct-medzy/medzy-cai/software/liftOver
chain=/lustre/home/acct-medzy/medzy-cai/software/hg19ToHg38.over.chain.gz
grch38=/lustre/home/acct-medzy/medzy-cai/project/project_tf_dl/tfnet/data/genome/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta
all_meme_file=/lustre/home/acct-medzy/medzy-cai/project/project_tf_dl/tfnet_52/data/matched.motif.meme
bedtools=/lustre/home/acct-medzy/medzy-cai/software/bedtools
FIMO=/lustre/home/acct-medzy/medzy-cai/software/meme/bin/fimo

echo Filterring da peaks by foldchange: $fc, and q-value: $qvalue
awk -v FS=',' -v fc=$fc -v qvalue=$qvalue '{if ($3 > fc && $2 < qvalue)print $1}' $1 > $filename.filter.csv
cut -d ',' -f 1 $filename.filter.csv | sed '1d' | sed 's/"//g' | awk -F "-" -v OFS='\t' '{print $1,$2,$3}' > $filename.filter.bed

#liftOver
if [ $2 == "grch38" ]; then
    echo Genome Version grch38, skipping liftOver!
    echo bed2fa
    $bedtools getfasta -fo $filename.filter.fa -fi $grch38 -bed $filename.filter.bed
    echo FIMO
    $FIMO --o FIMO_$filename $all_meme_file $filename.filter.fa

elif [ $2 == "grch37"]; then
    echo Genome Version grch37, conducting liftOver!
    $liftOver $filename.filter.bed $chain $filename.filter.liftOver.bed $$filename.filter.unmapped.bed
    echo bed2fa
    $bedtools getfasta -fo $filename.filter.liftOver.fa -fi $grch38 -bed $filename.filter.liftOver.bed
    echo FIMO
    $FIMO --o FIMO_$filename $all_meme_file $filename.filter.liftOver.fa
else
    echo unregonized Genome Version, provide grch38 or grch37