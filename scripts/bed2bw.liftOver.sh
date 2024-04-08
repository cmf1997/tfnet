#!/bin/bash

# $1 for Genome Version (grch38 or grch37)

bed_files=`ls *[0-9].bed`
liftOver=/lustre/home/acct-medzy/medzy-cai/software/liftOver
chain=/lustre/home/acct-medzy/medzy-cai/software/hg19ToHg38.over.chain.gz
chrom_size=/lustre/home/acct-medzy/medzy-cai/project/project_tf_dl/tfnet_52/data/scATAC/fragments/split/hg38.chrom.sizes


for bed_file in $bed_files
do
	Cluster=`basename $bed_file .bed`
	echo Processing $Cluster

    if [ $1 == "grch38" ]; then
        echo Genome Version grch38, skipping liftOver!
        scatac_fragment_tools bigwig -n -s 1 -c $chrom_size -i $Cluster.bed -o $Cluster.bw
        echo Finishing $Cluster bigWig
    elif [ $1 == "grch37"]; then
        echo Genome Version grch37, conducting liftOver!
        # liftover from hg19 to grch38
        $liftOver $bed_file $chain $Cluster.liftOver.bed $Cluster.unmapped.bed
        echo Finishing $Cluster liftOver
        # generate bigwig from bed
        scatac_fragment_tools bigwig -n -s 1 -c $chrom_size -i $Cluster.liftOver.bed -o $Cluster.liftOver.bw
        echo Finishing $Cluster bigWig
    else
        echo unregonized Genome Version, provide grch38 or grch37

done