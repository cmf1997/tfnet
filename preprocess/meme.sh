#!/bin/bash

set -e
cd ../data
meme_files=`cat all_tf_name.txt | cut -f 1`
rm all_tfs.meme
touch all_tfs.meme
for meme_file in $meme_files:
do
    cat tf_motif/$meme_file.meme >> all_tfs.meme
done