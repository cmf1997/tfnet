#!/usr/bin/env python
# -*- encoding: utf-8 -*-

'''
@File : motif_scan.py
@Time : 2023/11/22 11:37:51
@Author : Cmf
@Version : 1.0
@Desc : 
all tf motifs were downloaded from https://jaspar.elixir.no/downloads/
'''

# here put the import lib
from tfnet import all_tfs
# code

# ---------------------- cat all meme file to one ---------------------- #


# ---------------------- run fimo of meme suite ---------------------- #
# /Users/cmf/meme/bin/fimo --qv-thresh --thresh 1e-4 --o fimo_out all_tfs.meme *.fa


# ---------------------- process fimo out to a binary list indicating the ocurrency of specific tf ---------------------- #

