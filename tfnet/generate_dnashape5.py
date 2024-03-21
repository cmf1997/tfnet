#!/usr/bin/env python
# -*- encoding: utf-8 -*-

'''
@File : generate_dnashape5.py
@Time : 2024/03/21 12:59:53
@Author : Cmf
@Version : 1.0
@Desc : dnasheape5 file collect from Table S2 of article “A novel convolution attention model for predicting transcription factor binding sites by combination of sequence and shap” 
'''

# here put the import lib
import pandas as pd
import numpy as np
import pdb

# code
#dnashape5 = pd.read_csv("tfnet/dnashape5.2.csv", header=0, index_col=0)
#dnashape5_norm = pd.read_csv("tfnet/dnashape5.3.csv", header=0, index_col=0)

def seq_to_shape(seq, dnashape5):

    query_name = ['EP'   ,'HelT'  ,'MGW'   ,'ProT',  'Roll']

    seq = seq.upper()

    assert len(seq) > 4, "Sequence too short (has to be at least 5 bases)"

    # generate empty dictionary to save shape value for each position
    shape_position_value = {}

    for shape_name in query_name:
        shape_position_value[shape_name] = {}

    for index in range(0, len(seq)):

        current_position = index + 1
        current_pentamer = seq[index - 2 : index + 3]

        if (len(current_pentamer) < 5) or not all(
            base in "ACGT" for base in current_pentamer
        ):
            for shape_name in query_name:
                shape_position_value[shape_name][current_position] = None
        else:
            for shape_name in query_name:
                shape_position_value[shape_name][current_position] = dnashape5.loc[
                    current_pentamer, shape_name
                ]
    
    #pdb.set_trace()

    translation = pd.DataFrame(shape_position_value)
    translation = np.array(translation.drop([1, 2, len(seq), len(seq)-1]))
    
    #---------------- solved by discard seq contain N by data_utils of get data lazy ---------------------- #
    '''
    try:
        translation[np.isnan(translation)] = 0
    except TypeError as e:
        #pdb.set_trace()
        print("TypeError due to seq consist of all NNNN")
    '''

    translation[np.isnan(translation)] = 0
    return translation