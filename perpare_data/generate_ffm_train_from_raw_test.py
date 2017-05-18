# -*- coding: utf-8 -*-
"""
Created on Thu May 18 11:34:51 2017

@author: Z.Y
"""
import pandas as pd
from collections import OrderedDict

d = 'D:/dataScience/pre/'

ffm_raw = pd.read_csv('%sformatted_ffm_raw_test4.csv' % d )


fields = tuple(list(OrderedDict.fromkeys(i.split('_')[0] for i in ffm_raw.columns)))  # including 'label'
features = ffm_raw.columns

def ffm_format(row):
    str_ = '1' if int(row[0]) == 1 else '-1'
    for i, v in enumerate(row[1:]):
        # get rid of zero values 
        if v == 0:
            continue
        else :
            str_ += '\t%d:%d:%s' % (fields.index(features[i + 1].split('_')[0]) - 1, i, str(v))
    return str_

formatted = ffm_raw.apply(ffm_format, axis=1)

formatted.to_csv('%sformatted_test4.csv' % d, header=False, index=False)