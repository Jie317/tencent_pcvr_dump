# -*- coding: utf-8 -*-
"""
Created on Thu May 18 11:20:50 2017

@author: Z.Y
"""

from collections import OrderedDict
import pandas as pd

d = 'D:/dataScience/pre/'


def ffm_format(row):
    # 注意每一列的数据不同之处
    str_ = '1' if int(row[0]) == 1 else '-1'
    for i, v in enumerate(row[1:]):
        # get rid of zero values 
        if v == 0:
            continue
        else:
            str_ += '\t%d:%d:%s' % (fields.index(features[i + 1].split('_')[0]) - 1, i, str(v))
    return str_


loop = True
chunkSize = 100000
# chunkSize = 10000
chunks = []
reader = pd.read_csv('%sformatted_ffm_raw_train6.csv' % d, iterator=True)
# For ffm_raw is too big to deal with it, we choicn to spead it.
i = 0
while loop:
    try:
        chunk = reader.get_chunk(chunkSize)

        fields = tuple(list(OrderedDict.fromkeys(i.split('_')[0] for i in chunk.columns)))  # including 'label'
        features = chunk.columns

        chunk = chunk.apply(ffm_format, axis=1)
        chunks.append(chunk)
        i += 1
        print('iteration %s' % i)
    except StopIteration:
        loop = False
        print("Iteration is stopped.")

formatted = pd.concat(chunks, ignore_index=True)

formatted.to_csv('%sformatted_train6.csv' % d, header=False, index=False)
