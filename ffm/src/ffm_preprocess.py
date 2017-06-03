
# coding: utf-8


# parameters for this script
d = '../data/pre/' # data directory

# output
ffm_train_path = '../ffm_train'
ffm_test_path = '../ffm_test' 

import os
import pandas as pd
import numpy as np
from collections import Counter
from time import time


def format_ffm(df):
    cat_cols = ['gender', 'education', 'marriageStatus', 'haveBaby', 'appPlatform', 
                'sitesetID', 'positionType', 'connectionType', 'telecomsOperator', 'weekDay', 'clickTime_h' ]
    all_cat_to_one_hot = []

    for c in cat_cols:
        all_cat_to_one_hot.append(pd.get_dummies(df[c], prefix=c))


    ffm_raw = pd.concat([df[['label', 'age']]]+all_cat_to_one_hot+
                       [df[['hometown', 'residence', 'adID', 'camgaignID', 'advertiserID', 'appID', 'clickTime_m',]]], axis=1)

    ffm_raw.fillna(-1, inplace=True  ) # TODO: confirm replacing nan with zero


    print(ffm_raw.columns)

    from collections import OrderedDict
    fields = tuple(list(OrderedDict.fromkeys(i.split('_')[0] for i in ffm_raw.columns))) # including 'label'
    features = ffm_raw.columns


    def raw_format(row):
        str_ = '1' if int(row[0])==1 else '-1'
        for i,v in enumerate(row[1:]):
            str_ += '\t%d:%d:%s' % (fields.index(features[i+1].split('_')[0])-1, i, str(v))
        return str_
        
    formatted = ffm_raw.apply(raw_format, axis=1)


    return formatted


tr = pd.read_csv('%snew_generated_train.csv' % d, index_col=0)

te = pd.read_csv('%snew_generated_test.csv' % d, index_col=0)
te['label'] = -1

tr_idx = len(tr.index)

all_data = pd.concat([tr, te]) # to avoid cols conflictions between tr and te

all_formatted = format_ffm(all_data)

tr = all_formatted[: tr_idx, :]
te = all_formatted[tr_idx:, :]

assert len(tr.index) == tr_idx

tr.to_csv(ffm_train_path, header=False, index=False)
te.to_csv(ffm_test_path, header=False, index=False)

print('Finished ffm input formatting')
        

