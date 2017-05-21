
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


def format_ffm(df, outp):
    cat_cols = ['gender', 'education', 'marriageStatus', 'haveBaby',
            'appPlatform', 'sitesetID', 'positionType', 'connectionType', 'telecomsOperator']
    all_cat_to_one_hot = []

    for c in cat_cols:
        all_cat_to_one_hot.append(pd.get_dummies(df[c], prefix=c))


    ffm_raw = pd.concat([df[['label', 'age']]]+all_cat_to_one_hot+
                       [df[['hometown', 'residence', 'adID', 'camgaignID', 'advertiserID', 'appID','clickTime_d', 'clickTime_h', 'clickTime_m',]]], axis=1)

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
    formatted.to_csv(outp, header=False, index=False)



tr = pd.read_csv('%snew_generated_train.csv' % d, index_col=0)
format_ffm(tr, ffm_train_path)

te = pd.read_csv('%snew_generated_test.csv' % d, index_col=0)
te['label'] = -1
format_ffm(te, ffm_test_path)


print('Finished ffm input formatting')
        

