
# coding: utf-8

# In[1]:

# 2 parameters for this script
d = '../data/pre/' # data directory
format_train = True # True for train, False for test

# output
ffm_train_path = '%sffm_train' % d
ffm_test_path = '%sffm_test' % d

import os
import pandas as pd
import numpy as np
from collections import Counter
from time import time

new_train = '%snew_generated_train.csv' % d
new_test = '%snew_generated_test.csv' % d


# In[2]:




# In[4]:

if format_train:
    data = pd.read_csv(new_train, index_col=0)
else: 
    data = pd.read_csv(new_test, index_col=0)
    data['label'] = -1


# In[5]:

cat_cols = ['gender', 'education', 'marriageStatus', 'haveBaby',
        'appPlatform', 'sitesetID', 'positionType', 'connectionType', 'telecomsOperator']
all_cat_to_one_hot = []

for c in cat_cols:
    all_cat_to_one_hot.append(pd.get_dummies(data[c], prefix=c))



# In[6]:

ffm_raw = pd.concat([data[['label', 'age']]]+all_cat_to_one_hot+
                   [data[['hometown', 'residence', 'adID', 'camgaignID', 'advertiserID', 'appID','clickTime',
        'conversionTime']]], axis=1)

ffm_raw.fillna(0, inplace=True) # TODO: confirm replacing nan with zero


# In[7]:

from collections import OrderedDict
fields = tuple(list(OrderedDict.fromkeys(i.split('_')[0] for i in ffm_raw.columns))) # including 'label'
features = ffm_raw.columns


# In[ ]:

def ffm_format(row):
    str_ = '1' if int(row[0])==1 else '-1'
    for i,v in enumerate(row[1:]):
        str_ += '\t%d:%d:%s' % (fields.index(features[i+1].split('_')[0])-1, i, str(v))
    return str_
    
formatted = ffm_raw.apply(ffm_format, axis=1)
formatted.to_csv(ffm_train_path if format_train else ffm_test_path, header=False, index=False)


print('Finished ffm input formatting')
        

