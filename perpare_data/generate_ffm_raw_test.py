# -*- coding: utf-8 -*-
"""
Created on Thu May 18 11:29:47 2017

@author: Z.Y
"""

import pandas as pd

d = 'D:/dataScience/pre/'
data = pd.read_csv('%snew_generated_test.csv' % d, index_col=0)
# tr = tr[1:10000]

data['hour'] = data['clickTime'].map(lambda x: int(str(x)[2:4]))
data['mins'] = data['clickTime'].map(lambda x: int(str(x)[4:6]))

all_cat_to_one_hot = []
cat_cols = ['age',  # add age
            'gender', 'education', 'marriageStatus', 'haveBaby',
            'appPlatform', 'sitesetID', 'positionType', 'connectionType', 'telecomsOperator',
            'hometown', 'residence'  # add hometown, residence
            ]

for c in cat_cols:
    all_cat_to_one_hot.append(pd.get_dummies(data[c], prefix=c))

# conversionTime 应该去掉
# ==============================================================================
# ffm_raw = pd.concat([data[['label', 'age']]] + all_cat_to_one_hot +
#                     [data[['hometown', 'residence', 'adID', 'camgaignID', 'advertiserID', 'appID', 'clickTime',
#                            'conversionTime']]], axis=1)
# ==============================================================================

# 在test 里面label 以-1做处理

ffm_raw = pd.concat([data[['label']]] + all_cat_to_one_hot +
                    [data[['adID', 'camgaignID', 'advertiserID', 'appID',
                           # 'clickTime',  # replace cilcktime with hour and mins
                           'hour', 'mins',
                           ]]], axis=1)

ffm_raw.fillna(0, inplace=True)

ffm_raw.to_csv('%sformatted_ffm_raw_test4.csv' % d)
print('ffm_raw_test completed')


# from collections import OrderedDict
#
# fields = tuple(list(OrderedDict.fromkeys(i.split('_')[0] for i in ffm_raw.columns)))  # including 'label'
# features = ffm_raw.columns
#
#
# def ffm_format(row):
#     str_ = '1' if int(row[0]) == 1 else '-1'
#     for i, v in enumerate(row[1:]):
#         # get rid of zero values
#         if v == 0:
#             continue
#         else:
#             str_ += '\t%d:%d:%s' % (fields.index(features[i + 1].split('_')[0]) - 1, i, str(v))
#     return str_
#
#
# formatted = ffm_raw.apply(ffm_format, axis=1)
#
# formatted.to_csv('%sformatted_ffm_raw_test4.csv' % d, header=False, index=False)
