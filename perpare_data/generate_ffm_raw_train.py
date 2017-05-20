# -*- coding: utf-8 -*-
"""
Created on Thu May 18 08:55:59 2017

@author: Z.Y
"""
import pandas as pd

d = 'D:/dataScience/pre/'
data = pd.read_csv('%snew_generated_train.csv' % d, index_col=0)
# tr = tr[1:10000]

data['hour'] = data['clickTime'].map(lambda x: int(str(x)[2:4]))
data['mins'] = data['clickTime'].map(lambda x: int(str(x)[4:6]))

all_cat_to_one_hot = []
cat_cols = ['age',  # add age
            'gender', 'education', 'marriageStatus', 'haveBaby',
            'appPlatform', 'sitesetID', 'positionType', 'connectionType', 'telecomsOperator',
            'hometown',   # add hometown, residence
            ]
# 注意当以residence 作为特征项目时，会出现特征不对的情况

for c in cat_cols:
    all_cat_to_one_hot.append(pd.get_dummies(data[c], prefix=c))

# conversionTime 应该去掉
# ==============================================================================
# ffm_raw = pd.concat([data[['label', 'age']]] + all_cat_to_one_hot +
#                     [data[['hometown', 'residence', 'adID', 'camgaignID', 'advertiserID', 'appID', 'clickTime',
#                            'conversionTime']]], axis=1)
# ==============================================================================

ffm_raw = pd.concat([data[['label']]] + all_cat_to_one_hot +
                    [data[['adID', 'camgaignID', 'advertiserID', 'appID','residence',
                           # 'clickTime',  # replace cilcktime with hour and mins
                           'hour', 'mins',
                           ]]], axis=1)

ffm_raw.fillna(0, inplace=True)

ffm_raw.to_csv('%sformatted_ffm_raw_train6.csv' % d,index = False )
print('ffm_raw_train completed')
