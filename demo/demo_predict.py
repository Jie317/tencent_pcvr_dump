# -*- coding: utf-8 -*-
"""
Created on Sat May 13 19:32:52 2017

@author: Z.Y
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import FeatureHasher
from datetime import datetime
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss
from sklearn.externals import joblib

'''
导入模型参数
'''
model_file = 'D:\dataScience\pre\model-avazu-sgd6.pkl'
cls = joblib.load(model_file)

preproc_file = 'D:\dataScience\pre\model-avazu-preproc6.pkl'
preproc = joblib.load(preproc_file)

'''
数据清洗
'''


def clean_data(train_data):
    #    train_data['day'] = train_data['clickTime'].map(lambda x :int(str(x)[0:2]))
    train_data['hour'] = train_data['clickTime'].map(lambda x: int(str(x)[2:4]))
    train_data['mins'] = train_data['clickTime'].map(lambda x: int(str(x)[4:6]))
    x = train_data.drop(['label', 'instanceID', 'clickTime'], axis=1)
    x = np.asarray(x.astype(str))
    x = preproc.fit_transform(x)
    return x


chunk_size = 10000
reader = pd.read_csv(r'D:\dataScience\pre\new_generated_test.csv',
                     chunksize=chunk_size, )

# seed = 0  seed尚未定义
i = 0
with open(r'D:\dataScience\pre\submission6.csv', 'a') as outfile:
    outfile.write('instanceID, prob\n')
    for data in reader:
        i += 1
        instanceID = data['instanceID'].values
        x = clean_data(data)
        prob = cls.predict_proba(x)[:, 1]
        dfjo = pd.DataFrame(dict(instanceID=instanceID, prob=prob), columns=['instanceID', 'prob'])
        dfjo.to_csv(outfile, header=None, index_label=None, index=False)
        print(i)
