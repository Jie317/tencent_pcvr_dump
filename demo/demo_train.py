# -*- coding: utf-8 -*-
"""
Created on Sat May 13 17:51:55 2017

@author: Z.Y
"""

'''
和时间肯定有关系
不同时间段点击概率不同
应该要进行时间的截取
'''
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import FeatureHasher
from datetime import datetime

'''
数据清洗

将标签由1|0,转换为1|-1
使用FeatureHasher进行处理
'''
preproc = Pipeline([('fh',
                     FeatureHasher(n_features=2 ** 27,
                                   input_type='string',
                                   non_negative=False))])

def clean_data(train_data):
    #    train_data['day'] = train_data['clickTime'].map(lambda x :int(str(x)[0:2]))
    train_data['hour'] = train_data['clickTime'].map(lambda x: int(str(x)[2:4]))
    train_data['mins'] = train_data['clickTime'].map(lambda x: int(str(x)[4:6]))
    y = train_data['label'].values + train_data['label'].values - 1
    y = np.asarray(y).ravel()

    x = train_data.drop(['label', 'clickTime', 'conversionTime'], axis=1)
    x = np.asarray(x.astype(str))

    x = preproc.fit_transform(x)
    return x, y


'''
模型采用简单的SGDClassifier

'''
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss
from sklearn.externals import joblib

chunk_size = 10000
reader = pd.read_csv(r'D:\dataScience\pre\train.csv',
                     chunksize=chunk_size, )
# seed = 0  seed尚未定义

start = datetime.now()
i = 0
for data in reader:
    i += 1
    x, y = clean_data(data)
    cls = SGDClassifier(loss='log', n_iter=200, alpha=.0000001, penalty='l2', \
                        learning_rate='invscaling',
                        power_t=0.5,
                        eta0=4.0,
                        shuffle=True,
                        n_jobs=-1,
                        # random_state=seed
                        )
    all_classes = np.array([-1, 1])

    cls.partial_fit(x, y, classes=all_classes)
    y_pred = cls.predict_proba(x)

    LogLoss = log_loss(y, y_pred)

    print('iter:%s' % i)
    print(LogLoss)
    if i % 15 == 0:
        print(str(datetime.now() - start))

model_file = 'D:\dataScience\pre\model-avazu-sgd4.pkl'
joblib.dump(cls, model_file)

preproc_file = 'D:\dataScience\pre\model-avazu-preproc4.pkl'
joblib.dump(preproc, preproc_file)
