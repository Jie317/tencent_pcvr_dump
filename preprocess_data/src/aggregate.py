
# coding: utf-8

d = '../data/pre/' # raw data directory

import os
import pandas as pd
import numpy as np
from collections import Counter
from time import time
from imblearn.under_sampling import RandomUnderSampler


# ## Load all csv files into pandas DataFrames
'''user ad position app_categories'''
tr_df = pd.read_csv('../data/pre/new_generated_train.csv', index_col=0)
te_df = pd.read_csv('../data/pre/new_generated_test.csv', index_col=0)





user_installedapps = pd.read_csv('%suser_installedapps.csv' % d)
ui = user_installedapps.groupby('userID').apply(lambda df: tuple(df.appID.values)).reset_index()
ui.columns = ['userID', 'insApps']




user_app_actions = pd.read_csv('%suser_app_actions.csv' % d)
aC = pd.read_csv('../data/pre/app_categories.csv')

uact_cate = pd.merge(user_app_actions, aC, on='appID', how='left')
uact_cate['installTime_d'] = uact_cate.installTime.map(lambda x: int(str(x)[0:2]))
uact_cate = uact_cate.groupby('userID').apply(lambda df: tuple(df[['appID', 'appCategory', 'installTime_d']].values)).reset_index()
uact_cate.columns = ['userID', 'actApps']




def aggregate(data, tr=False):
	
	data = pd.merge(data, ui, on='userID', how='left')
	data = pd.merge(data, uact_cate, on='userID', how='left')

	print(data.head())

	data.to_csv('../data/pre/new_with_lists_%s.csv'%'train' if tr else 'test', index=False)	

aggregate(tr_df, True)
aggregate(te_df, False)

