
# coding: utf-8

d = '../data/pre/' # raw data directory

import os
import pandas as pd
import numpy as np
from collections import Counter
from time import time
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


# ## Load all csv files into pandas DataFrames
'''user ad position app_categories'''




aC = pd.read_csv('../data/pre/app_categories.csv')

user_installedapps = pd.read_csv('%suser_installedapps.csv' % d)
ui_cates = pd.merge(user_installedapps, aC, on='appID', how='left')
ui_dum_cats = pd.get_dummies(ui_cates['appCategory'], prefix='appCategory', dummy_na=True)
ui_cates = pd.concat([ui_cates, ui_dum_cats], axis=1)
print('\n\n',ui_cates.head(), len(ui_cates))


ui = ui_cates.groupby('userID').apply(lambda df: np.sum(df[[col for col in df.columns if 'appCategory_' in col]].values, axis=0)).reset_index()

ui.columns = ['userID', 'insAppCates']

print('\nDev info\n',ui.head(), len(ui))





user_app_actions = pd.read_csv('%suser_app_actions.csv' % d)

uact_cate = pd.merge(user_app_actions, aC, on='appID', how='left')
uact_dum_cats = pd.get_dummies(uact_cate['appCategory'], prefix='appActCategory', dummy_na=True)
uact_cate = pd.concat([uact_cate, uact_dum_cats], axis=1)

print('\n\nDev info...', ...)
uact_cate['installTime_d'] = uact_cate.installTime.map(lambda x: int(str(x)[0:2]))
uact_cate = uact_cate.groupby('userID').apply(lambda df: np.sum(df[[col for col in df.columns if 'appActCategory_' in col]].values, axis=0)).reset_index()
uact_cate.columns = ['userID', 'actApps']

print('\n\n',uact_cate.head(), len(uact_cate))




te_df = pd.read_csv('../data/pre/new_generated_test.csv', index_col=0)
te_df = pd.merge(te_df, ui, on='userID', how='left')
te_df = pd.merge(te_df, uact_cate, on='userID', how='left')
te_df[['actApps', 'insAppCates']].fillna(str(np.zeros(28)), inplace=True)
te_df.to_csv('../data/pre/new_with_lists_test.csv', index=False)


tr_df = pd.read_csv('../data/pre/new_generated_train.csv', index_col=0)
tr_df = pd.merge(tr_df, ui, on='userID', how='left')
tr_df = pd.merge(tr_df, uact_cate, on='userID', how='left')
tr_df[['actApps', 'insAppCates']].fillna(str(np.zeros(28)), inplace=True)
tr_df.to_csv('../data/pre/new_with_lists_train.csv', index=False)




