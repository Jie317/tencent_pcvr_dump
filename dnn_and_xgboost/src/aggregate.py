
# coding: utf-8

# In[1]:

d = '../data/pre/' # raw data directory

import os
import pandas as pd
import numpy as np
from collections import Counter
from time import time
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


# In[2]:

aC = pd.read_csv('../data/pre/app_categories.csv')

user_installedapps = pd.read_csv('%suser_installedapps.csv' % d)
ui_cates = pd.merge(user_installedapps, aC, on='appID', how='left')
ui_dum_cats = pd.get_dummies(ui_cates['appCategory'], prefix='appCategory', dummy_na=True)
ui_cates = pd.concat([ui_cates, ui_dum_cats], axis=1)


# In[3]:

ui = ui_cates.groupby('userID').apply(lambda df: list(np.sum(df[[col for col in df.columns if 'appCategory_' in col]].values, axis=0))).reset_index()

ui.columns = ['userID', 'insAppCates']

print('\nDev info\n',ui.head(), len(ui))


user_app_actions = pd.read_csv('%suser_app_actions.csv' % d)

uact_cate = pd.merge(user_app_actions, aC, on='appID', how='left')
uact_dum_cats = pd.get_dummies(uact_cate['appCategory'], prefix='appActCategory', dummy_na=True)
uact_cate = pd.concat([uact_cate, uact_dum_cats], axis=1)

print('\n\nDev info...', ...)


# In[6]:

uact_cate['installTime_d'] = uact_cate.installTime.map(lambda x: int(str(x)[0:2]))
uact_cate = uact_cate.groupby('userID').apply(lambda df: list(np.sum(df[[col for col in df.columns if 'appActCategory_' in col]].values, axis=0))).reset_index()
uact_cate.columns = ['userID', 'actApps']


# In[9]:

print('\n\n',uact_cate.head(), len(uact_cate))


# In[10]:

te_df = pd.read_csv('../data/pre/new_generated_test.csv', index_col=0)
te_df = pd.merge(te_df, ui, on='userID', how='left')
te_df = pd.merge(te_df, uact_cate, on='userID', how='left')
te_df.actApps.fillna(str([0] *28), inplace=True)
te_df.insAppCates.fillna(str([0] *28), inplace=True)


# In[11]:

tr_df = pd.read_csv('../data/pre/new_generated_train.csv', index_col=0)
tr_df = pd.merge(tr_df, ui, on='userID', how='left')
tr_df = pd.merge(tr_df, uact_cate, on='userID', how='left')
tr_df.actApps.fillna(str([0] *28), inplace=True)
tr_df.insAppCates.fillna(str([0] *28), inplace=True)


va_df = tr_df.loc[tr_df['clickTime_d'] == 24]


tr_ui_ = pd.DataFrame([eval(str(r)) for r in tr_df.insAppCates.values])
te_ui_ = pd.DataFrame([eval(str(r)) for r in te_df.insAppCates.values])
va_ui_ = pd.DataFrame([eval(str(r)) for r in va_df.insAppCates.values])


# In[18]:

tr_ua_ = pd.DataFrame([eval(str(r)) for r in tr_df.actApps.values])
te_ua_ = pd.DataFrame([eval(str(r)) for r in te_df.actApps.values])
va_ua_ = pd.DataFrame([eval(str(r)) for r in va_df.actApps.values])


# In[19]:

tr_ui_.to_csv('../data/pre/new_tr_ui.csv', header=None, index=None)
te_ui_.to_csv('../data/pre/new_te_ui.csv', header=None, index=None)
va_ui_.to_csv('../data/pre/new_va_ui.csv', header=None, index=None)

tr_ua_.to_csv('../data/pre/new_tr_ua.csv', header=None, index=None)
te_ua_.to_csv('../data/pre/new_te_ua.csv', header=None, index=None)
va_ua_.to_csv('../data/pre/new_va_ua.csv', header=None, index=None)


print('Finished')



