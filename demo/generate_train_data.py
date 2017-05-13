# -*- coding: utf-8 -*-
"""
Created on Sat May 13 23:09:40 2017

@author: Z.Y
"""

import os
import pandas as pd
import numpy as np
from collections import Counter
from time import time

d = 'D:/dataScience/pre/'

user = pd.read_csv('%suser.csv' % d)
ad = pd.read_csv('%sad.csv' % d)
position = pd.read_csv('%sposition.csv' % d)
test = pd.read_csv('%stest.csv' % d)
train = pd.read_csv('%strain.csv' % d)
app_categories = pd.read_csv('%sapp_categories.csv' % d)

user_info = user.set_index('userID').ix[train['userID']].reset_index()
creative_info = ad.set_index('creativeID').ix[train['creativeID']].reset_index()
position_info = position.set_index('positionID').ix[train['positionID']].reset_index()
ad_info = pd.concat([creative_info, position_info], axis=1)

action_info = train[['clickTime', 'conversionTime', 'connectionType', 'telecomsOperator']]
user_info = user_info.drop('userID', axis=1)
ad_info = ad_info.drop(['creativeID', 'positionID'], axis=1)

new_train = pd.concat([user_info, ad_info, action_info, train['label']], axis=1)

new_train.to_csv('%snew_generated_train.csv' % d)
