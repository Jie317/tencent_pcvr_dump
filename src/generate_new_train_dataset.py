
# coding: utf-8

# 2 parameters for this script
d = '../data/pre/' # data directory
generate_train = True # True for train, False for test

import os
import pandas as pd
import numpy as np
from collections import Counter
from time import time
from imblearn.under_sampling import RandomUnderSampler


# ## Load all csv files into pandas DataFrames

user = pd.read_csv('%suser.csv' % d)
ad = pd.read_csv('%sad.csv' % d)
position = pd.read_csv('%sposition.csv' % d)
test = pd.read_csv('%stest.csv' % d)
train = pd.read_csv('%strain.csv' % d)
app_categories = pd.read_csv('%sapp_categories.csv' % d)
user_installedapps = pd.read_csv('%suser_installedapps.csv' % d)
user_app_actions = pd.read_csv('%suser_app_actions.csv' % d)

s = Counter(train['label'])
print('Data imbalance:', s[1]/s[0], s)

def generate_new_dataset(data, train=True):
	# ### Check out imbalance ratio from the training data


	# ## Parse the train.csv and test csv which mainly consist of three feature groups: user info, ad info and action info(click and conversion)

	# ### 1 User info (user, user_installedapps and user_app_actions)
	user_info = user.set_index('userID').ix[data['userID']].reset_index()

	# ### 2 Ad info (creative and position)
	creative_info = ad.set_index('creativeID').ix[data['creativeID']].reset_index()
	position_info = position.set_index('positionID').ix[data['positionID']].reset_index()
	ad_info = pd.concat([creative_info, position_info], axis=1)

	# ### 3 Action info (clickTime, conversionTime, connectionType, and telecomsOperator)
	if not train:
	    data['conversionTime'] = np.nan # TODO: alternative?
	action_info = data[['clickTime', 'conversionTime','connectionType', 'telecomsOperator']]
	# ## Generated training dataset
	# ### Drop out the id columns (or we keep them?)

	user_info = user_info.drop('userID', axis=1)
	ad_info = ad_info.drop(['creativeID', 'positionID'], axis=1)

	# ### Concatenate the three feature groups, appended by label column
	if train:
	    new_train = pd.concat([user_info, ad_info, action_info, data['label']], axis=1)
	    new_train.to_csv('%snew_generated_train.csv' % d)
	else:
	    new_test = pd.concat([user_info, ad_info, action_info], axis=1)
	    new_test.to_csv('%snew_generated_test.csv' % d)

generate_new_dataset(train, train=True)
generate_new_dataset(test, train=False)

print('New datasets generated from raw csv files')


