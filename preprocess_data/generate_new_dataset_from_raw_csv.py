d = '../data/pre/' # raw data directory

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

def generate_new_dataset(data, train=True):
	# ### Check out imbalance ratio from the training data
	if train:
		s = Counter(data['label'])
		print('Data imbalance: ', s[1]/s[0], s)

	# ## Parse the train.csv and test csv which mainly consist of three feature groups: user info, ad info and action info(click and conversion)

	data = pd.merge(data, ad, on='creativeID', how='left')
	data = pd.merge(data, user, on='userID', how='left')
	data = pd.merge(data, position, on='positionID', how='left')
	
	if not train:
		data['conversionTime'] = '000000'
	data['conversionTime'].fillna('000000', inplace=True) # TODO: alternative?

	data['clickTime_d'] = data['clickTime'].map(lambda x: int(str(x)[0:2]))
	data['weekDay'] = data['clickTime_d'].map(lambda x: (x%7)+1)
	data['clickTime_h'] = data['clickTime'].map(lambda x: int(str(x)[2:4]))
	data['clickTime_m'] = data['clickTime'].map(lambda x: int(str(x)[4:6]))
	data['conversionTime_d'] = data['conversionTime'].map(lambda x: int(str(x)[0:2]))

	# ## Generated training dataset
	# Drop out the id columns (or we keep them?)
	# user_info = user_info.drop('userID', axis=1)
	# ad_info = ad_info.drop(['creativeID', 'positionID'], axis=1)

	# ### Concatenate the three feature groups, appended by label column
	data = data.drop(['conversionTime', 'clickTime'], axis=1)

	if train:
	    data.to_csv('%snew_generated_train.csv' % d)
	else:
	    data.to_csv('%snew_generated_test.csv' % d)

generate_new_dataset(train, train=True)
generate_new_dataset(test, train=False)

print('New datasets generated from raw csv files')


