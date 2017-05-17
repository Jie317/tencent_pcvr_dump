
# coding: utf-8

# In[1]:

# 2 parameters for this script
d = '../data/pre/' # data directory
generate_train = False # True for train, False for test

import os
import pandas as pd
import numpy as np
from collections import Counter
from time import time
from imblearn.under_sampling import RandomUnderSampler


# ## Load all csv files into pandas DataFrames

# In[2]:

user = pd.read_csv('%suser.csv' % d)
ad = pd.read_csv('%sad.csv' % d)
position = pd.read_csv('%sposition.csv' % d)
test = pd.read_csv('%stest.csv' % d)
train = pd.read_csv('%strain.csv' % d)
app_categories = pd.read_csv('%sapp_categories.csv' % d)
user_installedapps = pd.read_csv('%suser_installedapps.csv' % d)
user_app_actions = pd.read_csv('%suser_app_actions.csv' % d)


# ### Check out imbalance ratio from the training data

# In[3]:

s = Counter(train['label'])
print(s[1]/s[0], s)


# ## Parse the train.csv and test csv which mainly consist of three feature groups: user info, ad info and action info(click and conversion)

# In[16]:

if generate_train:
    data = train
else: 
    data = test


# ### 1 User info (user, user_installedapps and user_app_actions)

# In[17]:

user_info = user.set_index('userID').ix[data['userID']].reset_index()


# ### 2 Ad info (creative and position)

# In[18]:

creative_info = ad.set_index('creativeID').ix[data['creativeID']].reset_index()
position_info = position.set_index('positionID').ix[data['positionID']].reset_index()
ad_info = pd.concat([creative_info, position_info], axis=1)


# ### 3 Action info (clickTime, conversionTime, connectionType, and telecomsOperator)

# In[19]:

if not generate_train:
    data['conversionTime'] = np.nan # TODO: alternative?
action_info = data[['clickTime', 'conversionTime','connectionType', 'telecomsOperator']]


# ## Generated training dataset
# ### Drop out the id columns (or we keep them?)

# In[20]:

user_info = user_info.drop('userID', axis=1)
ad_info = ad_info.drop(['creativeID', 'positionID'], axis=1)


# ### Concatenate the three feature groups, appended by label column

# In[21]:

if generate_train:
    new_train = pd.concat([user_info, ad_info, action_info, data['label']], axis=1)
    new_train.to_csv('%snew_generated_train.csv' % d)
else:
    new_test = pd.concat([user_info, ad_info, action_info], axis=1)
    new_test.to_csv('%snew_generated_test.csv' % d)


# In[22]:

new_test[:3].values


# In[23]:

# new_train[:3].values


# In[ ]:



