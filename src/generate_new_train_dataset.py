
# coding: utf-8

# In[3]:

import os
import pandas as pd
import numpy as np
from collections import Counter
from time import time
from imblearn.under_sampling import RandomUnderSampler


# ## Load all csv files into pandas DataFrames

# In[16]:

d = '../data/pre/' # data directory

user = pd.read_csv('%suser.csv' % d)
ad = pd.read_csv('%sad.csv' % d)
position = pd.read_csv('%sposition.csv' % d)
test = pd.read_csv('%stest.csv' % d)
train = pd.read_csv('%strain.csv' % d)
app_categories = pd.read_csv('%sapp_categories.csv' % d)
user_installedapps = pd.read_csv('%suser_installedapps.csv' % d)
user_app_actions = pd.read_csv('%suser_app_actions.csv' % d)


# ### Check out imbalance ratio from the training data

# In[28]:

s = Counter(train['label'])
print(s[1]/s[0], s)


# ## Parse the train.csv and test csv which mainly consist of three feature groups: user info, ad info and action info(click and conversion)

# In[126]:

generate_train = False # True for train, False for test
if generate_train:
    data = train
else: 
    data = test


# ### 1 User info (user, user_installedapps and user_app_actions)

# In[127]:

user_info = user.set_index('userID').ix[data['userID']].reset_index()


# ### 2 Ad info (creative and position)

# In[128]:

creative_info = ad.set_index('creativeID').ix[data['creativeID']].reset_index()
position_info = position.set_index('positionID').ix[data['positionID']].reset_index()
ad_info = pd.concat([creative_info, position_info], axis=1)


# ### 3 Action info (clickTime, conversionTime, connectionType, and telecomsOperator)

# In[129]:

action_info = train[['clickTime', 'conversionTime','connectionType', 'telecomsOperator']]


# ## Generated training dataset
# ### Drop out the id columns (or we keep them?)

# In[130]:

user_info = user_info.drop('userID', axis=1)
ad_info = ad_info.drop(['creativeID', 'positionID'], axis=1)


# ### Concatenate the three feature groups, appended by label column

# In[131]:

if generate_train:
    new_train = pd.concat([user_info, ad_info, action_info, data['label']], axis=1)
    new_train.to_csv('%snew_generated_train.csv' % d)
else:
    new_test = pd.concat([user_info, ad_info, action_info], axis=1)
    new_test.to_csv('%snew_generated_test.csv' % d)


# In[113]:




# In[ ]:



