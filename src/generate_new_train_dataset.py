
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


# ## Parse the train.csv which mainly consists of three feature groups: user info, ad info and action info(click and conversion)
# ### 1 User info (user, user_installedapps and user_app_actions)

# In[105]:

user_info = user.set_index('userID').ix[train['userID']].reset_index()


# ### 2 Ad info (creative and position)

# In[106]:

creative_info = ad.set_index('creativeID').ix[train['creativeID']].reset_index()


# In[107]:

position_info = position.set_index('positionID').ix[train['positionID']].reset_index()


# In[108]:

ad_info = pd.concat([creative_info, position_info], axis=1)


# ### 3 Action info (clickTime, conversionTime, connectionType, and telecomsOperator)

# In[109]:

action_info = train[['clickTime', 'conversionTime','connectionType', 'telecomsOperator']]


# ## Generated training dataset
# ### Drop out the id columns

# In[110]:

user_info = user_info.drop('userID', axis=1)
ad_info = ad_info.drop(['creativeID', 'positionID'], axis=1)


# ### Concatenate the three feature groups, appended by label column

# In[112]:

new_train = pd.concat([user_info, ad_info, action_info, train['label']], axis=1)


# In[113]:

new_train.to_csv('%snew_generated_train.csv' % d)


# In[ ]:



