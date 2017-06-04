
# coding: utf-8

# In[1]:

import pandas as pd
tr_df = pd.read_csv('../data/pre/new_generated_train.csv', index_col=0)
te_df = pd.read_csv('../data/pre/new_generated_test.csv', index_col=0)
aC = pd.read_csv('../data/pre/app_categories.csv')


# In[3]:

al = pd.concat([tr_df.appCategory, te_df.appCategory, aC.appCategory], ignore_index=True)


# In[4]:

adAppCate = pd.DataFrame({'appCategory': al.values})
adAppCate = pd.get_dummies(adAppCate.appCategory)


# In[5]:

tr = adAppCate[:len(tr_df)]


# In[6]:

te = adAppCate[len(tr_df):len(tr_df)+len(te_df)]


# In[ ]:




# In[ ]:

tr.to_csv('../data/pre/new_adAppCate_tr.csv')


# In[ ]:

te.to_csv('../data/pre/new_adAppCate_te.csv')


# In[9]:

tr_ = pd.read_csv('../data/pre/new_adAppCate_te.csv', index_col=0)


# In[ ]:




# In[11]:

print(tr_.head())


# In[ ]:



