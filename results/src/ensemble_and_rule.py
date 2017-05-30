
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np


# In[7]:

ps = ['../callback_1754_0530_dnn_tl_result_0.0212_0.0262_mlp.csv'
     ,'../2025_0527_dnn_tl_result_0.0229_0.0309_no_mess.csv']
print(ps)


# In[26]:

probs = []
for p in ps:
    d = pd.read_csv(p)
    probs.append(d.proba.values)
pr_np = np.vstack(probs)


# In[27]:

pr_np = np.mean(pr_np, axis=0)


# In[30]:

preds=pr_np

df = pd.DataFrame({'instanceID': range(1, len(preds)+1), 'proba': preds})
df.to_csv('../ensembled.csv', index=False)


# In[25]:

pr_np


# In[31]:

df


# In[ ]:



