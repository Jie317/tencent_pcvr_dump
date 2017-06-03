
# coding: utf-8

# In[1]:

d = '../data/pre/' # raw data directory

import os
import pandas as pd
import numpy as np
from collections import Counter
from time import time
from imblearn.under_sampling import RandomUnderSampler


# ## Load all csv files into pandas DataFrames
'''
user = pd.read_csv('%suser.csv' % d)
ad = pd.read_csv('%sad.csv' % d)
position = pd.read_csv('%sposition.csv' % d)
test = pd.read_csv('%stest.csv' % d)
train = pd.read_csv('%strain.csv' % d)'''
ac = pd.read_csv('%sapp_categories.csv' % d)
ui = pd.read_csv('%suser_installedapps.csv' % d)
ua = pd.read_csv('%suser_app_actions.csv' % d)


# In[2]:

tr_ori = pd.read_csv('../data/pre/new_generated_train.csv', index_col=0)
te_ori = pd.read_csv('../data/pre/new_generated_test.csv', index_col=0)


# In[3]:

ui_ids = ui.userID.unique()


# In[4]:

ua_ids = ua.userID.unique()


# In[5]:

len(ui_ids), len(ua_ids
                )


# In[6]:

inters = set(ui_ids).intersection(ua_ids)


# In[7]:

va = tr_ori.sample(frac=.1, random_state=10)
tr = tr_ori.drop(va.index, axis=0)


# In[8]:

len(va), len(tr)


# In[47]:

va_x = va.appCategory.values.reshape(-1,1)
va_y = va.label.values.reshape(-1,1)
tr_x = tr.appCategory.values.reshape(-1,1)
tr_y = tr.label.values.reshape(-1,1)


# In[54]:

tr_ = tr.groupby('appCategory').apply(lambda df: np.mean(df.label))
va_ = va.groupby('appCategory').apply(lambda df: np.mean(df.label))


# In[10]:

tr_


# In[11]:

ac_stat = tr.groupby('appCategory').apply(lambda df: len(df))


# In[57]:

va_-np.ravel(model_.predict(tr_.index))


# In[75]:

model.predict(tr_.index)


# In[15]:

ac_stat.values
total_tr = 2*max(ac_stat.values)
new_x = []
new_y = []
for cate,occ in zip(ac_stat.index, ac_stat.values):
    tmp = tr.loc[tr.appCategory==cate]
    x = tmp.appCategory.values
    y = tmp.label.values
    x = list(x)
    y = list(y)
    x = x*(int(total_tr/len(x)))
    y = y*(int(total_tr/len(y)))    
    new_x += x
    new_y += y
new_x = np.array(new_x)
new_y = np.array(new_y)


# In[16]:

new_y = new_y.reshape(-1,1)
new_x = new_x.reshape(-1,1)


# In[17]:

new_y.shape


# In[30]:

ac_stat.index


# In[14]:

tr_ac = tr.appCategory.values.reshape(-1,1)


# In[15]:

tr_y = tr.label.values.reshape(-1, 1)


# In[16]:

tr_ac_oh = pd.get_dummies(pd.concat([tr, te]).appCategory).values


# In[17]:

tr_ac_oh = tr_ac_oh[:len(tr)]
tr_ac_oh.shape


# In[21]:

from keras.models import load_model, Sequential, Model
from keras.utils import plot_model, to_categorical
from keras.callbacks import TensorBoard, ModelCheckpoint, Callback
from keras.layers import Dense, Embedding, LSTM, GRU, SimpleRNN, BatchNormalization
from keras.layers import Dropout, Bidirectional, Flatten, Input, Reshape
from keras.layers.merge import Concatenate, Add, concatenate, add


# In[62]:

# no ajust
np.random.seed(10)
i = Input(shape=(1,))
o = Embedding(np.max(tr_x)+1, 64)(i)
o = Flatten()(o)
o = Dense(64, activation='tanh')(o)
o = Dense(1, activation='sigmoid')(o)
model_ = Model(i,o)
model_.summary()
model_.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['binary_crossentropy'])
# print(model.layers[1].get_weights())
model_.fit(tr_x, tr_y,validation_data=(va_x,va_y), verbose=1, epochs=2, batch_size=128,  shuffle=True)


# In[58]:

np.random.seed(10)
i = Input(shape=(1,))
o = Embedding(np.max(new_x)+1, 64)(i)
o = Flatten()(o)
o = Dense(64, activation='tanh')(o)
o = Dense(1, activation='sigmoid')(o)
model = Model(i,o)
model.summary()
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['binary_crossentropy'])
# print(model.layers[1].get_weights())
model.fit(new_x, new_y,validation_data=(va_x,va_y), verbose=1, epochs=2, batch_size=256,  shuffle=True)


# In[37]:

va_df = va_.to_frame().reset_index()
dict_cat_prob = va_df.set_index(va_df.appCategory).to_dict()[0]
dict_cat_prob


# In[64]:

from sklearn.metrics import log_loss
ideal_loss = log_loss(va_y, np.array([dict_cat_prob[c[0]] for c in va_x]).reshape(-1,1))
model_loss = log_loss(va_y, model.predict(va_x ))
model_loss_ = log_loss(va_y, model_.predict(va_x ))
ideal_loss-model_loss , model_loss_-model_loss


# In[86]:

np.random.seed(10)
i = Input(shape=(len(tr_ac_oh[0]),))
o = Dense(64, activation='tanh')(i)
o = Dense(1, activation='sigmoid')(o)
model_oh = Model(i,o)
model_oh.summary()
model_oh.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['binary_crossentropy'])
model_oh.fit(tr_ac_oh, tr_y, validation_split=.1, verbose=1, epochs=5, batch_size=2048,  shuffle=True)


# In[62]:

model.fit(tr_ac_oh, tr_y, validation_split=.1, verbose=1, epochs=5, batch_size=256,  shuffle=True)


# In[ ]:




# In[22]:

np.random.seed(10)
i = Input(shape=(1,))
o = Embedding(np.max(new_x)+1, 16)(i)
o = Flatten()(o)
o = Dense(64, activation='tanh')(o)
o = Dense(1, activation='sigmoid')(o)
model = Model(i,o)
model.summary()
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['binary_crossentropy'])
# print(model.layers[1].get_weights())
model.fit(new_x, new_y, validation_data=(va_x,va_y), verbose=1, epochs=2, batch_size=128,  shuffle=True)

