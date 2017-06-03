
# coding: utf-8

d = '../data/pre/' # raw data directory
print('dev stat 22666633')
import os
import pandas as pd
import numpy as np
from time import time
from sklearn.metrics import log_loss


from keras.models import load_model, Sequential, Model
from keras.utils import plot_model, to_categorical
from keras.callbacks import TensorBoard, ModelCheckpoint, Callback
from keras.layers import Dense, Embedding, LSTM, GRU, SimpleRNN, BatchNormalization
from keras.layers import Dropout, Bidirectional, Flatten, Input, Reshape
from keras.layers.merge import Concatenate, Add, concatenate, add

ac = pd.read_csv('%sapp_categories.csv' % d)
ui = pd.read_csv('%suser_installedapps.csv' % d)
ua = pd.read_csv('%suser_app_actions.csv' % d)


tr_ori = pd.read_csv('../data/pre/new_generated_train.csv', index_col=0)
te_ori = pd.read_csv('../data/pre/new_generated_test.csv', index_col=0)



va = tr_ori.sample(frac=.1, random_state=3)
tr = tr_ori.drop(va.index, axis=0)



va_y = va.label.values.reshape(-1,1)
tr_y = tr.label.values.reshape(-1,1)


# features = [ 'positionType', 'connectionType', 'age', 'haveBaby', 'telecomsOperator',
#             'gender', 'education', 'clickTime_h', 'clickTime_d', 'weekDay',
#             'marriageStatus', 'appPlatform', 'clickTime_m']

features = ['appCategory']



for f in features:
    print('\n\n','='*20,f)
    va_x = va[f].values.reshape(-1,1)
    tr_x = tr[f].values.reshape(-1,1)

    va_ = va.groupby(f).apply(lambda df: np.mean(df.label))
    tr_ = tr.groupby(f).apply(lambda df: np.mean(df.label))


    tr_stat = tr.groupby(f).apply(lambda df: len(df))

    print(tr_stat.values)

    total_tr = 2*max(tr_stat.values)
    new_x = []
    new_y = []
    for cate,occ in zip(tr_stat.index, tr_stat.values):
        tmp = tr.loc[tr[f]==cate]
        x = tmp[f].values
        y = tmp.label.values
        x = list(x)
        y = list(y)
        x = x*(int(total_tr/len(x)))
        y = y*(int(total_tr/len(y))) 
        print(len(x))   
        new_x += x
        new_y += y
    new_x = np.array(new_x).reshape(-1,1)
    new_y = np.array(new_y).reshape(-1,1)
    print('Length of new x:', len(new_x))

    # no ajust
    np.random.seed(323)
    i = Input(shape=(1,))
    o = Embedding(np.max(tr_x)+1, 64)(i)
    o = Flatten()(o)
    o = Dense(64, activation='tanh')(o)
    o = Dense(1, activation='sigmoid')(o)
    model_ = Model(i,o)
    model_.summary()
    model_.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['binary_crossentropy'])
    model_.fit(tr_x, tr_y,validation_data=(va_x,va_y), verbose=1, epochs=2, batch_size=1024,  shuffle=True)

    print('\nUnbalanced model predict:\n', model_.predict(va_.index))

    # balanced
    np.random.seed(323)
    i = Input(shape=(1,))
    o = Embedding(np.max(new_x)+1, 64)(i)
    o = Flatten()(o)
    o = Dense(64, activation='tanh')(o)
    o = Dense(1, activation='sigmoid')(o)
    model = Model(i,o)
    model.summary()
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['binary_crossentropy'])
    model.fit(new_x, new_y,validation_data=(va_x,va_y), verbose=1, epochs=2, batch_size=4096,  shuffle=True)

    print('\nBalanced model predict:\n', model.predict(va_.index))

    va_df = va_.to_frame().reset_index()
    dict_prob = va_df.set_index(va_df[f]).to_dict()[0]

    ideal_loss = log_loss(va_y, np.array([dict_prob[c[0]] for c in va_x]).reshape(-1,1))
    model_loss = log_loss(va_y, model.predict(va_x ))
    model_loss_ = log_loss(va_y, model_.predict(va_x))
    print('ideal-model', 'ideal-model_', 'model_-model')
    print('%.8f, %.8f, %.8f'%(ideal_loss-model_loss , model_loss_-model_loss,ideal_loss-model_loss_))

    print(va_-np.ravel(model_.predict(va_.index)), '\nBalanced model\n',va_-np.ravel(model.predict(va_.index)) )


    model.save('balanced_tl_%s_%.6f.h5'%(f,ideal_loss-model_loss))
