print('\n >>>>>>>> Dev new stat2892\n')
import argparse
trained_model_path = '../trained_models/last_dnn.h5'
parser = argparse.ArgumentParser(
    description='DNNs on Keras to predict conversion rate.')
parser.add_argument('-et', type=str, default=None, 
    help='evaluate trained model (the last run by default))')
parser.add_argument('-ns', action='store_true', 
    help='don\'t save model in the end')
parser.add_argument('-e', type=int, default=5,
    help='epochs')
parser.add_argument('-v', type=int, default=0,
    help='verbose')
parser.add_argument('-s', action='store_true',
    help='print model summary')
parser.add_argument('-tdo', action='store_true',
    help='two days only (17 and 24)')
parser.add_argument('-olv', action='store_true',
    help='use offline validation train and test datasets')
parser.add_argument('-ct', action='store_true',
    help='continue training last model')

group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('-lr', action='store_true',
    help='logistic regression')
group.add_argument('-mlp', action='store_true',
    help='multilayer perceptrons')
group.add_argument('-rnn', action='store_true',
    help='recurrent networks')

args = parser.parse_args()

import os
import shutil
import json
import datetime
import math
import numpy as np
import pandas as pd
from time import time, strftime
from collections import Counter
from keras import backend as K
from keras.models import load_model, Sequential, Model
from keras.utils import plot_model, to_categorical
from keras.callbacks import TensorBoard
from keras.layers import Dense, Embedding, LSTM, GRU, SimpleRNN, BatchNormalization
from keras.layers import Dropout, Bidirectional, Flatten, Input
from keras.layers.merge import Concatenate, Add, concatenate, add

def save_preds(preds, p):
    assert len(preds)==338489
    print('Train average: ', np.average(tr_y))
    print('Preds average: ', np.average(preds))
    print('Preds std dev.: ', np.std(preds))
    with open(p, 'w') as res:
        res.write('instanceID,prob\n')
        for i,pr in enumerate(preds): res.write('%s,%.8f\n' % ((i+1), pr))
    print('\nWritten to result file: ', p)



'''
       ['userID', 'age', 'gender', 'education', 'marriageStatus', 'haveBaby',
        'hometown', 'residence', 'creativeID', 'adID', 'camgaignID',
        'advertiserID', 'appID', 'appPlatform', 'positionID', 'sitesetID',
        'positionType', 'weekDay', 'clickTime_d', 'clickTime_h', 'clickTime_m',
        'connectionType', 'telecomsOperator', 'conversionTime_d', 'label']
'''    
# ========================= 1 Data preparation ========================= #


tr_df = pd.read_csv('../data/pre/new_generated_train.csv', index_col=0)
te_df = pd.read_csv('../data/pre/new_generated_test.csv', index_col=0)


# tr_df = tr_df.drop(['userID', 'creativeID', 'positionID'], axis=1)
# te_df = te_df.drop(['userID', 'creativeID', 'positionID'], axis=1)

if args.tdo:
    print('17, 24 -> 31')
    tr_df = tr_df.loc[(tr_df['clickTime_d'] == 17) | (tr_df['clickTime_d'] == 24)]

if args.olv:
    print('Using offline validation: 16-23 -> 24 (val)')
    va_df = tr_df.loc[(tr_df['clickTime_d'] == 24)]
    va_df
    tr_df = tr_df.loc[(tr_df['clickTime_d'] > 16) & (tr_df['clickTime_d'] < 23)]




tr_df = tr_df.drop('conversionTime_d', axis=1)
te_df = te_df.drop('conversionTime_d', axis=1)
tr = tr_df.values
te = te_df.values
input_length = len(tr[0])-1
max_feature = tr.max()+1
max_f_cols = tr_df.max().values[:-1]
print('Train cols: ', tr_df.columns)
print('Max feature:', max_feature)

tr_x, tr_y = tr[:, :-1], tr[:, -1:]


# ====================== 2 Build model and training ==================== #

# 0 hyperparameters
optimizer = 'rmsprop' # rmsprop, adam
loss = 'binary_crossentropy'
metrics = ['binary_crossentropy'] # can't be empty in this script


batch_size = 4096
workers = 1
fined_embedding = True

# 0.2 parameter instantiations
tbCallBack = TensorBoard(log_dir='../meta/tbGraph/', histogram_freq=1,
             write_graph=True, write_images=True)


# 2 fix random seed for reproducibility
np.random.seed(7)

# 3 build model or load last trained model
start = time()
if args.et and args.et != '0':
    trained_model_path = args.et

if args.et or args.ct:
    model = load_model(trained_model_path)
    print('Loaded model: %s' % trained_model_path)
    if args.s: model.summary() 

else:
    if args.rnn: # recurrent networks
        model = Sequential()
        model.add(Embedding(max_feature, 16, input_length = input_length))
        model.add(LSTM(128, activation='tanh', return_sequences=True))   
        model.add(LSTM(32, activation='tanh'))
        model.add(Dense(1, activation='sigmoid')) 

    if args.mlp: # multilayer perceptrons
        if fined_embedding:
            print('Using fine-grained embedding layers')
            cols_in = []
            cols_out = []
            f = lambda x: int((math.log10(x)+1)*3)
            print([f(fe) for fe in max_f_cols])
            for fe in max_f_cols:
                col_in = Input(shape=(1,))
                col_out = Embedding(fe, f(fe))(col_in)
                col_out = Flatten()(col_out)
                col_out = BatchNormalization()(col_out) 

                cols_in.append(col_in)
                cols_out.append(col_out)

            cols_concatenated = concatenate(cols_out)
            y = Dense(1024, activation='relu', 
                      kernel_regularizer='l1')(cols_concatenated)
            # y = Dropout(.3)(y)
            y = Dense(512, activation='relu')(y)
            y = Dense(1, activation='sigmoid')(y)  
            model = Model(cols_in, y)  

        else:
            model = Sequential()
            model.add(Embedding(max_feature, 16, input_length = input_length))        
            model.add(Flatten())
            model.add(Dense(1024, activation='relu'))
            model.add(Dense(1024, activation='relu'))
            model.add(Dense(512, activation='relu'))
            model.add(Dense(1, activation='sigmoid')) 

    if args.lr: # logistic regression
        model = Sequential()
        model.add(Embedding(max_feature, 16, input_length = input_length))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid')) 

        
if not args.et:
    # 4 compile model
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    if args.s: model.summary() 
    print(strftime('%c'))
  #  plot_model(model, to_file='../meta/model.png', show_shapes=True)

    # 5 fit the model (training)
    if fined_embedding:
        tr_cols = [tr_x[:, i:i+1] for i in range(len(tr_x[0]))]
        model.fit(tr_cols, tr_y, epochs=args.e, shuffle=True, verbose=args.v,
                  batch_size=batch_size, callbacks=[tbCallBack])
    else:
        model.fit(tr_x, tr_y, epochs=args.e, shuffle=True, verbose=args.v,
                  batch_size=batch_size, callbacks=[tbCallBack])

    if not args.ns: 
        model.save('../trained_models/%s_dnn.h5' % strftime("%m%d_%H%M%S"))
        model.save(trained_model_path)
        print('Saved model')


print('Runtime:', str(datetime.timedelta(seconds=int(time()-start))))
# 7 calculate predictions
print('\nPrediction')
if fined_embedding:
    te = [te[:, i:i+1] for i in range(len(te[0]))]

predict_probas = np.ravel(model.predict(te, batch_size=4096, verbose=args.v))
p = '../%s_dnn_sub.csv' % (strftime('%H%M_%m%d'), )
save_preds(predict_probas, p=p)
