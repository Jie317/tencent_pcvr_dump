print('\n >>>>>>>> Dev new stat2892\n')
import argparse
trained_model_path = '../trained_models/last_dnn.h5'
parser = argparse.ArgumentParser(
    description='DNNs on Keras to predict conversion rate.')
parser.add_argument('-et', type=str, default=None, 
    help='evaluate trained model (the last run by default))')
parser.add_argument('-ns', action='store_true', 
    help='don\'t save model in the end')
parser.add_argument('-of', action='store_true', 
    help='use only one feature')
parser.add_argument('-nm', action='store_true', 
    help='not mask future information')
parser.add_argument('-e', type=int, default=5,
    help='epochs')
parser.add_argument('-f', type=int, default=9,
    help='the f most important independent features (<=22)')
parser.add_argument('-v', type=int, default=1,
    help='verbose')
parser.add_argument('-vd', type=int, default=24,
    help='which day for validation')
parser.add_argument('-s', action='store_true',
    help='print model summary')
parser.add_argument('-nfe', action='store_true',
    help='not use fine-grained embedding layers')
parser.add_argument('-ct', action='store_true',
    help='continue training last model')
parser.add_argument('-m', type=str, required=True,
    help='leave a message')


g = parser.add_mutually_exclusive_group(required=False)
g.add_argument('-tdo', action='store_true',
    help='two days only (17 and 24)')
g.add_argument('-rml', action='store_true',
    help='remove day 30')
g.add_argument('-olv', action='store_true',
    help='use offline validation train and test datasets')


group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('-elr', action='store_true',
    help='logistic regression after embedding')
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
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.layers import Dense, Embedding, LSTM, GRU, SimpleRNN, BatchNormalization
from keras.layers import Dropout, Bidirectional, Flatten, Input
from keras.layers.merge import Concatenate, Add, concatenate, add

def s_c(x):
    return [x[:, i:i+1] for i in range(len(x[0]))]

def save_preds(preds):
    assert len(preds)==338489
    print('\nTrain average: ', np.average(tr_y))
    avg = np.average(preds)
    std = np.std(preds)
    print('Preds average: ', avg)
    print('Preds std dev.: ', std)

    p = '../%s_dnn_sub_%.4f_%.4f_%s.csv' % (strftime('%H%M_%m%d'), avg, std, args.m)

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
features = ['positionID', 'positionType', 'creativeID', 'appID', 'adID',
            'advertiserID', 'camgaignID', 'sitesetID', 'connectionType',
            'residence', 'age', 'hometown', 'haveBaby', 'telecomsOperator',
            'gender', 'education', 'clickTime_h', 'clickTime_d', 'weekDay',
            'marriageStatus', 'appPlatform', 'clickTime_m'][:args.f]


if args.of: 
    features = features[-1: ]
features.append('label')

tr_df = pd.read_csv('../data/pre/new_generated_train.csv', index_col=0)
te_df = pd.read_csv('../data/pre/new_generated_test.csv', index_col=0)
va_df = tr_df.loc[tr_df['clickTime_d'] == args.vd]


if args.rml:
    print('--- Dropping day 30: ', len(tr_df[tr_df.clickTime_d == 30]))
    tr_df = tr_df[tr_df.clickTime_d != 30]

if args.tdo:
    print('--- 17, 24 -> 31')
    tr_df = tr_df.loc[(tr_df['clickTime_d'] == 17) | (tr_df['clickTime_d'] == 24)]

if args.olv:
    print('--- Using offline validation like: 16-23 -> 24 (val)')
    tr_df = tr_df.loc[(tr_df['clickTime_d'] >= 17) & (tr_df['clickTime_d'] <= 23)]
    if not args.nm:
        print("--- Masked ", np.sum(tr_df.loc[tr_df['conversionTime_d'] 
            >= args.vd, 'label'])/(np.sum(tr_df.loc[tr_df['clickTime_d'] == args.vd-1, 'label'])+1e-5))
        tr_df.loc[tr_df['conversionTime_d'] >= args.vd, 'label'] = 0

tr_df = tr_df[features]
va_df = va_df[features]
te_df = te_df[features[:-1]]

tr = tr_df.values
va = va_df.values
te = te_df.values
np.random.shuffle(tr)
input_length = len(tr[0])-1
max_feature = max(tr.max(), va.max() if args.olv else 0, te.max())+1
max_f_cols = tr_df.max().values[:-1]+1
print('--- Train cols: ', tr_df.columns)
print('--- Validation day:', args.vd, len(va_df))
print('--- Max feature:', max_feature)
print('--- Max column features:', max_f_cols)
print('--- Selected most important features:', args.f)

tr_x, tr_y = tr[:, :-1], tr[:, -1:]
va_x, va_y = va[:, :-1], va[:, -1:]
if not args.nfe:
    va_x = s_c(va_x)


# ====================== 2 Build model and training ==================== #

# 0 hyperparameters
optimizer = 'rmsprop' # rmsprop, adam
loss = 'binary_crossentropy'
metrics = ['binary_crossentropy'] # can't be empty in this script


batch_size = 4096
workers = 1

# 0.2 parameter instantiations
tbCallBack = TensorBoard(log_dir='../meta/tbGraph/', histogram_freq=1,
             write_graph=True, write_images=True)
checkpoint = ModelCheckpoint('../trained_models/{epoch:02d}-{val_loss:.4f}.h5', monitor='val_loss', 
             verbose=1, save_best_only=True, period=1)


# 2 fix random seed for reproducibility
np.random.seed(7)

# 3 build model or load last trained model
start = time()
if args.et and args.et != '0':
    trained_model_path = args.et

if args.et or args.ct:
    model = load_model(trained_model_path)
    print('\nLoaded model: %s' % trained_model_path)
    if args.s: model.summary() 

else:
    if args.rnn: # recurrent networks
        model = Sequential()
        model.add(Embedding(max_feature, 16, input_length = input_length))
        model.add(LSTM(128, activation='tanh', return_sequences=True))   
        model.add(LSTM(32, activation='tanh'))
        model.add(Dense(1, activation='sigmoid')) 

    if args.mlp: # multilayer perceptrons
        if not args.nfe:
            print('Using fine-grained embedding layers')
            cols_in = []
            cols_out = []
            f = lambda x: int((math.log10(x)+3)*4)
            print([f(fe) for fe in max_f_cols])
            for fe in max_f_cols:
                col_in = Input(shape=(1,))
                col_out = Embedding(int(fe), f(fe))(col_in)
                col_out = Flatten()(col_out)
                col_out = BatchNormalization()(col_out) 

                cols_in.append(col_in)
                cols_out.append(col_out)

            cols_concatenated = concatenate(cols_out)
            y = Dense(1024, activation='relu', 
                      kernel_regularizer='l1')(cols_concatenated)
            y = Dropout(.3)(y)
            y = Dense(1024, activation='relu')(y)
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

    if args.elr: # logistic regression after embedding
        model = Sequential()
        model.add(Embedding(max_feature, 16, input_length = input_length))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))

        model.add(Dense(1, activation='sigmoid')) 

        
if not args.et:
    # 4 compile model
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    if args.s: model.summary() 
    print('\n', strftime('%c'))
  #  plot_model(model, to_file='../meta/model.png', show_shapes=True)

    # 5 fit the model (training)
    if not args.nfe:
        tr_x = s_c(tr_x)
    model.fit(tr_x, tr_y, epochs=args.e, validation_data=(va_x, va_y), 
        shuffle=True, verbose=args.v,
        batch_size=batch_size, callbacks=[checkpoint, tbCallBack])

    if not args.ns: 
        model.save('../trained_models/%s_dnn.h5' % strftime("%m%d_%H%M%S"))
        model.save(trained_model_path)
        print('Saved model')


print('Runtime:', str(datetime.timedelta(seconds=int(time()-start))))

# va_y_real = va_df_real.label.values
# print('Mean and sum va_y:     \t', np.mean(va_y), np.sum(va_y), 
# print('Mean and sum va_y_real:\t', np.mean(va_y_real), np.sum(va_y_real))
#       '\tDiff: ', (np.sum(va_y_real)-np.sum(va_y))/np.sum(va_y), '\n')
# scores_real = model.evaluate(va_x, va_y_real, batch_size=8196, verbose=args.v)
if args.olv:
    print('Evaluation on day ', args.vd)
    scores = model.evaluate(va_x, va_y, batch_size=8196, verbose=args.v)
    # print('\nEvaluation real scores: ', scores_real)
    print('\nEvaluation scores: ', scores)
    # print('\nDifference: ', np.array(scores_real)-np.array(scores))

# 7 calculate predictions
print('\nPrediction')
if not args.nfe:
    te = s_c(te)

predict_probas = np.ravel(model.predict(te, batch_size=4096*2, verbose=args.v))
save_preds(predict_probas)
