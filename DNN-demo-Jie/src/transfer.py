'''
Three IDs: userID, creativeID, positionID
'''
import argparse
trained_model_path = '../trained_models/last_dnn.h5'
parser = argparse.ArgumentParser(
    description='DNNs on Keras to predict conversion rate.')
parser.add_argument('-r', action='store_true', 
    help='retrain feature embedding')
parser.add_argument('-ct', action='store_true',
    help='continue training last model')
parser.add_argument('-s', action='store_true',
    help='print model summary')
parser.add_argument('-m', type=str, required=True, default='no_mess',
    help='leave a message')
parser.add_argument('-et', type=str, default=None, 
    help='evaluate trained model (the last run by default))')
parser.add_argument('-nm', action='store_true', 
    help='not mask future information')

group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('-elr', action='store_true',
    help='logistic regression after embedding')
group.add_argument('-mlp', action='store_true',
    help='multilayer perceptrons')
group.add_argument('-rnn', action='store_true',
    help='recurrent networks')

g = parser.add_mutually_exclusive_group(required=False)
g.add_argument('-tdo', action='store_true',
    help='two days only (17 and 24)')
g.add_argument('-rml', action='store_true',
    help='remove day 30')
g.add_argument('-olv', action='store_true',
    help='use offline validation train and test datasets')

args = parser.parse_args()


import os
import json
import numpy as np
import pandas as pd
from time import time, strftime
from keras import backend as K
from keras.callbacks import TensorBoard, ModelCheckpoint, Callback
from keras.models import load_model, Sequential, Model
from keras.layers import Dense, Embedding, LSTM, GRU, SimpleRNN, BatchNormalization
from keras.layers import Dropout, Bidirectional, Flatten, Input, Reshape
from keras.layers.merge import Concatenate, Add, concatenate, add
from keras.initializers import Identity 
print(' >>>>>>>>> Devv stat 15777 >>>>>>>>>>>> ')

def save_preds(preds, cb=False):
    assert len(preds)==338489
    avg = np.average(preds)
    std = np.std(preds)
    p = '../%s%s_dnn_tl_result_%.4f_%.4f_%s.csv' % ('callback_' if cb else '', 
        strftime('%H%M_%m%d'), avg, std, args.m)
    with open(p, 'w') as res:
        res.write('instanceID,prob\n')
        for i,pr in enumerate(preds): res.write('%s,%.8f\n' % ((i+1), pr))
    if cb: 
        return avg, std
    else:
        print('\nTrain average: ', tr_avg)
        print('Preds average: ', avg)
        print('Preds std dev.: ', std)
        print('\nWritten to result file: ', p)

def s_c(x):
    return [x[:, i:i+1] for i in range(len(x[0]))]

def identity_reg(weight_matrix):
    shape = weight_matrix.shape
    if len(shape) != 2 or shape[0] != shape[1]:
      raise ValueError('Identity matrix initializer can only be used '
                       'for 2D square matrices.')
    return 0.01 * K.sum(K.abs(weight_matrix-np.identity(shape[0])))

class predCallback(Callback):
    def __init__(self, test_data):
        self.te = test_data

    def on_epoch_end(self, epoch, logs={}):
        predict_probas = np.ravel(self.model.predict(self.te, batch_size=40960, verbose=1))
        avg, std = save_preds(predict_probas, cb=True)
        print('\nTr avg: %.4f, avg: %.4f, std: %.4f\n'%(tr_avg, avg, std))


# ====================================================================================== #
# ====================================================================================== #
# data
features = ['positionID', 'positionType', 'creativeID', 'appID', 'adID',
            'advertiserID', 'camgaignID', 'sitesetID', 'connectionType',
            'residence', 'age', 'hometown', 'haveBaby', 'telecomsOperator',
            'gender', 'education', 'clickTime_h', 'clickTime_d', 'weekDay',
            'marriageStatus', 'appPlatform', 'clickTime_m'] #'userID'
features.reverse()

tr_df = pd.read_csv('../data/pre/new_generated_train.csv', index_col=0)
te_df = pd.read_csv('../data/pre/new_generated_test.csv', index_col=0)
va_df = tr_df.loc[tr_df['clickTime_d'] == 24]


if args.rml:
    print('--- Dropping day 30: ', len(tr_df[tr_df.clickTime_d == 30]))
    tr_df = tr_df[tr_df.clickTime_d != 30]

if args.tdo:
    print('--- 17, 24 -> 31')
    tr_df = tr_df.loc[(tr_df['clickTime_d'] == 17) | (tr_df['clickTime_d'] == 24)]

if args.olv:
    print('--- Using offline validation like: 17-23 -> 24 (val)')
    tr_df = tr_df.loc[(tr_df['clickTime_d'] >= 17) & (tr_df['clickTime_d'] <= 23)]
    if not args.nm:
        print("--- Masked ", np.sum(tr_df.loc[tr_df['conversionTime_d'] 
            >= 24, 'label'])/(np.sum(tr_df.loc[tr_df['clickTime_d'] == 23, 'label'])+1e-5))
        tr_df.loc[tr_df['conversionTime_d'] >= 24, 'label'] = 0


tr_df = tr_df[features+['label']]
va_df = va_df[features+['label']]
te_df = te_df[features]
tr = tr_df.values
va = va_df.values
te = te_df.values
np.random.shuffle(tr)
tr_x, tr_y = tr[:, :-1], tr[:, -1:]
va_x, va_y = va[:, :-1], va[:, -1:]
tr_x = s_c(tr_x)
va_x = s_c(va_x)
te = s_c(te)
tr_avg = np.average(tr_y)
max_f_cols = pd.concat([tr_df[features], te_df]).max().values+1
print(tr_df.columns.values, '\n', max_f_cols)

# ====================================================================================== #
# parameters
trained_model_path = '../trained_models/last_tl_dnn.h5'
f_model_paths = ['../trained_models/f_model_%d.h5'%i for i in range(len(features))]

checkpoint_all = ModelCheckpoint('../trained_models/tl_all_{epoch:02d}_{val_loss:.4f}.h5', 
            monitor='val_loss', verbose=1, save_best_only=True, period=1)

# ====================================================================================== #
# model
if args.et or args.ct:

    model_final = load_model(trained_model_path)
    print('\nLoaded trained model: %s' % trained_model_path)
    if args.s: model_final.summary() 

else:

    if args.r:
        f_emb_models = []
        inps = [Input(shape=(1,), name='inp_%d'%i) for i in range(len(features))]
        for i,f in enumerate(features):
            print('\nStarting feature embedding traing for feature: ', i, f)

            y = Embedding(max_f_cols[i], 16, name='emb_%d'%i)(inps[i])
            y = Flatten(name='fla_%d'%i)(y)
            y = BatchNormalization()(y)
            y = Dense(64, activation='relu', name='den_%d'%i, kernel_regularizer='l2')(y)
            y = Dense(1, activation='sigmoid', name='y_%d'%i)(y)
            f_model = Model(inps[i], y)

            print('\n--- Max feature:', max_f_cols[i])
            # f_model.summary()
            print([l.name for l in f_model.layers])

            f_model.compile('rmsprop', 'binary_crossentropy')

            checkpoint = ModelCheckpoint('../trained_models/tl_%d_{epoch:02d}_{val_loss:.4f}.h5'%i, 
                        monitor='val_loss', verbose=1, save_best_only=True, period=1)

            f_model.fit(tr_x[i], tr_y, epochs=1, validation_data=(va_x[i], va_y), 
                shuffle=True, verbose=2, batch_size=1024, callbacks=[checkpoint])

            f_model.save('../trained_models/f_emb_model_%d'%i)
            print('\n--- %d %s Evaluation on day '%(i,f), 24)
            scores = f_model.evaluate(va_x[i], va_y, batch_size=8196, verbose=1)
            print('--- %d %s Evaluation scores: '%(i,f), scores)

            for l in f_model.layers: l.trainable = False
            f_emb_models.append(f_model)
            f_model.save(f_model_paths[i])

        print('\nFinished feature embedding training')

    else:
        f_emb_models = [load_model(f_model_paths[i]) for i in range(len(features))]
        inps = [m.get_layer('inp_%d'%i).input for i,m in enumerate(f_emb_models)]
        print('\nLoaded trained feature models')


    y = [m.get_layer('den_%d'%i).output for i,m in enumerate(f_emb_models)]

    if args.rnn:
        y = concatenate(y)
        y = Reshape((len(features), 64))(y)
        y = LSTM(80, activation='tanh', return_sequences=True, kernel_regularizer='l2')(y)
        y = LSTM(32, activation='tanh')(y)
        y = Dense(32, activation='tanh', kernel_regularizer=identity_reg, kernel_initializer=Identity(1))(y)
        y = Dense(1, activation='sigmoid')(y)
    if args.mlp:
        y = concatenate(y)
        y = Dense(64, activation='relu', kernel_regularizer='l1')(y)
        y = Dropout(.1)(y)
        y = Dense(32, activation='relu')(y)
        y = Dense(1, activation='sigmoid')(y)
    if args.elr:
        y = concatenate(y)
        # y = Dropout(.5)(y)
        y = Dense(1, activation='sigmoid', kernel_regularizer='l1')(y)


    model_final = Model(inps, y)
    if args.s: model_final.summary()
    model_final.compile('rmsprop', 'binary_crossentropy', ['binary_crossentropy'])


# ====================================================================================== #
# training
if not args.et:
    print('\nStart training final model')
    model_final.fit(tr_x, tr_y, epochs=5, validation_data=(va_x, va_y), 
                shuffle=True, verbose=1,
                batch_size=4096, callbacks=[predCallback(te)])

    # save model
    model_final.save('../trained_models/%s_tl_dnn.h5' % strftime("%m%d_%H%M%S"))
    model_final.save(trained_model_path)
    print('Saved model: ', trained_model_path)


# ====================================================================================== #
# predict
print('\nPredict')
predict_probas = np.ravel(model_final.predict(te, batch_size=40960, verbose=1))
save_preds(predict_probas)