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
parser.add_argument('-f', type=int, default=24,
    help='the f most important independent features (<=24)')
parser.add_argument('-v', type=int, default=1,
    help='verbose')
parser.add_argument('-b', type=int, default=4096,
    help='batch size')
parser.add_argument('-mt', type=int, default=0,
    help='model type')
parser.add_argument('-va-seed', type=int, default=62,
    help='numpy random seed number to split tr and val')
parser.add_argument('-va', action='store_true', 
    help='split validation from train')
parser.add_argument('-s', action='store_true',
    help='print model summary')
parser.add_argument('-nfe', action='store_true',
    help='not use fine-grained embedding layers')
parser.add_argument('-ct', action='store_true',
    help='continue training last model')
parser.add_argument('-mess', type=str, default='no_mess',
    help='leave a message')
parser.add_argument('-m', type=str, default='mlp', required=True,
                    help='model name - lr | mlp | mlp_fe | elr | rf | xgb')


g = parser.add_mutually_exclusive_group(required=False)
g.add_argument('-tdo', action='store_true',
    help='two days only (17 and 24)')
g.add_argument('-rml', action='store_true',
    help='remove day 30')
g.add_argument('-fra', action='store_true',
    help='take fraction of data')
g.add_argument('-olv', action='store_true',
    help='use offline validation train and test datasets')


args = parser.parse_args()
print(args)
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
from keras.callbacks import TensorBoard, ModelCheckpoint, Callback
from keras.layers import Dense, Embedding, LSTM, GRU, SimpleRNN, BatchNormalization
from keras.layers import Dropout, Bidirectional, Flatten, Input, Reshape
from keras.layers.merge import Concatenate, Add, concatenate, add

def s_c(x):
    return [x[:, i:i+1] for i in range(len(x[0]))]

def save_preds(preds, cb=False):
    preds = np.ravel(preds)
    assert len(preds)==338489
    avg = np.average(preds)
    std = np.std(preds)
    p = '../%s%s_tl_result_%.4f_%.4f_%s.csv' % ('cb_' if cb else '', 
        strftime('%H%M_%m%d'), avg, std, args.m)

    df = pd.DataFrame({'instanceID': te_df_['instanceID'].values, 'proba': preds})
    df.sort_values('instanceID', inplace=True)
    df.to_csv(p, index=False)

    if cb: 
        return avg, std, p
    else:
        print('\nTrain average: ', tr_avg)
        print('Preds average: ', avg)
        print('Preds std dev.: ', std)
        print('\nWritten to: ', p)


class predCallback(Callback):
    def __init__(self, test_data):
        self.te_x = test_data

    def on_epoch_end(self, epoch, logs={}):
        predict_probas = np.ravel(self.model.predict(self.te_x, batch_size=40960, verbose=1))
        avg, std, p = save_preds(predict_probas, cb=True)
        print('\nTr avg: %.4f,   avg: %.4f, std: %.4f, written to: %s\n'%(tr_avg, avg, std, p))

'''
       ['userID', 'age', 'gender', 'education', 'marriageStatus', 'haveBaby',
        'hometown', 'residence', 'creativeID', 'adID', 'camgaignID',
        'advertiserID', 'appID', 'appPlatform', 'positionID', 'sitesetID',
        'positionType', 'weekDay', 'clickTime_d', 'clickTime_h', 'clickTime_m',
        'connectionType', 'telecomsOperator', 'conversionTime_d', 'label']
OrderedDict([('appCategory', 0.11342039643011995),
           ('positionID', 0.097800108293958354),
           ('positionType', 0.087460470534174911),
           ('creativeID', 0.071309827665513831),
           ('appID', 0.06958037043067912),
           ('adID', 0.063169867995414214),
           ('advertiserID', 0.05492409862736055),
           ('camgaignID', 0.050796291439061121),
           ('sitesetID', 0.013700972348893198),
           ('connectionType', 0.010308483261562163),
           ('residence', 0.0075065408905755091),
           ('age', 0.006662250035074235),
           ('hometown', 0.0057415423374340075),
           ('haveBaby', 0.0050835138043705281),
           ('telecomsOperator', 0.0049696062876111837),
           ('gender', 0.0046607920728349225),
           ('education', 0.0030160253068509456),
           ('clickTime_h', 0.0029165451700999038),
           ('clickTime_d', 0.0028077837477102278),
           ('weekDay', 0.001772181032721335),
           ('marriageStatus', 0.0016869540469555094),
           ('appPlatform', 0.0008981783007564663),
           ('clickTime_m', 0.00072661907607473827),
           ('userID', unkown)])
'''    
# ========================= 1 Data preparation ========================= #
# features = ['appID', 'connectionType', 'age', 'telecomsOperator', 'gender', 'education', 'clickTime_h', 'weekDay',
#             'marriageStatus', 'appPlatform', 'clickTime_m']
features = ['appCategory', 'positionID', 'positionType', 'creativeID', 'appID', 'adID',
            'advertiserID', 'camgaignID', 'sitesetID', 'connectionType',
            'residence', 'age', 'hometown', 'haveBaby', 'telecomsOperator',
            'gender', 'education', 'clickTime_h', 'clickTime_d', 'weekDay',
            'marriageStatus', 'appPlatform', 'clickTime_m', 'userID']

             
features = features[:args.f]
print(features, len(features))
if args.of: 
    features = features[-1: ]

tr_ui_ = pd.read_csv('../data/pre/new_tr_ui.csv', header=None)
te_ui_ = pd.read_csv('../data/pre/new_te_ui.csv', header=None)
tr_ua_ = pd.read_csv('../data/pre/new_tr_ua.csv', header=None)
te_ua_ = pd.read_csv('../data/pre/new_te_ua.csv', header=None)
tr_adAppCate_ = pd.read_csv('../data/pre/new_adAppCate_tr.csv', index_col=0)
te_adAppCate = pd.read_csv('../data/pre/new_adAppCate_te.csv', index_col=0).values
tr_df_ = pd.read_csv('../data/pre/new_generated_train.csv')
te_df_ = pd.read_csv('../data/pre/new_generated_test.csv')

print('\n\nLoaded datasets')

# batch_sizes = list(range(3072,8192,1024))
# batch_sizes.reverse()
# np.random.shuffle(batch_sizes)
batch_sizes = [args.b]
for bs in batch_sizes:
    print('\n\nSeed:', args.va_seed, 'Batch size: ', bs)
    frac = .1
    if not args.va:
        frac = 0
        print('Not using validation')

    va_ui = tr_ui_.sample(frac=frac, random_state=args.va_seed)
    tr_ui = tr_ui_.drop(va_ui.index, axis=0).values
    va_ui = va_ui.values
    te_ui = te_ui_.values

    va_ua = tr_ua_.sample(frac=frac, random_state=args.va_seed)
    tr_ua = tr_ua_.drop(va_ua.index, axis=0).values
    va_ua = va_ua.values
    te_ua = te_ua_.values


    va_adAppCate = tr_adAppCate_.sample(frac=frac, random_state=args.va_seed)
    tr_adAppCate = tr_adAppCate_.drop(va_adAppCate.index, axis=0).values
    va_adAppCate = va_adAppCate.values

    # va_df = tr_df.loc[tr_df['clickTime_d'] == 24]
    va_df = tr_df_.sample(frac=frac, random_state=args.va_seed)
    tr_df = tr_df_.drop(va_df.index, axis=0)

    if args.va:
        va_df = va_df[features+['label']]
        va = va_df.values
        va_x, va_y = va[:, :-1], va[:, -1:]


    if args.fra:
        tr_df = tr_df.sample(frac=.1)
        print('--- Sample to ', len(tr_df.index))

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
                >= args.vd, 'label'])/(np.sum(tr_df.loc[tr_df['clickTime_d'] == args.vd-1, 'label'])+1e-5))
            tr_df.loc[tr_df['conversionTime_d'] >= args.vd, 'label'] = 0

    tr_df = tr_df[features+['label']]
    te_df = te_df_[features]

    tr = tr_df.values
    te_x = te_df.values
    np.random.shuffle(tr)
    input_length = len(tr[0])-1
    max_feature = max(tr.max(), va.max() if args.va else 0, te_x.max())+1
    max_feature_ui = max(tr_ui.max(), te_ui.max(), va_ui.max() if args.va else 0)+1
    max_feature_ua = max(tr_ua.max(), te_ua.max(), va_ua.max() if args.va else 0)+1
    max_f_cols = pd.concat([tr_df_[features], te_df_[features]]).max().values +1

    max_f_ua_cols = pd.concat([tr_ua_, te_ua_]).max().values +1
    max_f_ui_cols = pd.concat([tr_ui_, te_ui_]).max().values +1

    print('--- Train cols: ', tr_df.columns)
    print('--- Max feature:', max_feature, max_feature_ua, max_feature_ui)
    print('--- Max column features:', max_f_cols)
    print('--- Max column ua features:', max_f_ua_cols)
    print('--- Max column ui features:', max_f_ui_cols)
    print('--- Selected most important features:', args.f)

    tr_x, tr_y = tr[:, :-1], tr[:, -1:]
    tr_avg = np.average(tr_y)

    # add two lists
    # tr_x = np.hstack([tr_x, tr_ui, tr_ua])
    # te = np.hstack([te, te_ui, te_ua])



    # ====================== 2 Build model and training ==================== #

    # 0 hyperparameters
    optimizer = 'rmsprop' # rmsprop, adam
    loss = 'binary_crossentropy'
    metrics = ['binary_crossentropy'] # can't be empty in this script


    workers = 1

    # 0.2 parameter instantiations
    tbCallBack = TensorBoard(log_dir='../meta/tbGraph/', histogram_freq=0,
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
        if args.m=='rnn': # recurrent networks
            model = Sequential()
            # model.add(Embedding(max_feature, 16, input_length = input_length))
            model.add(LSTM(128, activation='tanh', return_sequences=False, input_shape=(3, 28)))   
            # model.add(LSTM(32, activation='tanh'))
            model.add(Dense(64, activation='tanh')) 
            model.add(Dense(1, activation='sigmoid')) 
            model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

            tr_x = list(zip(tr_ui, tr_ua, tr_adAppCate))
            te_x = list(zip(te_ui, te_ua, te_adAppCate))
            if args.va: va_x = list(zip(va_ui, va_ua, va_adAppCate))
        
        if args.m=='mlp_fe': # multilayer perceptrons
            print('Using fine-grained embedding layers')
            cols_in = []
            cols_out = []
            inp_ui = Input(shape=(28, ))
            inp_ua = Input(shape=(28, ))

            # f = lambda x: int((math.log10(x)+1)*4-3)
            emb_dims = [32-2*i for i in range(16)] + [2]*(len(features)-16)
            f = lambda idx: emb_dims[idx]
            print([f(idx) for idx,_ in enumerate(max_f_cols)])
            for i,fe in enumerate(max_f_cols):
                col_in = Input(shape=(1,))
                col_out = Embedding(int(fe), f(i))(col_in)
                col_out = Flatten()(col_out)

                cols_in.append(col_in)
                cols_out.append(col_out)

            for f in max_f_ui_cols:
                col_in = Input(shape=(1,))
                col_out = Embedding(int(f), 8)(col_in)
                col_out = Flatten()(col_out)

                cols_in.append(col_in)
                cols_out.append(col_out) 
                
            for f in max_f_ua_cols:
                col_in = Input(shape=(1,))
                col_out = Embedding(int(f), 8)(col_in)
                col_out = Flatten()(col_out)

                cols_in.append(col_in)
                cols_out.append(col_out)


            cols_concatenated = concatenate(cols_out)
            if args.mt == 0:
                y = Dense(1024, activation='relu', 
                          kernel_regularizer='l1')(cols_concatenated)
                y = Dropout(.3)(y)
                y = Dense(512, activation='relu')(y)

            if args.mt == 1:
                y = Dense(1024, activation='relu', 
                          kernel_regularizer='l1')(cols_concatenated)
                y = Dropout(.2)(y)
                y = Dense(512, activation='relu')(y)

            if args.mt == 2:
                y = Dense(1024, activation='relu', 
                          kernel_regularizer='l1')(cols_concatenated)
                y = Dropout(.1)(y)
                y = Dense(512, activation='relu')(y)

            if args.mt == 3:
                y = Dense(1024, activation='relu')(cols_concatenated)
                y = Dropout(.1)(y)
                y = Dense(512, activation='relu')(y)

            if args.mt == 4:
                y = Dense(1024, activation='relu')(cols_concatenated)
                y = Dense(512, activation='relu')(y)
                y = Dense(512, activation='tanh')(y)


            y = Dense(1, activation='sigmoid')(y)  
            model = Model(cols_in, y)  

            tr_x = s_c(np.hstack([tr_x, tr_ui, tr_ua]))
            te_x = s_c(np.hstack([te_x, te_ui, te_ua]))
            if args.va: va_x = s_c(np.hstack([va_x, va_ui, va_ua]))

        if args.m=='mlp':
            print('Building MLP >>>>>>>>>>>')
            inp_x = Input(shape=(input_length, ))
            inp_adCate = Input(shape=(28, ))
            inp_ui = Input(shape=(28, ))
            inp_ua = Input(shape=(28, ))

            o_x = Embedding(max_feature, 16)(inp_x)

            o_adCate = Embedding(2, 2)(inp_adCate)
            o_ui = Embedding(max_feature_ui, 16)(inp_ui)
            o_ua = Embedding(max_feature_ua, 16)(inp_ua)

            o_x = Flatten()(o_x) # 16* max_feature
            o_adCate = Flatten()(o_adCate) # 16* max_feature
            o_ui = Flatten()(o_ui) # 16* max_feature
            o_ua = Flatten()(o_ua) # 16* max_feature

            y = concatenate([o_x, o_adCate, o_ui, o_ua])

            y = Dense(1024, activation='relu', kernel_regularizer='l1')(y)
            y = Dropout(.3)(y)
            y = Dense(512, activation='relu')(y)
            y = Dense(1, activation='sigmoid')(y)

            model = Model([inp_x, inp_adCate, inp_ui, inp_ua], y)

            tr_x = [tr_x, tr_adAppCate, tr_ui, tr_ua]
            te_x = [te_x, te_adAppCate, te_ui, te_ua]
            if args.va: va_x = [va_x, va_adAppCate, va_ui, va_ua]


        if args.m=='elr': # logistic regression after embedding
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

        # 5 fit the model (training)  callbacks=[predCallback(te_x)]   validation_data=(va_x, va_y), 
        if args.va: vali_data = (va_x, va_y)
        else: vali_data = None
        model.fit(tr_x, tr_y, epochs=args.e, validation_data=vali_data, 
                    shuffle=True, verbose=args.v, 
                    callbacks=[predCallback(te_x)] ,
                    batch_size=bs)

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

    predict_probas = np.ravel(model.predict(te_x, batch_size=4096*2, verbose=args.v))
    save_preds(predict_probas)
