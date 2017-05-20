import argparse
parser = argparse.ArgumentParser(
    description='DNNs on Keras to predict conversion rate.')
parser.add_argument('-el', action='store_true', 
    help='evaluate the model from last run')
parser.add_argument('-ns', action='store_true', 
    help='don\'t save model in the end')
parser.add_argument('-os', action='store_true',
    help='oversampling')
parser.add_argument('-us', action='store_true',
    help='undersampling')
parser.add_argument('-cw', type=int, default=1,
    help='class weight for class 1')
parser.add_argument('-e', type=int, default=5,
    help='epochs')
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
import numpy as np
import pandas as pd
from time import time, strftime
from collections import Counter
from keras import backend as K
from keras.models import load_model, Sequential
from keras.utils import plot_model, to_categorical
from keras.callbacks import TensorBoard
from keras.layers import Dense, Embedding, LSTM, GRU, SimpleRNN
from keras.layers import Dropout, Bidirectional, Flatten 

def save_preds(preds, p):
    assert len(preds)==338489
    with open(p, 'w') as res:
        res.write('instanceID,prob\n')
        for i,pr in enumerate(preds): res.write('%s,%.8f\n' % (i, pr))
    print('\nWritten to result file: ', p)

    
# ========================= 1 Data preparation ========================= #
tr = pd.read_csv('../data/pre/new_generated_train.csv', index_col=0)
te = pd.read_csv('../data/pre/new_generated_test.csv', index_col=0)





tr = tr.values
te = te.values
input_length = len(tr[0])-1
max_feature = tr.max()+1
print('Max feature:', max_feature)

# 0.1 imbalanced learning strategies
class_weight = {0: 1, 1: args.cw}
imb = None
if args.os: imb = 'os'
if args.us: imb = 'us'

tr = (tr[:, :-1], tr[:, -1:])


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


# 2 fix random seed for reproducibility
np.random.seed(7)

# 3 build model or load last trained model
start = time()
if args.el or args.ct:
    model = load_model('../trained_models/dnn_last.h5')
else:
    if args.rnn: # recurrent networks
        model = Sequential()
        model.add(Embedding(max_feature, 16, input_length = input_length))
        model.add(LSTM(128, activation='tanh', return_sequences=True))   
        model.add(LSTM(32, activation='tanh'))
        model.add(Dense(1, activation='sigmoid')) 

    fined_embedding = True
    if args.mlp: # multilayer perceptrons
        if fined_embedding: pass



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

        
if not args.el:
    # 4 compile model
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.summary() 
    print(strftime('%c'))
  #  plot_model(model, to_file='../meta/model.png', show_shapes=True)

    # 5 fit the model (training)
    model.fit(*tr, epochs=args.e, shuffle=True, batch_size=batch_size, callbacks=[tbCallBack])

    if not args.ns: 
        model.save('../trained_models/dnn_%s.h5' % strftime("%m%d%H%M%S"))
        model.save('../trained_models/dnn_last.h5')


print('Runtime:', str(datetime.timedelta(seconds=int(time()-start))))
# 7 calculate predictions
print('\nPrediction')

predict_probas = np.ravel(model.predict_proba(te, batch_size=4096))
p = '../results_%s.csv' % (strftime('%m%d%H%M'), )
save_preds(predict_probas, p=p)
