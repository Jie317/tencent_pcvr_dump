import argparse
parser = argparse.ArgumentParser(
    description='A MLP network on Keras to predict click-through rate.')
parser.add_argument('-r', action='store_true', 
    help='re-encode raw csv file')
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
parser.add_argument('-th', type=float, default=None,
    help='threshold to predict class')
parser.add_argument('-e', type=int, default=5,
    help='epochs')
parser.add_argument('-ct', action='store_true',
    help='continue training last model')
parser.add_argument('-la', action='store_true',
    help='load all training dataset (not use generator)')

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


def pred_classes_with_threshold(y_prob):
    y_class = []
    for v in y_prob:
        if v >= threshold: y_class.append(1)
        else: y_class.append(0)
    return y_class


def metrics_PRFm(y_real, y_pred): 
    '''
    compute precision, recall and f-measure
    '''
    # true_positives = K.sum(K.round(K.clip(y_real * y_pred, 0, 1)))
    # predicted_positives = K.sum(K.clip(y_pred, 0, 1))
    TP = np.dot(y_pred, y_real)[0]

    real_positive = np.sum(y_real)
    predicted_positives = np.sum(y_pred)
   
    P = TP/(predicted_positives+K.epsilon())
    R = TP/(real_positive+K.epsilon())
    Fm = 2*P*R/(P+R+K.epsilon())

    FP = predicted_positives - TP
    FN = real_positive - TP
    TN = len(y_real) - real_positive - FP

    return [P, R, Fm, TP, FP, FN, TN, real_positive, predicted_positives]


def save_preds(preds, p):
    with open(p, 'w') as res, open('../data/pre/test.csv', 'r') as raw:
        res.write('id,click\n')
        header = raw.readline()
        for i,p in zip(raw, preds):
            i = i.split(',')[0]
            res.write('%s,%.4f\n' % (i, p))
    print('\nWritten to result file')

# ===================================================================== #

# 0 hyperparameters
optimizer = 'rmsprop' # rmsprop, adam
# loss = 'categorical_crossentropy'
# metrics = ['categorical_crossentropy'] # can't be empty in this script
loss = 'binary_crossentropy'
metrics = ['binary_crossentropy'] # can't be empty in this script

# 0.1 imbalanced learning strategies
class_weight = {0: 1, 1: args.cw}
imb = None
if args.os: imb = 'os'
if args.us: imb = 'us'

batch_size = 4096
workers = 1


# 0.2 parameter instantiations
tbCallBack = TensorBoard(log_dir='../meta/tbGraph/', 
             write_graph=True, write_images=True)

input_length = 19
max_feature = 700000
print('Max feature:', max_feature)

# 2 fix random seed for reproducibility
np.random.seed(7)

# 3 build model or load last trained model
start = time()
if args.el or args.ct:
    model = load_model('../trained_models/pcvr_model_last.h5')
else:
    model = Sequential()
    model.add(Embedding(max_feature, 32, input_length = input_length))

    if args.rnn: # recurrent networks
        model.add(LSTM(128, activation='relu', return_sequences=True))   
        model.add(LSTM(32, activation='relu'))

    if args.mlp: # multilayer perceptrons
        model.add(Flatten())
        model.add(Dense(32*20, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))

    if args.lr: # logistic regression
        model.add(Flatten())
        # model.add(Dense(32*20, activation='relu'))       

    model.add(Dense(1, activation='sigmoid')) 

if not args.el:
    # 4 compile model
    model.compile(optimizer=optimizer, 
                  loss=loss, 
                  metrics=metrics)
    model.summary() 
    print(strftime('%c'))
  #  plot_model(model, to_file='../meta/model.png', show_shapes=True)


    # 5 fit the model (training)
    data = pd.read_csv('../data/pre/new_generated_train.csv').values
    data = (data[:, :-1], data[:, -1:])
    model.fit(*data,
              epochs=args.e,
              shuffle=True,
              batch_size=batch_size)

if not args.ns: 
    model.save('../trained_models/pcvr_model_%s.h5' % strftime("%m%d%H%M%S"))
    tmp_path = '../trained_models/pcvr_model_python_%s.py' % strftime('%m%d%H%M%S') 
    shutil.copyfile('main_pcvr_model.py', tmp_path)
    model.save('../trained_models/pcvr_model_last.h5')


print('Runtime:', str(datetime.timedelta(seconds=int(time()-start))))
# 7 calculate predictions
print('Prediction')
data_te = pd.read_csv('../data/pre/new_generated_test.csv').values
predict_probas = np.ravel(model.predict_proba(data_te, batch_size=4096))
p = '../data/results_%s.csv' % strftime('%m%d%H%M')
save_preds(predict_probas, p=p)
