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


parser.add_argument('-et', type=str, default=None, 
    help='evaluate trained model (the last run by default))')
parser.add_argument('-ns', action='store_true', 
    help='don\'t save model in the end')
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
parser.add_argument('-nfe', action='store_true',
    help='not use fine-grained embedding layers')
parser.add_argument('-m', type=str, required=True,
    help='leave a message')

import os
import json
import numpy as np
import pandas as pd
from time import time, strftime
from keras import backend as K
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.models import load_model, Sequential, Model
from keras.layers import Dense, Embedding, LSTM, GRU, SimpleRNN, BatchNormalization
from keras.layers import Dropout, Bidirectional, Flatten, Input
from keras.layers.merge import Concatenate, Add, concatenate, add
print(' >>>>>>>>> Devv stat 1532 >>>>>>>>>>>> ')

def save_preds(preds, p):
    assert len(preds)==338489
    print('\nTrain average: ', np.average(tr_y))
    print('Preds average: ', np.average(preds))
    print('Preds std dev.: ', np.std(preds))
    with open(p, 'w') as res:
        res.write('instanceID,prob\n')
        for i,pr in enumerate(preds): res.write('%s,%.8f\n' % ((i+1), pr))
    print('\nWritten to result file: ', p)

def s_c(x):
    return [x[:, i:i+1] for i in range(len(x[0]))]

# ====================================================================================== #
# ====================================================================================== #

features = ['appCategory', 'positionID', 'positionType', 'creativeID', 'appID', 'adID',
            'advertiserID', 'camgaignID', 'sitesetID', 'connectionType',
            'residence', 'age', 'hometown', 'haveBaby', 'telecomsOperator',
            'gender', 'education', 'clickTime_h', 'clickTime_d', 'weekDay',
            'marriageStatus', 'appPlatform', 'clickTime_m', 'userID']

# features = ['positionID', 'positionType']

tr_df = pd.read_csv('../data/pre/new_generated_train.csv', index_col=0)
te_df = pd.read_csv('../data/pre/new_generated_test.csv', index_col=0)
va_df = tr_df.loc[tr_df['clickTime_d'] == 24]


tr_df = tr_df[features+['label']]
va_df = va_df[features+['label']]
te_df = te_df[features]

tr = tr_df.values
va = va_df.values
te = te_df.values

tr_x, tr_y = tr[:, :-1], tr[:, -1:]
va_x, va_y = va[:, :-1], va[:, -1:]

tr_x = s_c(tr_x)
va_x = s_c(va_x)
te = s_c(te)

max_f_cols = tr_df.max().values[:-1]+1
print(tr_df.columns.values, '\n', max_f_cols)

# ====================================================================================== #

trained_model_path = '../trained_models/last_tl_dnn.h5'
f_model_paths = ['../trained_models/f_model_%.h5'%i for i in range(len(features))]

tbCallBack = TensorBoard(log_dir='../meta/tbGraph/', histogram_freq=1,
            write_graph=True, write_images=True)
checkpoint_all = ModelCheckpoint('../trained_models/tl_all_{epoch:02d}_{val_loss:.4f}.h5', 
			monitor='val_loss', verbose=1, save_best_only=True, period=1)
checkpoint = ModelCheckpoint('../trained_models/tl_%d_{epoch:02d}_{val_loss:.4f}.h5'%i, 
			monitor='val_loss', verbose=1, save_best_only=True, period=1)

# ====================================================================================== #

if args.et or args.ct:
    model_final = load_model(trained_model_path)
    print('\nLoaded model: %s' % trained_model_path)
    if args.s: model_final.summary() 


if args.r:
	# train models
	f_emb_models = []
	inps = [Input(shape=(1,), name='inp_%d'%i) for i in range(len(features))]
	for i,f in enumerate(features):
		print('\nStarting feature embedding traing for feature: ', i, f)

		y = Embedding(max_f_cols[i], 16, name='emb_%d'%i)(inps[i])
		y = Flatten(name='fla_%d'%i)(y)
		y = Dense(64, activation='relu', name='den_%d'%i)(y)
		y = Dense(1, activation='sigmoid', name='y_%d'%i)(y)
		f_model = Model(inps[i], y)

		print('--- Max feature:', max_f_cols[i])
		# f_model.summary()
		print([l.name for l in f_model.layers])

		f_model.compile('rmsprop', 'binary_crossentropy')

		f_model.fit(tr_x[i], tr_y, epochs=2, validation_data=(va_x[i], va_y), 
			shuffle=True, verbose=2, batch_size=512, callbacks=[checkpoint])

		f_model.save('../trained_models/f_emb_model_%d'%i)
		print('--- %d %s Evaluation on day '%(i,f), 24)
		scores = f_model.evaluate(va_x[i], va_y, batch_size=8196, verbose=1)
		print('--- %d %s Evaluation scores: '%(i,f), scores)

		for l in f_model.layers: l.trainable = False
		f_emb_models.append(f_model)
		f_model.save(filepath)


else:
	
	print('\nFinished feature embedding training')

	assert_w_before = f_emb_models[5].get_weights()
	print('Weights before', assert_w_before[3][:3])

	emb_ys = [m.get_layer('den_%d'%i).output for i,m in enumerate(f_emb_models)]

	y = concatenate(emb_ys)
	y = Dense(1024, activation='relu', kernel_regularizer='l1')(y)
	y = Dropout(.1)(y)
	y = Dense(512, activation='relu')(y)
	y = Dense(1, activation='sigmoid')(y)

	model_final = Model(inps, y)
	model_final.summary()
	model_final.compile('rmsprop', 'binary_crossentropy', ['binary_crossentropy'])


# ====================================================================================== #


print('Start training final model')
model_final.fit(tr_x, tr_y, epochs=5, validation_data=(va_x, va_y), 
		    shuffle=True, verbose=1,
		    batch_size=4096, callbacks=None)

# save model
model_final.save('../trained_models/%s_tl_dnn.h5' % strftime("%m%d_%H%M%S"))
model_final.save(trained_model_path)
print('Saved model: ', trained_model_path)


# predict
print('\nPredict')
predict_probas = np.ravel(model_final.predict(te, batch_size=40960, verbose=1))
p = '../%s_dnn_tl_sub.csv' % (strftime('%H%M_%m%d'), )
save_preds(predict_probas, p=p)


assert_w_after = f_emb_models[5].get_weights()
print('weights after: ', assert_w_after[3][:3])
