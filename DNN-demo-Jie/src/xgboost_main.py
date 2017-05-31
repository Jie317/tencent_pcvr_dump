'''
Three IDs: userID, creativeID, positionID
'''
import argparse
trained_model_path = '../trained_models/last_dnn.h5'
parser = argparse.ArgumentParser(
    description='DNNs on Keras to predict conversion rate.')
parser.add_argument('-ct', action='store_true',
    help='continue training last model')
parser.add_argument('-ptd', action='store_true',
    help='part of train data in feature embedding training')
parser.add_argument('-m', type=str, default='no_mess',
    help='leave a message')
parser.add_argument('-et', type=str, default=None, 
    help='evaluate trained model (the last run by default))')
parser.add_argument('-nm', action='store_true', 
    help='not mask future information')


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
from collections import Counter
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
print(' >>>>>>>>> Devv stat 1577799 >>>>>>>>>>>> ')

def save_preds(preds, cb=False):
    preds = np.ravel(preds)
    assert len(preds)==338489
    avg = np.average(preds)
    std = np.std(preds)
    p = '../%s%s_dnn_tl_result_%.4f_%.4f_%s.csv' % ('callback_' if cb else '', 
        strftime('%H%M_%m%d'), avg, std, args.m)

    df = pd.DataFrame({'instanceID': te_df_['instanceID'].values, 'proba': preds})
    df.sort_values('instanceID', inplace=True)
    df.to_csv(p, index=False)

    if cb: 
        print(' Written to: ', p)
        return avg, std
    else:
        print('\nTrain average: ', tr_avg)
        print('Preds average: ', avg)
        print('Preds std dev.: ', std)
        print('\nWritten to: ', p)


'''
OrderedDict([('conversionTime_d', 0.25819888974716115),
             ('userID', 0.16155678776166404),
             ('appCategory', 0.11342039643011995),
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
             ('clickTime_m', 0.00072661907607473827)])
'''
# ====================================================================================== #
# ====================================================================================== #
# data
# features = ['appCategory', 'positionID', 'positionType', 'creativeID', 'appID', 'adID',
#             'advertiserID', 'camgaignID', 'sitesetID', 'connectionType',
#             'residence', 'age', 'hometown', 'haveBaby', 'telecomsOperator',
#             'gender', 'education', 'clickTime_h', 'clickTime_d', 'weekDay',
#             'marriageStatus', 'appPlatform', 'clickTime_m', 'userID']

cate_features = ['appCategory', 'positionType', 'connectionType', 'haveBaby', 'telecomsOperator',
            'gender', 'education', 'clickTime_h', 'weekDay',
            'marriageStatus', 'appPlatform']

features = ['advertiserID', 'sitesetID', 'age']
# features = ['connectionType', 'telecomsOperator', 'appPlatform', 'gender',
#        'education', 'marriageStatus', 'haveBaby', 'sitesetID', 'positionType',
#        'weekDay']
features.reverse()

tr_df = pd.read_csv('../data/pre/new_generated_train.csv')
te_df_ = pd.read_csv('../data/pre/new_generated_test.csv')
va_df = tr_df.loc[tr_df['clickTime_d'] == 24]

print(tr_df.head())

tr_ui = pd.read_csv('../data/pre/new_tr_ui.csv', header=None).values
te_ui = pd.read_csv('../data/pre/new_te_ui.csv', header=None).values
va_ui = pd.read_csv('../data/pre/new_va_ui.csv', header=None).values

tr_ua = pd.read_csv('../data/pre/new_tr_ua.csv', header=None).values
te_ua = pd.read_csv('../data/pre/new_te_ua.csv', header=None).values
va_ua = pd.read_csv('../data/pre/new_va_ua.csv', header=None).values




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
        print('--- Masked ', np.sum(tr_df.loc[tr_df['conversionTime_d'] 
            >= 24, 'label'])/(np.sum(tr_df.loc[tr_df['clickTime_d'] == 23, 'label'])+1e-5))
        tr_df.loc[tr_df['conversionTime_d'] >= 24, 'label'] = 0



x_all = pd.concat([tr_df[cate_features], te_df_[cate_features], va_df[cate_features]])

encoded_x = None
for f in cate_features:
    feature = x_all[f].values
    onehot_encoder = OneHotEncoder(sparse=False)
    feature = onehot_encoder.fit_transform(feature)
    if encoded_x is None:
        encoded_x = feature
    else:
        encoded_x = np.concatenate((encoded_x, feature), axis=1)
print("X shape: : ", encoded_x.shape)


tr_x = np.concatenate((encoded_x[:,:len(tr_df)], tr_ui, tr_ua, tr_df[features].values), axis=1)
te_x = np.concatenate((encoded_x[:,len(te_df_):-len(va_df)], te_ui, te_ua, te_df_[features].values), axis=1)
te_x = np.concatenate((encoded_x[:,-len(va_df):], te_ui, te_ua, te_df_[features].values), axis=1)
assert len(te_x)==len(te_df_)

tr_y = tr_df['label'].values
te_y = te_df_['label'].values
va_y = va_df['label'].values

# tr_df = tr_df[features+['label']]
# va_df = va_df[features+['label']]
# te_df = te_df_[features]
# tr = tr_df.values
# va = va_df.values
# te = te_df.values
# np.random.shuffle(tr)
# tr_x, tr_y = tr[:, :-1], tr[:, -1:]
# va_x, va_y = va[:, :-1], va[:, -1:]
# tr_avg = np.average(tr_y)
# max_f_cols = pd.concat([tr_df[features], te_df]).max().values+1
# print(tr_df.columns.values, '\n', max_f_cols)
# s_bef = Counter(np.ravel(tr_y))

# ====================================================================================== #
# xgboost

import xgboost as xgb  
from numpy import sort
from xgboost import plot_importance
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import log_loss
from matplotlib import pyplot
tr_y = np.ravel(tr_y)
va_y = np.ravel(va_y)

if 0: 
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.over_sampling import  RandomOverSampler
    # rus = RandomUnderSampler(.1)
    rus = RandomOverSampler(.1)
    tr_x, tr_y = rus.fit_sample(tr_x, tr_y)
    s_aft = Counter(list(tr_y))




gbm = xgb.XGBClassifier(max_depth=5, max_delta_step=1, silent=True, n_estimators=500, 
                        learning_rate=0.3, objective='binary:logistic', 
                        min_child_weight = 1, scale_pos_weight = 1,  
                        subsample=0.8, colsample_bytree=0.8,
                       
                       ).fit(tr_x, tr_y, eval_set=[(va_x, va_y)], 
                        eval_metric='logloss', verbose=True)
predict_probas = gbm.predict_proba(te)[:,1]

va_y_pred = gbm.predict_proba(va_x)[:,1]

logloss = log_loss(va_y, va_y_pred)
print('Logloss: ', logloss)
va_y_pred_class = gbm.predict(va_x)

s= classification_report(va_y, va_y_pred_class)
print(s)

va_y_pred = (va_y_pred > 0.5).astype('int32')
s= classification_report(va_y, va_y_pred)
print(s)


# Fit model using each importance as a threshold
thresholds = sort(gbm.feature_importances_)
print('\nFeature importance: ', thresholds)
for thresh in thresholds:
    # select features using threshold
    selection = SelectFromModel(gbm, threshold=thresh, prefit=True)
    select_tr_x = selection.transform(tr_x)
    # train model
    selection_model = xgb.XGBClassifier()
    selection_model.fit(select_tr_x, tr_y)
    # eval model
    select_va_x = selection.transform(va_x)
    va_y_pred = selection_model.predict(select_va_x)
    logloss = log_loss(va_y, va_y_pred)
    print('Logloss: ', logloss)
    s= classification_report(va_y, va_y_pred)
    print(s, 'Importance threshold: ', thresh)


plot_importance(gbm)
pyplot.show()

# ====================================================================================== #
# save result
save_preds(predict_probas)

va_y_pred = (va_y_pred > 0.5).astype('int32')
s= classification_report(va_y, va_y_pred)
print(s)
print('>>>>>>>> Original dataset shape {}'.format(s_bef))
print('>>>>>>>> Undersampled dataset shape {}'.format(s_aft))
