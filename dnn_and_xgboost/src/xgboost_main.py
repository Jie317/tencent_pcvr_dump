import pandas as pd
import xgboost as xgb  
import numpy as np
from numpy import sort
from pprint import pprint
from xgboost import plot_importance
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import log_loss
from matplotlib import pyplot
from sklearn.metrics import  roc_auc_score
from sklearn.metrics import classification_report


def save_preds(preds, cb=False):
    preds = np.ravel(preds)
    assert len(preds)==338489
    avg = np.average(preds)
    std = np.std(preds)
    p = '../%s%s_tl_result_%.4f_%.4f_%s.csv' % ('cb_' if cb else '', 
        strftime('%H%M_%m%d'), avg, std, 'args.m')

    df = pd.DataFrame({'instanceID': te['instanceID'].values, 'proba': preds})
    df.sort_values('instanceID', inplace=True)
    df.to_csv(p, index=False)

    if cb: 
        return avg, std, p
    else:
        print('\nTrain average: ')
        print('Preds average: ', avg)
        print('Preds std dev.: ', std)
        print('\nWritten to: ', p)

# data

'''
fs = ['appCategory', 'positionID', 'positionType', 'creativeID', 'appID', 'adID',
            'advertiserID', 'camgaignID', 'sitesetID', 'connectionType',
            'residence', 'age', 'hometown', 'haveBaby', 'telecomsOperator',
            'gender', 'education', 'clickTime_h', 'clickTime_d', 'weekDay',
            'marriageStatus', 'appPlatform', 'clickTime_m', 'userID']
'''

tr = pd.read_csv('../data/pre/complete_train.csv')

te = pd.read_csv('../data/pre/complete_test.csv')
va = tr.sample(frac=.1, random_state=62)
tr = tr.drop(va.index, axis=0)




fs = ['appCategory', 'positionID', 'positionType', 'creativeID', 'appID', 'adID',
            'advertiserID', 'camgaignID', 'sitesetID', 'connectionType',
            'residence', 'age', 'hometown', 'haveBaby', 'telecomsOperator',
            'gender', 'education', 'clickTime_h', 'clickTime_d', 'weekDay',
            'marriageStatus', 'appPlatform', 'clickTime_m', 'userID']
fs_ac = [c for c in te.columns if 'adAppCat_' in c]
fs_ui = [c for c in te.columns if 'insCat_' in c]
fs_ua = [c for c in te.columns if 'actCat_' in c]

# ==================================================

a_fs = fs + fs_ac + fs_ui + fs_ua
print(len(a_fs))

tr_x = tr[a_fs].values
tr_y = tr['label'].values

va_x = va[a_fs].values
va_y = va['label'].values

te_x = te[a_fs].values
te_y = None

print(len(te_x), ...)

# xgboost


model = xgb.XGBClassifier(max_depth=10, max_delta_step=1, silent=False, n_estimators=73, 
                        learning_rate=0.3, objective='binary:logistic', 
                        min_child_weight = 1, scale_pos_weight = 1,  
                       ).fit(tr_x, tr_y, 
      eval_set=[(va_x, va_y)], 
      eval_metric='logloss', 
      verbose=True)

predict_probas = model.predict_proba(te_x)[:,1]
save_preds(predict_probas)

va_y_pred = model.predict_proba(va_x)[:,1]

logloss = log_loss(va_y, va_y_pred)
print('\n\nLogloss: ', logloss)
# va_y_pred_class = model.predict(va_x)

# s= classification_report(va_y, va_y_pred_class)
# s_rocauc= roc_auc_score(va_y, va_y_pred)
# print(s, '\n', s_rocauc)

pprint(list(zip(a_fs, model.feature_importances_)))