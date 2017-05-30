
# coding: utf-8
import sys
import pandas as pd
import numpy as np

try:
    ps = sys.argv[1:]
except Exception as e:
    raise e

print(ps)



# average
if len(ps)>1:
    probs = []
    for p in ps:
        d = pd.read_csv(p)
        probs.append(d.proba.values)
    preds=np.mean(np.vstack(probs), axis=0)
    df = pd.DataFrame({'instanceID': range(1, len(preds)+1), 'proba': preds})
    df.to_csv('../ensembled.csv', index=False)






if len(ps)==1: # rule out users already installed
    import pickle
    try:
        (ui_dic , adApp, userID) = pickle.load(open('../data/pre/dump_rule.bin', 'rb'))
    except Exception as e:
        print('Not found cached data.')
        print('Reading data')
        ui = pd.read_csv('../data/pre/user_installedapps.csv')
        te = pd.read_csv('../data/pre/new_generated_test.csv')
        ui_list = ui.groupby('userID').apply(lambda df: str(list(df.appID.values))).reset_index()
        ui_list.columns = ['userID', 'insApps']
        adApp = te.appID.values
        userID = te.userID.values
        ui_list = pd.read_csv('../data/pre/ui_list.csv')
        ui_dic = ui_list.set_index('userID').to_dict()['insApps']
        pickle.dump((ui_dic, adApp, userID), open('../data/pre/dump_rule.bin', 'wb'))


    preds = pd.read_csv(ps[0])['proba'].values
    c=0
    for i,uid in enumerate(userID):

        if uid in ui_dic:
            if adApp[i] in ui_dic[uid].replace('[|]','').split():
                preds[i] = 0
                c += 1
    print('Finished ruling ', c)








