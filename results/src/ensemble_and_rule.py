
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

    rule = pickle.load(open('preds_rule.bin', 'rb'))

    preds_df = pd.read_csv(ps[0])
    preds_df.columns = ['instanceID', 'proba']

    print(preds_df.proba.describe())
   
    preds_df.proba = preds_df['proba'].values * rule

    print(preds_df.proba.describe())
    preds_df.to_csv('../submission.csv', index=None)








