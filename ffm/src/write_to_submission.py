import os
import numpy as np
import pandas as pd
from time import strftime

def wtite_to_submission(p, outp):

    preds = pd.read_csv(p, header=None).values

    assert len(preds)==338489

    tr = pd.read_csv('../data/pre/new_generated_train.csv', index_col=0)
    print('Train average: ', np.average(tr.values[ :, -1]))
    print('Preds average: ', np.average(preds))
    print('Preds std dev.: ', np.std(preds))

    with open(outp, 'w') as res:
        res.write('instanceID,pred\n')
        for i,p in enumerate(preds):
            res.write('%s,%.8f\n' % ((i+1), p))



ffm_results = '../results_from_ffm'
outp = '%s_ffm_sub.csv' % strftime('%H%M_%m%d')
wtite_to_submission(ffm_results, outp)
# os.remove(ffm_results)

print('Written to ', outp)