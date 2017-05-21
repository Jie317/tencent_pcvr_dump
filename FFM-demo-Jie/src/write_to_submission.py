import os
import pandas as pd
from time import strftime

def wtite_to_submission(p, outp):

    preds = pd.read_csv(p, header=None)

    assert len(preds.index)==338489

    with open(outp, 'w') as res:
        res.write('instanceID,pred\n')
        for i,p in enumerate(preds.values):
            res.write('%s,%.8f\n' % ((i+1), p))



ffm_results = '../results_from_ffm'
wtite_to_submission(ffm_results, '../submission_%s.csv' % strftime('%m%d_%H%M'))
os.remove(ffm_results)

print('Written to submission.csv')