import sys
import pandas as pd
import numpy as np

'''
Task: check distribution of the prediction 
Input: result file

'''
def check_preds(p):
    preds = pd.read_csv(p, index_col=0).values
    assert len(preds)==338489

    print('Preds average: ', np.average(preds))
    print('Preds std dev.: ', np.std(preds))

def calibrate(p, d):
    preds = pd.read_csv(p, index_col=0).applymap(lambda x: x+d if x>-d else 0).values

    print('Calibrated preds average: ', np.average(preds))
    print('calibrated preds std dev.: ', np.std(preds))
    with open('calibrated.csv', 'w') as res:
        res.write('instanceID,prob\n')
        for i,pr in enumerate(preds): res.write('%s,%.8f\n' % ((i+1), pr))

check_preds(sys.argv[1])
if sys.argv[2] != '0':
	calibrate(sys.argv[1], float(sys.argv[2]))
