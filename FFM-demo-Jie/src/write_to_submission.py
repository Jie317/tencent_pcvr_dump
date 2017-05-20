import pandas as pd


def wtite_to_submission(preds):
	assert len(preds.index)==338489
    with open('submission.csv', 'w') as res:
        res.write('instanceID,pred\n')
        for i,p in enumerate(preds):
            res.write('%s,%.8f\n' % (i, p))



def read_ffm_preds(p):
	return pd.read_csv(p, header=None)


ffm_preds = read_ffm_preds('results_from_ffm')
wtite_to_submission(ffm_preds)


print('Written to submission.csv')