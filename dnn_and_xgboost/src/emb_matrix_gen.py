d = '../data/pre/' # raw data directory
print('dev stat 22666633')
import os
import pandas as pd
import numpy as np
from time import time
import math
from sklearn.metrics import log_loss

def i_log(y):
  if y<= 0: y = 1e-5
  if y>=1: y = 1 - 1e-5
  return math.log(y/(1-y))


features = ['appCategory', 'positionID', 'positionType', 'creativeID', 'appID', 'adID',
            'advertiserID', 'camgaignID', 'sitesetID', 'connectionType',
            'residence', 'age', 'hometown', 'haveBaby', 'telecomsOperator',
            'gender', 'education', 'clickTime_h', 'clickTime_d', 'weekDay',
            'marriageStatus', 'appPlatform', 'clickTime_m', 'userID']

tr_ori = pd.read_csv('../data/pre/new_generated_train.csv', index_col=0)
te_ori = pd.read_csv('../data/pre/new_generated_test.csv', index_col=0)

maxs = pd.concat([tr_ori[features], te_ori[features]]).max()+1
# features = [ 'positionType', 'connectionType', 'age', 'haveBaby', 'telecomsOperator',
#             'gender', 'education', 'clickTime_h', 'clickTime_d', 'weekDay',
#             'marriageStatus', 'appPlatform', 'clickTime_m']
'''OrderedDict([('appCategory', 0.11342039643011995),
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
           ('clickTime_m', 0.00072661907607473827),
           ('userID', unkown)])'''


for i,f in enumerate(features):
  print('Processing ', f)
  tr_ = tr_ori.groupby(f).apply(lambda df: np.mean(df.label))

  tr__df = tr_.to_frame().reset_index()
  tr_dict = tr__df.set_index(tr__df[f]).to_dict()[0]


  emb_mat = []
  for r in range(maxs[i]):
    ran = np.random.random(16)*.5 - .25
    if r in tr_dict:
      ran = ran * (i_log(tr_dict[r])/np.sum(ran))
    emb_mat.append(ran)
  emb_mat = np.vstack(emb_mat)
  print(emb_mat.shape, ran)

  np.savetxt('emb_matrix/%s'%f, emb_mat, fmt='%.8f')
