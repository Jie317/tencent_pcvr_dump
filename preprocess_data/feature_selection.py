
import pandas as pd
import sys


features = ['positionID', 'positionType', 'creativeID', 'appID', 'adID',
            'advertiserID', 'camgaignID', 'sitesetID', 'connectionType',
            'residence', 'age', 'hometown', 'haveBaby', 'telecomsOperator',
            'gender', 'education', 'clickTime_h', 'clickTime_d', 'weekDay',
            'marriageStatus', 'appPlatform', 'clickTime_m'][:int(sys.argv[1])]

tr_df = pd.read_csv('../data/pre/new_generated_train.csv', index_col=0)
te_df = pd.read_csv('../data/pre/new_generated_test.csv', index_col=0)



features.append('label')
tr_df = tr_df[features]
te_df = te_df[features[:-1]]

tr_df.to_csv('../data/pre/new_generated_train_features_%s.csv' % sys.argv[1])
te_df.to_csv('../data/pre/new_generated_test_features_%s.csv' % sys.argv[1])
