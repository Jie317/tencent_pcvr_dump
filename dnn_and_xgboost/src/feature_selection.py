
import pandas as pd
import sys
'''
OrderedDict([('appCategory', 0.11342039643011995),
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
           ('userID', unkown)])


'''

emb_dims = [32-2*i for i in range(16)] + [2]*(24-16)
print(emb_dims)
features = ['appCategory', 'positionID', 'positionType', 'creativeID', 'appID', 'adID',
            'advertiserID', 'camgaignID', 'sitesetID', 'connectionType',
            'residence', 'age', 'hometown', 'haveBaby', 'telecomsOperator',
            'gender', 'education', 'clickTime_h', 'clickTime_d', 'weekDay',
            'marriageStatus', 'appPlatform', 'clickTime_m', 'userID']

tr_df = pd.read_csv('../data/pre/new_generated_train.csv', index_col=0)
te_df = pd.read_csv('../data/pre/new_generated_test.csv', index_col=0)

print(tr_df[features].max())

# features.append('label')
# tr_df = tr_df[features]
# te_df = te_df[features[:-1]]

# tr_df.to_csv('../data/pre/new_generated_train_features_%s.csv' % sys.argv[1])
# te_df.to_csv('../data/pre/new_generated_test_features_%s.csv' % sys.argv[1])
