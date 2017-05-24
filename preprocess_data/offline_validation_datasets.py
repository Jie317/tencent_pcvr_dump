import pandas as pd

tr_df = pd.read_csv('../data/pre/new_generated_train.csv', index_col=0)

val_df = tr_df.loc[tr_df['clickTime_d'] == 24]

tr_df = tr_df.loc[(tr_df['clickTime_d'] >= 17) & (tr_df['clickTime_d'] <= 23)]


tr_df.loc[tr_df['conversionTime_d'] > 23, 'label'] = 0

assert sum(tr_df.loc[(tr_df.conversionTime_d > 23), 'label']) == 0

tr_df.to_csv('../data/pre/offline_17_to_24_train.csv')
val_df.to_csv('../data/pre/offline_24_val.csv')







print('Offline datasets gnerated')
