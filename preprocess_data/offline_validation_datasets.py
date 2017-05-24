import pandas as pd

tr_df_raw = pd.read_csv('../data/pre/new_generated_train.csv', index_col=0)

tr_df = tr_df_raw.loc[(tr_df_raw['clickTime_d'] >= 17) & (tr_df_raw['clickTime_d'] <= 23)]

assert tr_df['clickTime_d'].max() < 24
tr_df.to_csv('../data/pre/offline_17_to_24_train.csv')


val_df = tr_df_raw.loc[tr_df_raw['clickTime_d'] == 24]
val_df.loc[val_df['conversionTime_d'] > 24, 'label'] = 0

assert val_df.loc[(val_df.conversionTime_d > 24), 'label'] == 0
val_df.to_csv('../data/pre/offline_24_val.csv')







print('Offline datasets gnerated')
