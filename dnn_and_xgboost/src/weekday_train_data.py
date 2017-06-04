import pandas as pd

tr_df = pd.read_csv('../data/pre/new_generated_train.csv', index_col=0)

tr_df = tr_df.loc[(tr_df['clickTime_d'] == 17) | (tr_df['clickTime_d'] == 24)]

tr_df.to_csv('../data/pre/new_generated_train_2_days.csv')