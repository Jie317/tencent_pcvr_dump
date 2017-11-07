## Tencent pCVR competition

### 1 Project structure
#### 1.1 Preprocessing
- Input: raw data files, including both training and test datasets
- Output: pandas dataframes for training and test (no validation split)

#### 1.2 Model and training
- Input: pandas dataframes for training and test (no validation split)
- Output: standard test result file in submit format precised on the competition website

#### 1.3 Ensemble
- Input: test result files
- Output: final result file

## Key points from the online discussion (05.18)
  1 Xgboost might work as well.
  
  2 Most of the time should be focused on feature engineering.
  
  3 Some rules may be applied, such as a user won't install again any app in his installed app list.
  
  4 How to deal with NANs in the data?
  
  5 Someone found similiar cvr in different age segmentations.
  
  6 Computation power (feasible with 8 cores and 16G RAM).
  
  7 Can we know the week day? This is important for the test data.
  
  8 Encoding IDs to vectors should work as well (a kaggle team won 2nd place before).
  
  9 DNN works well just in research papers?
  
  10 How to deal with clickTime?
  
  11 FFM combines features itself, while it's better to add synthetic features.
  
  12 If undersampling is applied, log loss may need to be biased.
  
  13 Test data may not represent real values.
  
  14 ConversionTime is not too much important.
  

## Data flow in preprocessing
  
  1 generate_new_data_from_raw_csv.py (finished)
  
    Input: all raw csv files excepts the two lists (installed_apps and install_actions)
    Output: new_generated_train/test.csv with columns: 
        ['userID', 'age', 'gender', 'education', 'marriageStatus', 'haveBaby',
        'hometown', 'residence', 'creativeID', 'adID', 'camgaignID',
        'advertiserID', 'appID', 'appPlatform', 'positionID', 'sitesetID',
        'positionType', 'weekDay', 'clickTime_d', 'clickTime_h', 'clickTime_m',
        'connectionType', 'telecomsOperator', 'conversionTime_d', 'label']

  2 weekday_train_data.py (finished)
    
    Input: new_generated_train/test.csv
    Output: only_17_24_two_days_train.csv with the same columns in input

  3 offline_validation_datasets.py (finished)
    
    Input: new_generated_train.csv
    Output:
        (1) 17_to_23_train.csv which directly pop from the input.
        (2) 24_offline_validation.csv which pop from the input but set the label to `0` for the rows with `conversionTime_d` bigger than `24`

  4 add_two_lists_info.py
    
    Input: new_generated_train/test.csv  
    Output: new_generated_with_two_lists_info_train/test.csv which aggregates potential statistic information from the two lists (user_installedapps.csv and user_app_actions.csv)

  5 rule_out_already_installed_users.py
    
    Input: pre_submission.csv
    Output: submission.csv which manually set the pred to `0` for all the users who have already installed the app

  6 ensemble.py
  
    Input: a set of submission.csv
    Output: submission.csv which average (or with weights) all the input csvs.
  
