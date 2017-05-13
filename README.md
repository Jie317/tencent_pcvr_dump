# Tencent pCVR competition

# Code structure
## 1 Preprocessing
- Input: raw data files, including both training and test datasets
- Output: pandas dataframes for training and test (no validation split)

## 2 Model and training
- Input: pandas dataframes for training and test (no validation split)
- Output: standard test result file in submit format precised on the competition website

## 3 Ensemble
- Input: test result files
- Output: final result file
