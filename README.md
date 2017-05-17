## Tencent pCVR competition

### 1 Code structure
#### 1.1 Preprocessing
- Input: raw data files, including both training and test datasets
- Output: pandas dataframes for training and test (no validation split)

#### 1.2 Model and training
- Input: pandas dataframes for training and test (no validation split)
- Output: standard test result file in submit format precised on the competition website

#### 1.3 Ensemble
- Input: test result files
- Output: final result file

### 2 Progress
- Synthesized new train and test datasets from raw csv files

### 3 Problems
- How to encode the two non-deterministic lists (user_installedapps and user_app_actions)?

### 4 What's going on?
- Jie: Test FFM with avazu datasets preprocessed by Random Walker
- Yin:
- Wenwen:


About the methods of Random Walker's model

##PART I

In his model, two models are used.
### Model 1. FREL : Follow the regularized leader - proximal
- an adaptive-learning-rate sparse logistic-regression with efficient L1-L2-regularization
- This model is implemented with python
- Input data: 21 original features + 8 additional features + 1 LSA feature + 19 gbdt features
- ->output1

### Model 2. FFM  : Field-aware Factorization Machine
- This model is implemented with C++
- Input data:21 original features + 8 additional features + 19 gbdt features
- ->output2

##PART II
### Input data : (the data separated by sites and apps)
- L1->isapp->False->21 original features + 8 additional features + 1 LSA feature + 19 gbdt features
- L2->isapp->True ->21 original features + 8 additional features + 1 LSA feature + 19 gbdt features
- L1+L2->output3

### Model 2. FFM
- Input data:
- L1->isapp->False->Input data:21 original features + 8 additional features + 19 gbdt features
- L2->isapp->True->Input data:21 original features + 8 additional features + 19 gbdt features
- L1+L2->output4

### Part III
- Ensemble
- output1+
- output2+
- output3+
- output4->output
