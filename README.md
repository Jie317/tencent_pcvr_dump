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
- Jie: Benchmark on MLPs without the two lists
- Yin:
- Wenwen:
