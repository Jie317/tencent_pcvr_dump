<<<<<<< HEAD
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


## How to code data for Field-Feature Model
- 所有的特征必须转换成“field_id:feat_id:value”格式，field_id代表特征所属field的编号，feat_id是特征编号，value是特征的值


- 1. 采用 one-hot 进行数据重构
- Note: 源数值型特征的值归一化到 [0,1] 
- categorical特征需要经过One-Hot编码成数值型，编码产生的所有特征同属于一个field，而特征的值只能是0或1
        
- 2. 对所有的feature 进行编码
- 3. 对各个feature 根据所属的Field 进行编码
- 4. 选取每一列里面所有有值的数据，查找其Field和Feature
- 5. 将数据重构成 Label Field:Feature:value Field:Feature:value Field:Feature:value Field:Feature:value ....样式


