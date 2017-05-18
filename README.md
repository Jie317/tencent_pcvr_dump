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


## How to code data for Field-Feature Model
- 所有的特征必须转换成“field_id:feat_id:value”格式，field_id代表特征所属field的编号，feat_id是特征编号，value是特征的值


- 1. 采用 one-hot 进行数据重构
- Note: 源数值型特征的值归一化到 [0,1] 
- categorical特征需要经过One-Hot编码成数值型，编码产生的所有特征同属于一个field，而特征的值只能是0或1
        
- 2. 对所有的feature 进行编码
- 3. 对各个feature 根据所属的Field 进行编码
- 4. 选取每一列里面所有有值的数据(省略零值特征)，查找其Field和Feature
- 5. 将数据重构成 Label Field:Feature:value Field:Feature:value Field:Feature:value Field:Feature:value ....样式



数据处理代码框架
```
df1 = pd.read_csv(r'D:\dataScience\pre\new_generated_train.csv')

获取某一列的全部数据
df11.drop_duplicates([df11.columns[1]]).iloc[:,1].values
对每一个列下面的数据进行变换 
field 从0编号
label 从0编号

# 将需要进行one-hot编码的数据拆分出来
df11.drop(['dfdsf'],axis = 1,inplace = True)

# 获取一列的全部去重数据，每一数据作为one-hot 的一个标签
df11.drop_duplicates([df11.columns[1]]).iloc[:,1].values

# 将这一列提出，进行对比，得到True和False
df11.iloc[:,0] == 22

# 将True和False 转化为0和1， 并以类别和标签进行命名
df3['1:3'] = (df11.iloc[:,0] == 23).astype(int)

# 迭代每一行数据

# 对每一行数据进行处理，去除零值

# 讲每一行余下的数据，加上列表名做成Label:featur:values 的格式

```

## Key points from the online discussion (05.18)
  1 Xgboost might works as well.
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
