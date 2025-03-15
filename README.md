# 热带雨林降雨预测项目

## 项目概述
本项目是Kaggle竞赛 "Binary Prediction with a Rainfall Dataset" 的解决方案。目标是通过机器学习方法预测每日是否会降雨（二分类问题），使用ROC曲线下面积(AUC-ROC)作为评估指标。

## 详细方法论

### 1. 数据预处理
#### 1.1 数据加载与探索
- 使用`pandas`加载CSV格式的训练集和测试集
- 检查数据基本信息：形状、特征类型、缺失值等
- 分析特征分布和目标变量分布

#### 1.2 数据清洗
- 处理缺失值：填充或删除
- 处理异常值：检测和处理极端值
- 数据类型转换：确保特征类型合适

### 2. 特征工程
#### 2.1 原始特征（11个）
1. day（日期）
2. pressure（气压）
3. maxtemp（最高温度）
4. temparature（温度）
5. mintemp（最低温度）
6. dewpoint（露点）
7. humidity（湿度）
8. cloud（云量）
9. sunshine（日照）
10. winddirect（风向）
11. windspeed（风速）

#### 2.2 特征转换与创建
1. **标准化特征**
   - 使用`StandardScaler`将所有数值特征标准化
   - 目的：使所有特征在相同尺度上，提高模型性能

2. **平方特征**（11个）
   ```python
   for col in numeric_cols:
       df[f'{col}_squared'] = df[col] ** 2
   ```
   例如：
   - pressure_squared：捕捉气压的非线性影响
   - humidity_squared：捕捉湿度的非线性影响
   - temperature_squared：捕捉温度的非线性影响

3. **特征交互**（55个）
   ```python
   for i in range(len(numeric_cols)):
       for j in range(i+1, len(numeric_cols)):
           col1, col2 = numeric_cols[i], numeric_cols[j]
           df[f'{col1}_times_{col2}'] = df[col1] * df[col2]
   ```
   例如：
   - pressure_times_humidity：气压和湿度的交互作用
   - temperature_times_humidity：温度和湿度的交互作用
   - wind_times_humidity：风速和湿度的交互作用

#### 2.3 特征选择
- 使用`SelectKBest`和F检验从所有特征中选择最重要的20个特征
- 选择过程：
  1. 计算每个特征与目标变量（rainfall）的F值
  2. 选择F值最高的20个特征
  ```python
  selector = SelectKBest(score_func=f_classif, k=20)
  ```
- 选择20个特征的原因：
  1. 平衡模型复杂度和性能
  2. 避免维度灾难
  3. 减少过拟合风险

#### 2.4 特征重要性分析
在模型训练后，我们可以查看哪些特征被选中，以及它们的重要性排序。通常包括：
- 原始特征中的关键气象指标（如湿度、气压）
- 重要的特征交互项（如温度与湿度的交互）
- 显著的非线性特征（如重要特征的平方项）

### 3. 模型构建
#### 3.1 基础模型
使用两个强大的梯度提升树模型：

1. LightGBM模型
```python
lgb.LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
```

2. XGBoost模型
```python
xgb.XGBClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
```

#### 3.2 模型集成
- 使用简单平均法集成两个基础模型的预测结果
- 将概率预测转换为二元预测（使用阈值0.5）

### 4. 模型评估
- 使用5折交叉验证评估模型性能
- 使用ROC-AUC作为评估指标
- 分析模型在不同特征上的重要性

### 5. 预测与提交
- 对测试集进行预测
- 将预测概率转换为二元结果（0或1）
- 生成符合比赛要求的提交文件

## 项目结构
```
./
├── data/                    # 数据文件夹
│   ├── train.csv           # 训练数据
│   ├── test.csv            # 测试数据
│   └── sample_submission.csv# 提交样例
├── notebooks/              # Jupyter notebooks
│   └── EDA.ipynb          # 探索性数据分析
├── src/                    # 源代码
│   ├── data_preprocessing.py # 数据预处理
│   ├── feature_engineering.py# 特征工程
│   ├── model.py            # 模型定义
│   └── train.py           # 训练脚本
├── models/                 # 保存训练的模型
├── submissions/           # 预测结果
└── requirements.txt       # 项目依赖
```

## 如何使用

### 1. 环境配置
```bash
# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 运行项目
```bash
# 1. 数据探索（可选）
jupyter notebook notebooks/EDA.ipynb

# 2. 训练模型
python src/train.py

# 3. 生成预测
python src/predict.py
```

## 关键参数说明

### 特征工程
- `k=20`：选择20个最重要的特征
- 特征交互：生成特征间的乘积和平方特征

### 模型参数
1. LightGBM
- `n_estimators=1000`：树的数量
- `learning_rate=0.05`：学习率
- `num_leaves=31`：叶子节点数量
- `subsample=0.8`：样本采样比例
- `colsample_bytree=0.8`：特征采样比例

2. XGBoost
- `n_estimators=1000`：树的数量
- `learning_rate=0.05`：学习率
- `max_depth=6`：树的最大深度
- `subsample=0.8`：样本采样比例

### 预测
- `threshold=0.5`：二分类阈值，大于该值预测为1，否则为0

## 性能优化建议
1. 特征工程
   - 尝试创建更多的特征组合
   - 使用多项式特征
   - 尝试不同的特征选择方法

2. 模型调优
   - 使用网格搜索或贝叶斯优化调整超参数
   - 尝试其他模型（如CatBoost、随机森林等）
   - 使用更复杂的模型集成方法

3. 数据处理
   - 处理类别不平衡问题
   - 尝试不同的数据清洗策略
   - 考虑使用数据增强技术

## 常见问题解答
1. 模型过拟合
   - 减少模型复杂度
   - 增加正则化参数
   - 使用交叉验证

2. 特征选择
   - 可以尝试其他特征选择方法
   - 分析特征重要性
   - 考虑领域知识

3. 预测结果优化
   - 调整分类阈值
   - 分析错误预测案例
   - 优化模型集成策略
