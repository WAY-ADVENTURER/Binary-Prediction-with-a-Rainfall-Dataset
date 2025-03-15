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
#### 2.1 基础特征转换
- 标准化：使用`StandardScaler`将特征转换到相同尺度
- 特征平方：捕捉非线性关系
```python
for col in numeric_cols:
    df[f'{col}_squared'] = df[col] ** 2
```

#### 2.2 特征交互
- 特征乘积：捕捉特征间的相互作用
```python
for i in range(len(numeric_cols)):
    for j in range(i+1, len(numeric_cols)):
        col1, col2 = numeric_cols[i], numeric_cols[j]
        df[f'{col1}_times_{col2}'] = df[col1] * df[col2]
```

#### 2.3 特征选择
- 使用`SelectKBest`和F检验选择最重要的特征
- 固定选择20个最重要的特征，避免维度灾难
```python
selector = SelectKBest(score_func=f_classif, k=20)
```

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
