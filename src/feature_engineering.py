import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

class FeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.selector = SelectKBest(score_func=f_classif, k=20)  # 固定选择20个特征
        self.selected_features = None
        
    def create_features(self, df):
        """创建新特征"""
        # 复制数据框
        df = df.copy()
        
        # 统计特征
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        
        # 创建基本统计特征
        for col in numeric_cols:
            df[f'{col}_squared'] = df[col] ** 2
        
        # 限制特征交互以减少特征数量
        for i in range(len(numeric_cols)):
            for j in range(i+1, len(numeric_cols)):
                col1, col2 = numeric_cols[i], numeric_cols[j]
                df[f'{col1}_times_{col2}'] = df[col1] * df[col2]
        
        return df
    
    def fit_transform(self, X, y=None):
        """拟合并转换数据"""
        # 创建新特征
        X_new = self.create_features(X)
        
        # 标准化
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_new),
            columns=X_new.columns
        )
        
        # 特征选择
        if y is not None:
            self.selector.fit(X_scaled, y)
            self.selected_features = X_scaled.columns[self.selector.get_support()].tolist()
            X_selected = X_scaled[self.selected_features]
            return X_selected
        return X_scaled
    
    def transform(self, X):
        """转换测试数据"""
        if self.selected_features is None:
            raise ValueError("必须先调用fit_transform")
            
        # 创建新特征
        X_new = self.create_features(X)
        
        # 标准化
        X_scaled = pd.DataFrame(
            self.scaler.transform(X_new),
            columns=X_new.columns
        )
        
        # 使用已选择的特征
        X_selected = X_scaled[self.selected_features]
        
        return X_selected

if __name__ == "__main__":
    # 测试特征工程
    from data_preprocessing import DataPreprocessor
    
    # 加载数据
    preprocessor = DataPreprocessor()
    preprocessor.load_data()
    
    # 获取特征
    X = preprocessor.train_data.drop(['id', 'rainfall'], axis=1)
    y = preprocessor.train_data['rainfall']
    X_test = preprocessor.test_data.drop(['id'], axis=1)
    
    # 特征工程
    fe = FeatureEngineer()
    X_transformed = fe.fit_transform(X, y)
    X_test_transformed = fe.transform(X_test)
    
    print("转换后的训练集形状:", X_transformed.shape)
    print("转换后的测试集形状:", X_test_transformed.shape)
    print("选择的特征:", fe.selected_features) 