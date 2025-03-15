import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        
    def load_data(self):
        """加载训练集和测试集数据"""
        self.train_data = pd.read_csv('D:\\project\\kaggle\\热带雨林\\data\\train.csv')
        self.test_data = pd.read_csv('D:\\project\\kaggle\\热带雨林\\data\\test.csv')
        print(f"训练集形状: {self.train_data.shape}")
        print(f"测试集形状: {self.test_data.shape}")
        
    def preprocess_data(self):
        """数据预处理主函数"""
        # 分离特征和目标变量
        X = self.train_data.drop(['id', 'rainfall'], axis=1)
        y = self.train_data['rainfall']
        
        # 处理测试集
        X_test = self.test_data.drop(['id'], axis=1)
        
        # 特征标准化
        X_scaled = self.scaler.fit_transform(X)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 划分训练集和验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        return {
            'X_train': X_train,
            'X_val': X_val,
            'y_train': y_train,
            'y_val': y_val,
            'X_test': X_test_scaled,
            'test_ids': self.test_data['id']
        }

if __name__ == "__main__":
    # 测试数据预处理流程
    preprocessor = DataPreprocessor()
    preprocessor.load_data()
    data = preprocessor.preprocess_data()
    print("\n数据预处理完成！")
    print(f"训练集形状: {data['X_train'].shape}")
    print(f"验证集形状: {data['X_val'].shape}")
    print(f"测试集形状: {data['X_test'].shape}") 