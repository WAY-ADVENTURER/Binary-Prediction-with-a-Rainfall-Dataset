import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import xgboost as xgb
from data_preprocessing import DataPreprocessor
from feature_engineering import FeatureEngineer
import joblib
import os

class ModelTrainer:
    def __init__(self):
        self.models = {
            'lgb': self._create_lgb_model(),
            'xgb': self._create_xgb_model()
        }
        
    def _create_lgb_model(self):
        """创建LightGBM模型"""
        return lgb.LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
    
    def _create_xgb_model(self):
        """创建XGBoost模型"""
        return xgb.XGBClassifier(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
    
    def train_and_evaluate(self, X, y):
        """训练并评估模型"""
        results = {}
        
        for name, model in self.models.items():
            print(f"\n训练 {name.upper()} 模型...")
            # 交叉验证
            cv_scores = cross_val_score(
                model, X, y, 
                cv=5, scoring='roc_auc'
            )
            
            # 在完整训练集上训练
            model.fit(X, y)
            
            results[name] = {
                'cv_score_mean': cv_scores.mean(),
                'cv_score_std': cv_scores.std(),
                'model': model
            }
            
            print(f"CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        
        return results
    
    def save_models(self, results):
        """保存模型"""
        # 获取当前脚本所在目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(current_dir, '..', 'models')
        
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
            
        for name, result in results.items():
            model_path = os.path.join(models_dir, f'{name}_model.pkl')
            joblib.dump(result['model'], model_path)
            print(f"模型已保存到: {model_path}")

def main():
    try:
        print("开始数据预处理...")
        # 数据预处理
        preprocessor = DataPreprocessor()
        preprocessor.load_data()
        data = preprocessor.preprocess_data()
        
        print("开始特征工程...")
        # 特征工程
        fe = FeatureEngineer()
        X_train = fe.fit_transform(
            pd.DataFrame(data['X_train'], 
                        columns=preprocessor.train_data.drop(['id', 'rainfall'], axis=1).columns),
            data['y_train']
        )
        
        print(f"特征工程完成，选择的特征数量: {X_train.shape[1]}")
        
        # 训练模型
        trainer = ModelTrainer()
        results = trainer.train_and_evaluate(X_train, data['y_train'])
        
        # 保存模型
        trainer.save_models(results)
        
        # 保存特征工程器
        current_dir = os.path.dirname(os.path.abspath(__file__))
        fe_path = os.path.join(current_dir, '..', 'models', 'feature_engineer.pkl')
        joblib.dump(fe, fe_path)
        print(f"特征工程器已保存到: {fe_path}")
        
    except Exception as e:
        print(f"训练过程出错: {str(e)}")

if __name__ == "__main__":
    main() 