import pandas as pd
import numpy as np
import joblib
from data_preprocessing import DataPreprocessor
import os
import warnings
warnings.filterwarnings('ignore')


def load_models(models_dir='models'):
    """加载训练好的模型"""
    models = {}
    try:
        # 获取当前脚本所在目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir_path = os.path.join(current_dir, '..', models_dir)
        
        if not os.path.exists(models_dir_path):
            raise FileNotFoundError(f"模型目录不存在: {models_dir_path}")
            
        model_files = [f for f in os.listdir(models_dir_path) if f.endswith('_model.pkl')]
        if not model_files:
            raise FileNotFoundError("没有找到模型文件")
            
        for model_file in model_files:
            model_name = model_file.split('_')[0]
            model_path = os.path.join(models_dir_path, model_file)
            models[model_name] = joblib.load(model_path)
            print(f"成功加载模型: {model_name}")
            
    except Exception as e:
        print(f"加载模型时出错: {str(e)}")
        return None
        
    return models


def generate_predictions(threshold=0.5):
    """
    生成预测结果
    Args:
        threshold: 概率阈值，大于该值预测为1，否则预测为0
    """
    try:
        # 加载数据
        preprocessor = DataPreprocessor()
        preprocessor.load_data()
        data = preprocessor.preprocess_data()
        
        # 加载特征工程器
        current_dir = os.path.dirname(os.path.abspath(__file__))
        fe_path = os.path.join(current_dir, '..', 'models', 'feature_engineer.pkl')
        
        if not os.path.exists(fe_path):
            raise FileNotFoundError(f"特征工程器文件不存在: {fe_path}")
            
        fe = joblib.load(fe_path)
        print("成功加载特征工程器")
        
        # 准备测试数据
        X_test_df = pd.DataFrame(
            data['X_test'],
            columns=preprocessor.test_data.drop(['id'], axis=1).columns
        )
        
        # 转换测试集
        try:
            X_test_transformed = fe.transform(X_test_df)
            print(f"测试集特征转换完成，特征数量: {X_test_transformed.shape[1]}")
        except Exception as e:
            print(f"特征转换错误: {str(e)}")
            return None
            
        # 加载模型
        models = load_models()
        if not models:
            return None
            
        # 生成预测结果
        predictions = np.zeros(len(X_test_transformed))
        for name, model in models.items():
            try:
                pred = model.predict_proba(X_test_transformed)[:, 1]
                predictions += pred
                print(f"{name} 模型预测完成")
            except Exception as e:
                print(f"{name} 模型预测失败: {str(e)}")
                continue
                
        # 平均集成
        predictions /= len(models)
        
        # 将概率转换为二元预测（0或1）
        binary_predictions = (predictions >= threshold).astype(int)
        
        # 创建提交文件
        submission = pd.DataFrame({
            'id': data['test_ids'],
            'rainfall': binary_predictions
        })
        
        # 保存提交文件
        submissions_dir = os.path.join(current_dir, '..', 'submissions')
        if not os.path.exists(submissions_dir):
            os.makedirs(submissions_dir)
            
        submission_path = os.path.join(submissions_dir, 'submission.csv')
        submission.to_csv(submission_path, index=False)
        print(f"预测完成！提交文件已保存到: {submission_path}")
        
        # 打印预测结果统计
        print("\n预测结果统计:")
        print(f"预测值为1的样本数: {binary_predictions.sum()}")
        print(f"预测值为0的样本数: {len(binary_predictions) - binary_predictions.sum()}")
        print(f"使用的阈值: {threshold}")
        
        return submission
        
    except Exception as e:
        print(f"预测过程出错: {str(e)}")
        return None


if __name__ == "__main__":
    # 设置预测阈值
    THRESHOLD = 0.5  # 可以根据需要调整这个阈值
    
    submission = generate_predictions(threshold=THRESHOLD)
    if submission is not None:
        print("\n提交文件预览:")
        print(submission.head())
        
        # 验证预测值是否只包含0和1
        unique_values = submission['rainfall'].unique()
        print("\n预测值中包含的唯一值:", unique_values)
    else:
        print("生成预测失败")
