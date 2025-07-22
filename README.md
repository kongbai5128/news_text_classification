# 基于Transformer的新闻文本分类

本项目使用Transformer模型对AG News数据集进行分类，包含完整的数据预处理、模型训练、评估和预测流程。

## 数据集
- AG News数据集（4个类别:World, Sports, Business, Sci/Tech）
- 训练集: 108000条样本
- 验证样本: 12000
- 测试样本: 7600

## 项目结构
- **AG_News_Transformer_Classification/**
    - **data/**
        - ag_news_csv/
            - classes.txt
            - test.csv
            - train.csv
        - processed/
            - train_processed.pkl
            - test_processed.pkl
        - ag_news_csv.tgz
    - **models/**
        - transformer_model.pt
        - tokenizer_config.json
    - **src/**
        - config.yaml
        - data_preprocessing.py
        - model.py
        - train.py
        - evaluate.py
        - predict.py
    - **results/**
        - training_curves.png
        - confusion_matrix.png
        - classification_report.txt
    - requirements.txt
    - README.md


## 快速开始

1. **安装依赖**
```bash
pip install -r requirements.txt
```
数据预处理
```bash
python src/data_preprocessing.py
```
训练模型
```bash
python src/train.py
```
评估模型
```bash
python src/evaluate.py
```
交互式预测
```bash
python src/predict.py
```
实验结果
测试准确率: 约92%

训练曲线保存在 results/training_curves.png

混淆矩阵保存在 results/confusion_matrix.png

text

### 使用说明

1. **下载数据集**:
   - 从Fast.ai获取AG News数据集:`https://s3.amazonaws.com/fast-ai-nlp/ag_news_csv.tgz`
   - 解压到`data/ag_news_csv/`目录

2. **运行顺序**:
   ```bash
   python src/data_preprocessing.py  # 数据预处理
   python src/train.py               # 训练模型
   python src/evaluate.py            # 评估模型
   python src/predict.py             # 进行预测
自定义配置:

修改config.yaml中的超参数

修改src/model.py中的模型架构