# 🏡基于Transformer的新闻文本分类

本项目使用Transformer模型对AG News数据集进行分类，包含完整的数据预处理、模型训练、评估和预测流程。

## 📊数据集
- AG News数据集（4个类别:World, Sports, Business, Sci/Tech）
- 训练集: 108000条样本
- 验证样本: 12000
- 测试样本: 7600

## 📂项目结构
```bplaintext
AG_News_Transformer_Classification/
│
├── data/                           # 数据目录
│   ├── ag_news_csv/                # 原始数据集
│   │   ├── classes.txt             # 类别标签文件
│   │   ├── test.csv                # 测试集
│   │   └── train.csv               # 训练集
│   ├── processed/                  # 预处理后数据
│   │   ├── train_processed.pkl     # 处理后的训练数据
│   │   └── test_processed.pkl      # 处理后的测试数据
│   └── ag_news_csv.tgz             # 原始压缩包（可选）
├── src/                            # 源代码
│   ├── models/                         # 模型目录
│   │   ├── transformer_model.pt        # 训练好的模型权重
│   │   └── tokenizer_config.json       # 分词器配置
│   ├── results/                        # 实验结果
│   │   ├── training_curves.png         # 训练曲线图
│   │   ├── confusion_matrix.png        # 混淆矩阵
│   │   └── classification_report.txt   # 分类报告
│   ├── config.yaml                 # 配置文件
│   ├── data_preprocessing.py       # 数据预处理脚本
│   ├── model.py                    # 模型定义
│   ├── train.py                    # 训练脚本
│   ├── evaluate.py                 # 评估脚本
│   └── predict.py                  # 预测脚本
├── requirements.txt                # 依赖库列表
└── README.md                       # 项目说明文档
```

## 🚀快速开始
1. **拉取项目**
```bash
git clone https://github.com/kongbai5128/news_text_classification.git
```
2. **安装依赖**
```bash
pip install -r requirements.txt
```
3. **数据预处理**
```bash
cd src
python src/data_preprocessing.py
```
4. **训练模型**
```bash
python src/train.py
```
5. 评估模型**
```bash
python src/evaluate.py
```
6. **交互式预测**
```bash
python src/predict.py
```
## 实验结果
测试准确率: 约92%

训练曲线保存在 results/training_curves.png

混淆矩阵保存在 results/confusion_matrix.png



## 📄使用说明

1. **下载数据集**:
   - 从Fast.ai获取AG News数据集:`https://s3.amazonaws.com/fast-ai-nlp/ag_news_csv.tgz`
   - 解压到`data/ag_news_csv/`目录

2. **运行顺序**:
   ```bash
   python src/data_preprocessing.py  # 数据预处理
   python src/train.py               # 训练模型
   python src/evaluate.py            # 评估模型
   python src/predict.py             # 进行预测
   ```
自定义配置:

修改config.yaml中的超参数

修改src/model.py中的模型架构