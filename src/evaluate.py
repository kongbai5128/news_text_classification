import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os
from model import initialize_model
import yaml


class TextDataset:
    """文本数据集类"""

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens, label = self.data[idx]
        return tokens, label


def main():
    # 加载配置
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载测试数据
    test_data = torch.load('../data/processed/test_processed.pkl')
    test_dataset = TextDataset(test_data)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False
    )

    # 加载词汇表
    vocab = torch.load('../data/processed/vocab.pkl')

    # 初始化模型
    model = initialize_model(config, len(vocab)).to(device)

    # 加载最佳模型
    model.load_state_dict(torch.load('models/transformer_model.pt', map_location=device))
    model.eval()

    # 评估模型
    all_preds = []
    all_labels = []

    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # 计算评估指标
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"Test Accuracy: {accuracy:.4f}")

    # 分类报告
    class_names = ['World', 'Sports', 'Business', 'Sci/Tech']
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print("\nClassification Report:")
    print(report)

    # 保存分类报告
    os.makedirs('results', exist_ok=True)
    with open('results/classification_report.txt', 'w') as f:
        f.write(report)

    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('results/confusion_matrix.png')
    print("Confusion matrix saved to results/confusion_matrix.png")

    # 错误分析示例
    errors = []
    for i in range(len(test_data)):
        if all_preds[i] != all_labels[i]:
            tokens, _ = test_data[i]
            text = ' '.join([vocab.get_itos()[idx] for idx in tokens if idx != vocab['<pad>']])
            errors.append({
                'text': text,
                'actual': class_names[all_labels[i]],
                'predicted': class_names[all_preds[i]]
            })
            if len(errors) >= 10:
                break

    print("\nError Analysis Examples:")
    for error in errors:
        print(f"Text: {error['text'][:100]}...")
        print(f"Actual: {error['actual']}, Predicted: {error['predicted']}\n")


if __name__ == '__main__':
    main()