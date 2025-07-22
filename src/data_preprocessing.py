import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
import torch
import os


def clean_text(text):
    """清洗文本数据"""
    # 移除HTML标签
    text = re.sub(r'<[^>]+>', '', text)
    # 移除非字母数字字符
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # 转换为小写
    text = text.lower()
    # 移除多余空格
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def load_data(data_path):
    """加载并预处理数据"""
    # 读取CSV文件
    train_df = pd.read_csv(os.path.join(data_path, 'train.csv'), header=None, names=['label', 'title', 'description'])
    test_df = pd.read_csv(os.path.join(data_path, 'test.csv'), header=None, names=['label', 'title', 'description'])

    # 合并标题和描述
    train_df['text'] = train_df['title'] + ' ' + train_df['description']
    test_df['text'] = test_df['title'] + ' ' + test_df['description']

    # 清理文本
    train_df['text'] = train_df['text'].apply(clean_text)
    test_df['text'] = test_df['text'].apply(clean_text)

    # 标签从1-4调整为0-3
    train_df['label'] = train_df['label'] - 1
    test_df['label'] = test_df['label'] - 1

    return train_df, test_df


def build_vocabulary(train_df, max_vocab_size=20000):
    """构建词汇表"""
    tokenizer = get_tokenizer('basic_english')

    def yield_tokens(data_iter):
        for _, row in data_iter.iterrows():
            yield tokenizer(row['text'])

    vocab = build_vocab_from_iterator(
        yield_tokens(train_df),
        specials=['<unk>', '<pad>'],
        max_tokens=max_vocab_size
    )
    vocab.set_default_index(vocab['<unk>'])
    return vocab, tokenizer


def text_to_tensor(text, vocab, tokenizer, max_length=128):
    """将文本转换为张量"""
    tokens = tokenizer(text)[:max_length]
    token_indices = [vocab[token] for token in tokens]
    # 填充或截断
    if len(token_indices) < max_length:
        padding = [vocab['<pad>']] * (max_length - len(token_indices))
        token_indices = token_indices + padding
    else:
        token_indices = token_indices[:max_length]
    return torch.tensor(token_indices, dtype=torch.long)


def prepare_datasets(train_df, test_df, vocab, tokenizer, max_length=128):
    """准备训练和测试数据集"""
    # 转换为张量
    train_df['tokens'] = train_df['text'].apply(
        lambda x: text_to_tensor(x, vocab, tokenizer, max_length))
    test_df['tokens'] = test_df['text'].apply(
        lambda x: text_to_tensor(x, vocab, tokenizer, max_length))

    # 创建数据集
    train_data = list(zip(train_df['tokens'], train_df['label']))
    test_data = list(zip(test_df['tokens'], test_df['label']))

    # 划分验证集
    train_data, val_data = train_test_split(
        train_data, test_size=0.1, random_state=42)

    return train_data, val_data, test_data


def save_datasets(train_data, val_data, test_data, save_dir='../data/processed'):
    """保存处理后的数据集"""
    os.makedirs(save_dir, exist_ok=True)
    torch.save(train_data, os.path.join(save_dir, 'train_processed.pkl'))
    torch.save(val_data, os.path.join(save_dir, 'val_processed.pkl'))
    torch.save(test_data, os.path.join(save_dir, 'test_processed.pkl'))


if __name__ == '__main__':
    # 数据路径
    data_dir = '../data/ag_news_csv'

    # 加载和预处理数据
    train_df, test_df = load_data(data_dir)

    # 构建词汇表
    vocab, tokenizer = build_vocabulary(train_df)

    # 准备数据集
    train_data, val_data, test_data = prepare_datasets(train_df, test_df, vocab, tokenizer)

    # 保存处理后的数据集
    save_datasets(train_data, val_data, test_data)

    # 保存词汇表
    torch.save(vocab, '../data/processed/vocab.pkl')

    print(f"数据预处理完成！词汇表大小: {len(vocab)}")
    print(f"训练样本: {len(train_data)}")
    print(f"验证样本: {len(val_data)}")
    print(f"测试样本: {len(test_data)}")