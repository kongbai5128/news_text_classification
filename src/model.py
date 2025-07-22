import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """位置编码层"""

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerTextClassifier(nn.Module):
    """Transformer文本分类模型"""

    def __init__(self, vocab_size, embed_dim, num_heads, hidden_dim, num_layers, num_classes, max_length=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=1)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=max_length)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(embed_dim, num_classes)
        self.embed_dim = embed_dim

    def forward(self, x):
        # 嵌入层
        x = self.embedding(x) * math.sqrt(self.embed_dim)
        x = x.permute(1, 0, 2)  # [seq_len, batch, embed_dim]

        # 位置编码
        x = self.pos_encoder(x)

        # Transformer编码器
        x = self.transformer_encoder(x)

        # 平均池化
        x = x.mean(dim=0)

        # 分类层
        x = self.fc(x)
        return x


def initialize_model(config, vocab_size):
    """初始化模型"""
    model = TransformerTextClassifier(
        vocab_size=vocab_size,
        embed_dim=config['model']['embedding_dim'],
        num_heads=config['model']['num_heads'],
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
        num_classes=config['model']['num_classes'],
        max_length=config['model']['max_seq_length']
    )
    return model