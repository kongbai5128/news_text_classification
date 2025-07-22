import torch
import yaml
from data_preprocessing import clean_text, text_to_tensor
from model import initialize_model
import torchtext


def predict(text, model, vocab, tokenizer, device, max_length=128):
    """预测单个文本的类别"""
    # 清洗文本
    cleaned_text = clean_text(text)

    # 转换为张量
    input_tensor = text_to_tensor(cleaned_text, vocab, tokenizer, max_length).unsqueeze(0)

    # 预测
    model.eval()
    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        _, predicted_idx = torch.max(output, 1)

    return predicted_idx.item(), probabilities.squeeze().cpu().numpy()


def main():
    # 加载配置
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载词汇表
    vocab = torch.load('../data/processed/vocab.pkl')

    # 加载分词器
    tokenizer = torchtext.data.utils.get_tokenizer('basic_english')

    # 初始化模型
    model = initialize_model(config, len(vocab)).to(device)

    # 加载模型权重
    model.load_state_dict(torch.load('models/transformer_model.pt', map_location=device))

    # 类别名称
    class_names = ['World', 'Sports', 'Business', 'Sci/Tech']

    # 交互式预测
    print("新闻分类预测器 (输入 'exit' 退出)")
    while True:
        text = input("\n请输入新闻文本: ")
        if text.lower() == 'exit':
            break

        if len(text) < 10:
            print("文本太短，请输入更长的新闻内容")
            continue

        pred_idx, probs = predict(text, model, vocab, tokenizer, device,
                                  max_length=config['model']['max_seq_length'])

        print(f"\n预测类别: {class_names[pred_idx]}")
        print("各类别概率:")
        for i, prob in enumerate(probs):
            print(f"  {class_names[i]}: {prob:.4f}")


if __name__ == '__main__':
    main()