
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import Counter

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)


 # 定义中文字符池
chars = '你我他她它们的一是在有和了不人这中大合法萨科附近萨euro企鹅热，撒额为0o'

# ==================== 1. 数据生成 ====================
class TextClassificationDataset(Dataset):
    """
    1、生成训练的数据集
    2、构建词汇表
    3、提供将文本转换为词汇索引列表的方法
    4、提供数据集的大小
    5、提供获取单个样本的方法
    """

    def __init__(self, num_samples=1000, num_classes=5, max_length=5):
        """
        初始化函数,生成样本
        :param num_samples: 样本数量
        :param num_classes: 类别数量
        :param max_length: 最大文本长度
        """
        self.num_samples = num_samples
        self.num_classes = num_classes 
        self.max_length = max_length 

        # 生成随机文本数据
        self.texts = [] # 存储生成的文本列表
        self.labels = [] # 存储“你”对应的类别


        # 开始生成数据集 num_samples
        for i in range(num_samples):
            # 生成包含"你"字的五个字文本
            # "你"字的位置决定类别（0-4）
            you_pos = np.random.randint(0, 5)  # "你"字的位置
            label = you_pos  # "你"在第几位就属于第几类

            # 生成其他4个字符
            other_chars = np.random.choice([c for c in chars if c != '你'], size=4)

            # 构建文本，将"你"插入到指定位置
            text_list = list(other_chars)
            text_list.insert(you_pos, '你')
            text = ''.join(text_list) # 将列表转换为字符串，不使用分隔符

            self.texts.append(text)
            self.labels.append(label)

        # 构建词汇表
        self.build_vocab()

    
    def build_vocab(self):
        """
        构建词汇表
        """
        # 使用set创建集合进行去重，保留一个唯一字符  
        text1 = sorted(set(chars))

        # 构建词汇表，添加特殊token
        self.vocab = {'<PAD>': 0, '<UNK>': 1}
        for idx, char in enumerate(text1, start=2):
            self.vocab[char] = idx
        self.vocab_size = len(self.vocab)


    def tokenize(self, text):
        """将文本转换为词汇索引列表"""
        return [self.vocab.get(char, self.vocab['<UNK>']) for char in text]

    def __len__(self):
        """返回数据集大小"""
        return self.num_samples

    def __getitem__(self, idx):
        """返回一个训练样本"""
        text = self.texts[idx] # 获取文本
        label = self.labels[idx] # 获取文本对应的类别

        # 分词
        tokens = self.tokenize(text)

        # 填充或截断到固定长度
        if len(tokens) < self.max_length:
            tokens = tokens + [self.vocab['<PAD>']] * (self.max_length - len(tokens))
        else:
            tokens = tokens[:self.max_length]

        # 转换为 PyTorch 的张量
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(label, dtype=torch.long)

# ==================== 2. 模型定义 ====================
class TextRNNClassifier(nn.Module):
    """
    文本分类模型，使用LSTM
    架构：Embedding → LSTM → MaxPool → BN → Dropout → Linear → Sigmoid → (MSELoss)
    每个参数含义
    vocab_size: 词汇表大小
    embed_dim: 字符向量维度
    hidden_dim: 隐藏层维度
    num_layers: LSTM层数
    num_classes: 类别数量
    dropout: Dropout概率
    bidirectional: 是否使用双向LSTM
    """
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, num_classes, dropout=0.5, bidirectional=True):
        super(TextRNNClassifier, self).__init__()

        # Embedding层
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # LSTM层
        # batch_first=True 表示输入数据的形状为 (batch_size, seq_len, embed_dim)
        # 参数：输入的维度、隐藏层的维度、层数、是否使用BatchNorm、是否使用Dropout、是否双向
        self.lstm = nn.LSTM(embed_dim, 
                            hidden_dim, 
                            num_layers=num_layers, 
                            batch_first=True, 
                            dropout=dropout if num_layers > 1 else 0,
                            bidirectional=bidirectional)


        # 如果是双向LSTM，输出维度是 hidden_dim * 2
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        # 全连接层
        self.fc = nn.Linear(lstm_output_dim, num_classes)

        # Dropout 
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (batch_size, seq_len)

        # Embedding
        embedded = self.embedding(x)  # (batch_size, seq_len, embed_dim)

        # LSTM
        # output: (batch_size, seq_len, num_directions * hidden_dim)
        # hidden: (num_layers * num_directions, batch_size, hidden_dim)
        # c: (num_layers * num_directions, batch_size, hidden_dim)
        output, (hidden, c) = self.lstm(embedded)

        # 提取特征用于分类
        # 方法1：使用最后一个时间步的输出 (对于双向LSTM，需要拼接最后两个方向的输出)
        if self.lstm.bidirectional:
            # hidden[-1] 是前向最后一层, hidden[-2] 是后向最后一层
            hidden_cat = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden_cat = hidden[-1]

        # Dropout
        dropped = self.dropout(hidden_cat)

        # Linear + Sigmoid
        logits = self.fc(dropped)  # (batch_size, num_classes)
        output = torch.sigmoid(logits)

        return output

# ==================== 3. 训练函数 ====================
def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001):
    # 定义损失函数和优化器
    """"
    参数含义：
    model: 模型
    train_loader: 训练数据加载器
    val_loader: 验证数据加载器
    num_epochs: 训练轮数
    learning_rate: 学习率
    """
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # Adam优化器


    for epoch in range(num_epochs):
        model.train() # 切换到训练模式
        train_loss = 0.0 # 累加所有 batch 的损失值，用于计算平均损失。
        train_correct = 0 # 累加预测正确的样本数量。
        train_total = 0 # 累加总共处理的样本数量。

        """
        train_loader：它负责将数据分成一个个小批次。
        texts：当前 batch 的输入数据（例如文本索引）。
        labels：当前 batch 对应的真实标签（例如分类类别）。
        batch_idx：当前是第几批
        """
        for batch_idx, (texts, labels) in enumerate(train_loader):
            # 没32个样本，进行一次模型更新
            # 前向传播
            outputs = model(texts)
            # 计算损失
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad() # 清空梯度
            loss.backward() # 反向传播
            optimizer.step() # 更新参数

            # 统计
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1) # 获取预测结果，最大值的索引，即类别
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item() # 计算当前 batch 中预测正确的数量并累加

        # 验证
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad(): # 禁用梯度计算
            for texts, labels in val_loader:

                outputs = model(texts)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                print(f"outputs.data: {outputs.data},labels: {labels}")

        # 打印训练信息
        print(f'当前轮次 [{epoch+1}/{num_epochs}]')
        print(f'训练集的 Loss: {train_loss/len(train_loader):.4f}, 训练集的正确率 Acc: {100*train_correct/train_total:.2f}%')
        print(f'验证集的 Loss: {val_loss/len(val_loader):.4f}, 验证集的正确率 Acc: {100*val_correct/val_total:.2f}%')
        print('-' * 50)

# ==================== 4. 主函数 ====================
def main():
    # 参数设置
    num_samples = 1000
    num_classes = 5  # "你"字在5个位置，所以是5类
    max_length = 5   # 文本固定为5个字
    embed_dim = 128 # Embedding维度，每一个字对应一个128维的向量
    hidden_dim = 128      # RNN隐藏层维度
    num_layers = 2        # RNN层数
    bidirectional = False # 是否使用双向LSTM
    dropout = 0.5 # Dropout比例
    batch_size = 32 # 批量大小
    num_epochs = 10 # 训练轮数
    learning_rate = 0.001 # 学习率

    # 创建数据集
    dataset = TextClassificationDataset(num_samples=num_samples, 
                                       num_classes=num_classes, 
                                       max_length=max_length)

    # 划分训练集和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    # 随机打乱原始数据集，并将其分割为两个子集
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # 创建数据加载器
    # 通过 shuffle=True 防止模型过拟合数据的特定排列模式，提高模型的泛化能力。
    # batch_size：每个 batch 的样本数量。每次迭代，模型会从数据集中取出 batch_size 个样本进行训练。
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 创建模型
    model = TextRNNClassifier(
        vocab_size=dataset.vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout,
        bidirectional=bidirectional
    )

    # 打印模型信息
    print(f'词表大小 size: {dataset.vocab_size}')
    print(f'Model architecture:')
    print(model)
    print('-' * 50)

    # 训练模型
    train_model(model, train_loader, val_loader, num_epochs=num_epochs, learning_rate=learning_rate)

if __name__ == '__main__':
    main()
