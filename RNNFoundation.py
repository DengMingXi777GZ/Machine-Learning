#构建简单的RNN网络来预测英文字母

import torch
import torch.nn as nn
import torch.optim as optim

batch_size = 1
seq_len = 5  # 字符总数
input_size = 4  # 字符种类
hidden_size = 4  # 隐藏层大小
num_layers = 1  # 隐藏层层数

idx2char = ['e', 'h', 'l', 'o']
x_data = [1, 0, 2, 2, 3]  # hello
y_data = [3, 1, 2, 3, 2]  # ohlol
one_hot_lookup = [
    [1, 0, 0, 0],  # 0
    [0, 1, 0, 0],  # 1
    [0, 0, 1, 0],  # 2
    [0, 0, 0, 1]]  # 3

x_one_hot = [one_hot_lookup[x] for x in x_data]  # 将x_data转化成矩阵的数据编码
print('x_data转化成矩阵的数据编码', x_one_hot)
inputs = torch.Tensor(x_one_hot).view(seq_len, batch_size, input_size)  # 将输入数据转化成张量
labels = torch.LongTensor(y_data)  # 标签数据
print('inputs:', inputs)
print('labels的形状是:', labels.shape)

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_size):
        super(RNN, self).__init__()
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(1), self.hidden_size)  # 初始化隐藏状态 x.size(1)是batch_size,x.size包含了seq_len和batch_size
        print('x.size(1)是:', x.size(1))
        out, _ = self.rnn(x, h0)
        return out.view(-1, self.hidden_size)  # 转化成(seqlen*batch_size,hidden_size)

rnn = RNN(input_size, hidden_size, num_layers, batch_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(rnn.parameters(), lr=0.1)

for epoch in range(30):
    optimizer.zero_grad()
    outputs = rnn(inputs)
    print('outputs的形状是:', outputs.shape)
    loss = criterion(outputs, labels)

    loss.backward()
    optimizer.step()

    _, idx = outputs.max(1)
    idx = idx.data.numpy()  # 将张量转化成numpy数组
    result_str = [idx2char[x] for x in idx.squeeze()]  # 去除张量中所有大小为1的维度
    print('epoch:', epoch + 1, 'loss:', loss.item(), 'prediction:', ''.join(result_str))
