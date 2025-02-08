#构建简单的RNN网络加嵌入层来预测英文字母
'''Embedding层是将输入数据转化成稠密向量的层，这样可以减少数据的稀疏性，提高数据的表达能力。
他也可以做数据降维度'''
import torch
import torch.nn as nn
import torch.optim as optim

batch_size = 1
seq_len = 5  # 字符总数
input_size = 4  # 字符种类
hidden_size = 8  # 隐藏层大小
embed_size = 10  # 嵌入层大小
num_layers = 2  # 隐藏层层数


idx2char = ['e', 'h', 'l', 'o']
x_data = [[1, 0, 2, 2, 3]]  # hello （batch,seq_len)
y_data = [3, 1, 2, 3, 2]  # ohlol   (batch*seq_len)

inputs = torch.LongTensor(x_data)  # 将输入数据转化成张量
labels = torch.LongTensor(y_data)  # 标签数据
print('inputs:', inputs, inputs.shape)
print('labels:', labels, labels.shape)


class RNNEmbed(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_size):
        super(RNNEmbed, self).__init__()
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed = nn.Embedding(input_size, embed_size)  # 嵌入层
        self.rnn = nn.RNN(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers,batch_first=True)#batch_first=True表示输入数据的形状是(batch,seq_len,input_size),False表示(seq_len,batch,input_size)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size) 
        #为什么上面的参数要用size(0)而不是size(1)呢？因为这里的x是(batch,seq_len),所以要用size(0)
        x = self.embed(x)
        out, _ = self.rnn(x, h0)
        out=self.fc(out)
        return out.view(-1, input_size)#转化成(seq_len*batch_size,input_size)
        #input(batch,seq_len,input_size) output(batch,seq_len,hidden_size)
    

rnn = RNNEmbed(input_size, hidden_size, num_layers, batch_size)
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
