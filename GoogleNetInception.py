#Inception Module
'''Inception Module是GoogleNet中的一个重要模块，其结构如下：
1x1卷积核：用于减少通道数，减少计算量
3x3卷积核：用于检测图像中的特征
5x5卷积核：用于检测图像中的特征
3x3最大池化：用于减少图像尺寸
它提供了不同的卷积核和池化层，可以提取不同的特征，从而提高模型的性能。
哪个好就增加那个的权重，这样可以提高模型的性能。'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets,transforms
from torch.utils.data import DataLoader

data_train=datasets.MNIST(root='./data',train=True,download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))]))
data_test=datasets.MNIST(root='./data',train=False,download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))]))
'''定义数据预处理流程，通过transforms.Compose组合多个操作。
transforms.ToTensor()将PIL图像或NumPy数组转换为PyTorch的Tensor，并自动将像素值从[0, 255]缩放到[0.0, 1.0]。
transforms.Normalize((0.1307,), (0.3081,))
均值 0.1307：MNIST训练集像素值的均值。
标准差 0.3081：MNIST训练集像素值的标准差。
公式：normalized = (input - mean) / std
效果：数据分布接近均值为0、标准差为1的正态分布，有利于模型训练。'''

data_train_loader=DataLoader(dataset=data_train,batch_size=64,shuffle=True)
data_test_loader=DataLoader(dataset=data_test,batch_size=64,shuffle=False)

class Inception(nn.Module):
    def __init__(self,in_channels):
        super(Inception,self).__init__()
        self.conv1=nn.Conv2d(in_channels,16,kernel_size=1)
        self.avepool=nn.AvgPool2d(3,stride=1,padding=1)#这个3是指3x3的卷积核

        #平均池化和普通池化不一样的地方是，平均池化是对卷积核中的值求平均
        self.branch2=nn.Conv2d(in_channels,24,kernel_size=1)

        self.conv15_1=nn.Conv2d(in_channels,16,kernel_size=3,padding=1)
        self.conv15_2=nn.Conv2d(16,24,kernel_size=5,padding=2)


        self.conv133_1=nn.Conv2d(in_channels,16,kernel_size=1)
        self.conv133_2=nn.Conv2d(16,24,kernel_size=3,padding=1)
        self.conv133_3=nn.Conv2d(24,24,kernel_size=3,padding=1)

    def forward(self,x):
        branch1=self.avepool(self.conv1(x))
        branch2=self.branch2(x)
        branch3=self.conv15_2(self.conv15_1(x))
        branch4=self.conv133_3(self.conv133_2(self.conv133_1(x)))
        output=torch.cat([branch1,branch2,branch3,branch4],dim=1) #dim=1表示在通道维度上进行拼接
        #print(output.shape)
        return output

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1=nn.Conv2d(1,10,kernel_size=5)
        self.conv2=nn.Conv2d(88,20,kernel_size=5)

        self.mp=nn.MaxPool2d(2)

        self.inception1=Inception(10)
        self.inception2=Inception(20)
        self.fc=nn.Linear(1408,10)

    def forward(self,x):
        in_size=x.size(0)
        x=F.relu(self.mp(self.conv1(x)))
        x=self.inception1(x)
        #print('after I1',x.shape)
        x=F.relu(self.mp(self.conv2(x)))
        x=self.inception2(x)
        #print('after I2',x.shape)
        x=x.view(in_size,-1)
        x=self.fc(x)
        return x

model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

def train(epoch):
    loss_sum = 0
    for batch_idx, (input, target) in enumerate(data_train_loader):
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
        if batch_idx % 300 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(input), len(data_train_loader.dataset),
                100. * batch_idx / len(data_train_loader), loss.item()))
            
def test():
    with torch.no_grad():
        total = 0
        correct = 0
        for input, target in data_test_loader:
            output = model(input)
            _, predicted = torch.max(output, 1) #返回的是最大值的索引
            total += target.size(0)
            correct += (predicted == target).sum().item()
        print('Accuracy: {:.2f}%'.format(100 * correct / total))

if __name__ == '__main__':
    for epoch in range(1, 10):
        train(epoch)
        test()

#第二轮准确率达到了98.02% 第7轮是98.87%，第9轮是99.09% 但训练时长明显高于前两个网络

