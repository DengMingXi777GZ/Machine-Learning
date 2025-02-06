#构建卷积神经网络以分类MNIST手写字母
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets,transforms
from torch.utils.data import DataLoader

data_train=datasets.MNIST(root='./data',train=True,download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))]))
data_test=datasets.MNIST(root='./data',train=False,download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))]))
#transforms.Compose()将多个transform组合起来使用

data_train_loader=DataLoader(dataset=data_train,batch_size=64,shuffle=True)
data_test_loader=DataLoader(dataset=data_test,batch_size=64,shuffle=False)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.con1 = nn.Conv2d(1, 10, kernel_size=3)
        self.con2 = nn.Conv2d(10, 20, kernel_size=3)
        self.con3 = nn.Conv2d(20, 40, kernel_size=3, padding=1, bias=False)
        self.mp = nn.MaxPool2d(2)
        
        # 使用虚拟输入计算全连接层的输入尺寸
        self._initialize_fc()

    #下面这个函数是用来初始化全连接层的，求出全连接层的输入尺寸
    def _initialize_fc(self):
        with torch.no_grad():
            x = torch.zeros(1, 1, 28, 28)  # 创建一个虚拟输入
            x = self.forward_features(x)
            self.fc1 = nn.Linear(x.shape[1], 80)
            self.fc2 = nn.Linear(80, 40)
            self.fc3 = nn.Linear(40, 10)

    def forward_features(self, x):
        x = F.relu(self.mp(self.con1(x)))
        x = F.relu(self.mp(self.con2(x)))
        x = F.relu(self.mp(self.con3(x)))
        x = x.view(x.size(0), -1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
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
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        print('Accuracy: {:.2f}%'.format(100 * correct / total))

if __name__ == '__main__':
    for epoch in range(1, 11):
        train(epoch)
        test()

# 训练第五轮结束到达98%的准确率   