#编写一个检测MNIST手写字母的卷积神经网络
#卷积神经网络有什么优势？
#1.卷积神经网络是专门为处理图像数据而设计的，因此在图像识别任务上表现优异。
#2.卷积神经网络可以自动提取图像中的特征，无需人工干预。
#3.卷积神经网络可以减少参数数量，减少过拟合。
#4.卷积神经网络可以通过卷积核的滑动来检测图像中的特征，例如边缘、纹理等。
#5.卷积神经网络可以通过池化层减少图像的大小，减少计算量。
#6.卷积神经网络可以通过多层卷积层和池化层来提取更高级的特征。
#7.卷积神经网络可以通过全连接层来进行分类。
#8.卷积神经网络可以通过激活函数来引入非线性。

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets,transforms
from torch.utils.data import DataLoader

data_train=datasets.MNIST(root='./data',train=True,download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))]))
data_test=datasets.MNIST(root='./data',train=False,download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))]))
data_train_loader=DataLoader(dataset=data_train,batch_size=64,shuffle=True)
data_test_loader=DataLoader(dataset=data_test,batch_size=64,shuffle=False)

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1=nn.Conv2d(1,10,kernel_size=5)
        self.conv2=nn.Conv2d(10,20,kernel_size=5)
        self.mp=nn.MaxPool2d(2)
        self.fc=nn.Linear(320,10)
    def forward(self,x):
        #(n,1,28,28)to(n,784)
        in_size=x.size(0)
        x=F.relu(self.mp(self.conv1(x)))#28->24->12
        x=F.relu(self.mp(self.conv2(x)))#12->8->4,20个通道
        x=x.view(in_size,-1)#将多维张量展平为一维张量,20*4*4=320个元素
        x=self.fc(x)
        return x
    
model=Net()
criterion=nn.CrossEntropyLoss()
optimizer=optim.SGD(model.parameters(),lr=0.01,momentum=0.5)

def train(epoch):
    loss_sum=0
    for batch_idx,(input,target) in enumerate(data_train_loader):
        optimizer.zero_grad()
        output=model(input)
        loss=criterion(output,target)
        loss.backward()
        optimizer.step()
        loss_sum+=loss.item()
        if batch_idx%300==0:
            print('Train Epoch:{} [{}/{} ({:.0f}%)]\tLoss:{:.6f}'.format(epoch,batch_idx*len(input),len(data_train_loader.dataset),100.*batch_idx/len(data_train_loader),loss.item()))

def test():
    with torch.no_grad():
        total=0
        correct=0
        for input,target in data_test_loader:
            output=model(input)
            _,predicted=torch.max(output,1) #这个1是指在第一个维度上取最大值，即取每一行的最大值
            total=total + target.size(0)
            correct+=(predicted==target).sum().item()
            print('Accuracy:{:.0f}%'.format(100.*correct/total))

if __name__ == '__main__':
    for epoch in range(1,10):
        train(epoch)
        test()
