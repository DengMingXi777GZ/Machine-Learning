import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim

#数据准备
batch_size=64

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])  # 归一化：将图像的像素值从[0, 1]区间转换到均值为0.1307、标准差为0.3081的正态分布]) # 归一化,均值和方差

# 加载MNIST数据集
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = torch.nn.Linear(784,512)
        self.l2 = torch.nn.Linear(512,256)  
        self.l3 = torch.nn.Linear(256,128)
        self.l4 = torch.nn.Linear(128,64)
        self.l5 = torch.nn.Linear(64,10)

    def forward(self, x):
        x = x.view(-1, 784)  # 将输入的张量x展平为一维张量
        '''x 是一个多维张量，通常是一个包含图像数据的四维张量，形状为 (batch_size, channels, height, width)。
        例如，对于一批灰度图像，形状可能是 (batch_size, 1, 28, 28)。
        x.view(-1, 784) 将 x 重新调整形状为一个二维张量，其中 -1 表示自动计算该维度的大小，784 表示每个样本的特征数。
        这里 784 是由 28 * 28 得到的，因为每个图像有 28x28 个像素。
        -1 的作用是让 PyTorch 自动推断出这一维度的大小，以确保总元素数量不变。例如，如果 x 的原始形状是 (batch_size, 1, 28, 28)，
        那么 x.view(-1, 784) 会将其变为 (batch_size, 784)。
        这个操作的目的是将每个图像展平成一个一维向量，以便输入到全连接层（或其他需要一维输入的层）中。
        举个例子，如果 x 的形状是 (64, 1, 28, 28)，即一个批量包含 64 张 28x28 的灰度图像，
        那么 x.view(-1, 784) 会将其变为 (64, 784)，每张图像展平成一个 784 维的向量。'''
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x) # 最后一层不需要激活函数，因为在交叉熵损失函数中已经包含了softmax函数
    
model = Net()
criterion = torch.nn.CrossEntropyLoss()  # 交叉熵损失函数
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)  # 随机梯度下降优化器

def train(epoch):
    sum=0
    for batch_idx, (input, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output=model(input)
        loss=criterion(output,target)
        loss.backward()
        optimizer.step()

        sum=sum+loss.item()
        if batch_idx % 300 == 0:
            print(f'Train Epoch: {epoch} , Loss: {sum/300}')
            sum=0
        
def test():
    with torch.no_grad():
        test_loss=0
        correct=0
        for input, target in test_loader:
            output=model(input)
            test_loss+=criterion(output,target).item()
            pred=output.data.max(1,keepdim=True)[1]#返回每行的最大值及其索引，[1] 表示获取最大值的索引，即预测类别。
            correct+=pred.eq(target.data.view_as(pred)).sum()
        test_loss/=len(test_loader.dataset)
        print(f'\nTest set: Average loss: {test_loss}, Accuracy: {correct}/{len(test_loader.dataset)} ({100.*correct/len(test_loader.dataset):.0f}%)\n')

if __name__ == '__main__':
    for epoch in range(1, 10):
        train(epoch)
        test()