"""
Simple Linear Regression Model using PyTorch

This script demonstrates how to implement a simple linear regression model using PyTorch.
It includes the following steps:
1. Data preparation: Creates synthetic data for x and y values.
2. Model definition: Defines a forward function that represents the linear relationship y = wx.
3. Loss function: Defines a mean squared error loss function to evaluate the model's performance.
4. Training process: Iterates over the data, computes the loss, performs backpropagation to calculate gradients,
   and updates the model's weight using gradient descent.
5. Prediction: After training, makes a prediction using the trained model.

Code author: Deng MingXi
"""

import torch

x_data=[1.0,2.0,3.0]
y_data=[2.0,4.0,6.0]

w=torch.Tensor([1.0]) 
w.requires_grad=True #表明需要计算梯度
print(1)

def forward(x):
    #注意x和w都要转化成tensor
    return x*w

def loss(x,y):
    y_pre=forward(x)
    return (y_pre-y)**2

#此时以上的函数实际是在构建计算图，而非简单的乘法计算
print("predict before training",4,forward(4).item())

for poch in range(100):
    for x,y in zip(x_data,y_data):
        l=loss(x,y)
        l.backward() 
        #计算上面标注了requires_grad=True 的变量，将所有关于w的梯度存到w里面，然后马上释放计算图,每一次计算其实梯度都有可能不同，不要一直保留计算图
        print("\tgrad:",x,y,w.grad.item())
        #将一个包含单个元素的张量转换为 Python 标量。这在你需要获取张量的值作为 Python 数字时非常有用。
        #例如，如果你有一个标量张量 x = torch.tensor([[1.0]])，那么 x.item() 将返回 Python 浮点数 1.0
        w.data=w.data-0.01*w.grad.data 
        #一定要用w.grad.data把grad取出来作为数据，不然grad作为tensor只会做计算图运算
        #使用 w.grad.item() 时，确保梯度张量是标量（只有一个元素），否则会引发错误。
        #使用 w.grad.data 时，可以处理包含多个元素的梯度张量，适用于更广泛的应用场景
        w.grad.data.zero_() #跑完一次计算图一定要清0，不加的话每个循环的梯度就会相加

    print("progress",poch,l.item())


print("predict after training",4,forward(4).item())  #这个是检验的例子，输入4，看输出多少
    