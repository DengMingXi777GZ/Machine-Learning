"""
Simple Linear Regression Model using PyTorch

This script demonstrates how to implement a simple linear regression model using PyTorch.
It includes the following steps:
1. Data preparation: Creates synthetic data for x and y values.
2. Model definition: Defines a forward function that represents the linear relationship y = wx.
3. Loss function: Defines a mean squared error loss function to evaluate the model's performance.
4. Training process: Iterates over the data, computes the loss, performs backpropagation to calculate gradients,
   and updates the model's weight using various optimizers including Rprop.
5. Prediction: After training, makes a prediction using the trained model.

The script explores different optimizers and their impact on the training process.
It includes comments on the performance of each optimizer for the given task.

Code author: DengMingXi
"""


import torch

x_data=torch.Tensor([[1.0],[2.0],[3.0]])
y_data=torch.Tensor([[3.0],[5.0],[7.0]])
#注意输入数据是一个大列表，数据点是小列表
class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel,self).__init__()
        self.linear=torch.nn.Linear(1,1)
    def forward(self,x):
        y_pre=self.linear(x)
        return y_pre
model=LinearModel() #创建实例

'''先定义一个名叫LinearModel的新类，继承自torch.nn.Module,    
#def __init__(self): LinearModel 类的构造函数，用于初始化类实例。__init__ 方法会在每次创建类的新实例时被调用。
super(LinearModel,self).__init__()是调用父类torch.nn.Module的构造函数，是初始化父类必须的操作
最后一行是输出神经元，(1,1)分别代表输入和输出的特征数
self的存在是指定特殊的类属性    
'''
Mloss=torch.nn.MSELoss(reduction='sum')#参数是决定要不要除以N，其实不大影响
optimizer=torch.optim.Rprop(model.parameters(),lr=0.01)

#optimizer=torch.optim.Rprop(model.parameters(),lr=0.01)
#optimizer=torch.optim.Adam(model.parameters(),lr=0.01)
#optimizer=torch.optim.Adamax(model.parameters(),lr=0.01)
#optimizer=torch.optim.SGD(model.parameters(),lr=0.01) 
#optimizer=torch.optim.ASGD(model.parameters(),lr=0.01)
#optimizer=torch.optim.LBFGS(model.parameters(),lr=0.01)
#optimizer=torch.optim.RMSprop(model.parameters(),lr=0.01)
#这是一个生成器函数，它返回模型中所有需要优化的参数（即模型的权重和偏置）。在 PyTorch 中，只有标记为 requires_grad=True 的参数才会被优化器更新。
'''
 SGD随机梯度下降，从数据中随机抽一个求梯度然后继续w更新
 Adagrad自适应梯度算法，为不同参数调整学习率，对稀疏数据表现良好。具体原理看note文档
 Rprop弹性反向传播Resilient Backpropagation，梯度的符号（正或负）来决定步长是增加还是减少。如果梯度的方向与上一次相同，则增加步长；如果方向相反，则减少步长，对稀疏数据适应性好，具体细节看note
 torch.optim.Adam (自适应矩估计)daptive Moment Estimation，借用了动量变化公式和Adgrad那个自动调整学习率的公式,在本代码中收敛的太慢了
 torch.optim.Adamax 与adam类似，但此处效果很差
 torch.optim.ASGD (平均随机梯度下降),对SGD的改进，此处表现尚可

 torch.optim.RMSprop，通过保持梯度的移动平均的平方来调整学习率。参数：学习率 lr，衰减率 alpha。此处效果不算太好

 torch.optim.LBFGS (拟牛顿法)
 高效性：LBFGS 只需要存储有限数量的历史梯度和更新信息，这使得它在内存使用上非常高效，适合大规模问题。
 适用性：LBFGS 适用于平滑的、大规模的优化问题，特别是当目标函数的 Hessian 矩阵变化不大时。
 收敛性：LBFGS 通常比纯梯度下降法收敛得更快，因为它利用了二阶信息。 此处效果还不错
 
 ''' 
'''
# 定义 closure LBFGS特有的结构
def closure():
    optimizer.zero_grad()
    y_pred = model(x_data)
    loss = Mloss(y_pred, y_data)
    loss.backward()
    return loss

# 训练模型
for epoch in range(100):  # 通常 LBFGS 只需要几次迭代
    optimizer.step(closure)
'''


for poch in range(100):
    y_pre=model(x_data)
    l=Mloss(y_pre,y_data)
    print("times:",poch,l.item())  

    optimizer.zero_grad()
    l.backward()
    optimizer.step()
  

print("w=",model.linear.weight.item())
print("b=",model.linear.bias.item())

x_test=torch.tensor([4.0])
y_test=model(x_test)
print("y_test=",y_test.item())