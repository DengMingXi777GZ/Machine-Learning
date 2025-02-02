import numpy as np
import matplotlib.pyplot as plt
import random

# 数据
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# 初始参数
w = 1.0  # 初始权重
alpha = 0.01  # 学习率
batch_size = 2  # Mini-batch 的大小
threshold = 1e-2  # 收敛阈值
max_iterations = 100  # 最大迭代次数


# 前向传播函数
def forward(x, w):
    return x * w


# 损失函数
def cost(x, y, w):
    cost_sum = 0
    for x_t, y_t in zip(x, y):
        y_pre = forward(x_t, w)
        cost_sum += (y_t - y_pre) ** 2
    cost_sum = cost_sum / len(x)
    return cost_sum


# Mini-batch 梯度计算函数
def MBgradient(w, mini_batch):
    grad = 0
    for x, y in mini_batch:
        grad += 2 * x * (x * w - y)  # 计算梯度
    grad = grad / batch_size  # 对小批量梯度求平均
    return grad


# 存储损失函数值
cost_list = []

# Mini-batch 训练过程
print("Before training, w =", w)
converged = False  # 标记是否收敛
converged_iteration = 0  # 记录收敛时的迭代次数
cost_val=0

for i in range(max_iterations):
    cost_old = cost_val
    # 计算当前损失
    data = list(zip(x_data, y_data))  # 将 x 和 y 组合成 (x, y) 元组列表
    mini_batch = random.sample(data, batch_size)  # 随机抽取一个小批量
    x_test, y_test = zip(*mini_batch)
    cost_val = cost(x_test, y_test, w)
    cost_list.append(cost_val)

    # 计算 Mini-batch 梯度
    grad_val = MBgradient(w, mini_batch)

    # 更新权重
    w_new = w - alpha * grad_val
    print("Delta cost",cost_val-cost_old)
    # 检查是否收敛
    if abs(cost_val-cost_old) < threshold and not converged:
        converged = True
        converged_iteration = i
        print(f"Converged at iteration {i}")

    w = w_new  # 更新权重

print("After training, w =", w)

# 绘制损失函数变化图
plt.figure(figsize=(10, 6))
plt.plot(range(len(cost_list)), cost_list, label='Cost', color='blue')
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Mini-batch Gradient Descent: Cost vs. Iteration')

# 标记收敛点
if converged:
    plt.axvline(x=converged_iteration, color='red', linestyle='--',
                label=f'Converged at iteration {converged_iteration}')
    plt.legend()

plt.show()

# 输出收敛信息
if converged:
    print(f"Model converged at iteration {converged_iteration}")
else:
    print("Model did not converge within the maximum number of iterations.")
