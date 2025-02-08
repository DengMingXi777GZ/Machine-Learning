import numpy as np
import matplotlib.pyplot as plt
import random

# 数据
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# 初始参数
w = 1
alpha = 0.01

# 前向传播函数
def forward(x, w):
    return x * w

# 损失函数
def cost(x, y, w):
    y_pre = forward(x, w)
    cost_sum = (y - y_pre) ** 2
    return cost_sum

# 梯度计算函数
def Sgradient(x, y, w):
    data = list(zip(x, y))
    x_t, y_t = random.choice(data)
    grad = 2 * x_t * (x_t * w - y_t)
    return grad

# 存储 w 和 cost 的变化
w_list = []
cost_list = []
cost_val=0

# 训练过程
print("before training", 'w=', w)
converged = False  # 标记是否收敛
converged_iteration = 0  # 记录收敛时的迭代次数

for i in range(100):
    w_list.append(w)
    for x_t, y_t in zip(x_data, y_data):
        cost_old = cost_val
        grad_val = Sgradient(x_data, y_data, w)
        w_new = w - alpha * grad_val
        cost_val = cost(x_t, y_t, w_new)
        print(f"Iteration {i}, w={w}, grad={grad_val}, cost={cost_val}")

        # 检查 w 是否收敛
        if abs(cost_val-cost_old) < 0.01 and not converged:
            converged = True
            converged_iteration = i
            print(f"Converged at iteration {i}")

        w = w_new  # 更新 w

    cost_list.append(cost_val)

print("after training", 'w=', w)

# 绘制 cost 的变化图
plt.figure(figsize=(12, 5))
plt.plot(range(len(cost_list)), cost_list, label='Cost', color='orange')
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Change of Cost during SGD training')
plt.legend()

# 标记收敛点
if converged:
    plt.axvline(x=converged_iteration, color='red', linestyle='--', label=f'Converged at iteration {converged_iteration}')
    plt.legend()

plt.show()