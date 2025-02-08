import numpy as np
import matplotlib.pyplot as plt

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
    cost_sum = 0
    for x_t, y_t in zip(x, y):
        y_pre = forward(x_t, w)
        cost_sum = cost_sum + (y_t - y_pre) ** 2
    cost_sum = cost_sum / len(x)
    return cost_sum

# 梯度计算函数
def gradient(x, y, w):
    grad = 0
    for x_t, y_t in zip(x, y):
        grad = grad + 2 * x_t * (x_t * w - y_t)
    grad = grad / len(x)
    return grad

# 存储 w 和 cost 的变化
w_list = []
cost_list = []
converged = False  # 标记是否收敛
converged_iteration = 0  # 记录收敛时的迭代次数
cost_val=0

# 训练过程
print("before training", 'w', w)
for i in range(100):
    cost_old=cost_val
    w_list.append(i)
    cost_val = cost(x_data, y_data, w)
    grad_val = gradient(x_data, y_data, w)
    w = w - alpha * grad_val
    #print(grad_val)

    cost_list.append(cost_val)

    # 检查 w 是否收敛
    if abs(cost_val-cost_old) < 0.01 and not converged:
        converged = True
        converged_iteration = i
        print(f"Converged at iteration {i}")

print("after GD training", 'w', w)

# 绘制 w 和 cost 的变化图
plt.figure(figsize=(12, 5))

# 绘制 w 的变化
plt.subplot(1, 1, 1)
plt.plot(w_list, cost_list,label='w')
plt.xlabel('times')
plt.ylabel('cost')
plt.title('Change of cost during GD training')
plt.legend()

# 标记收敛点
if converged:
    plt.axvline(x=converged_iteration, color='red', linestyle='--', label=f'Converged at iteration {converged_iteration}')
    plt.legend()

plt.show()