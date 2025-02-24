from minisom import MiniSom
import numpy as np
import matplotlib.pyplot as plt

# 我想调用txt文件中的数据
data = np.genfromtxt('data\\Watermelon\\w3a.txt', delimiter=' ', usecols=(0, 1, 2), encoding='utf-8') # 加载数据，usecols是指定列,delimiter是指定分隔符
X = data[1:, :-1]  # 特征（密度和含糖率） [:, :-1]表示所有行，除了最后一列
y = data[1:, -1]   # 标签（好瓜）
print(X)

# 数据归一化
X_normalized = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

# 创建 SOM
som_dim = 20  # SOM 的维度（可以根据需要调整）
som = MiniSom(som_dim, som_dim, X.shape[1], sigma=2, learning_rate=0.4)  # 创建 SOM 网络，参数分别是行数，列数，输入数据的维度，sigma是半径，learning_rate是学习率

# 训练 SOM 网络
som.train_random(X_normalized, 5000)  # 训练 100 次迭代

# 可视化 SOM
plt.figure(figsize=(10, 10))
plt.imshow(som.distance_map(), cmap='bone_r')#颜色越深，距离越远，也就是说颜色越深，该节点的类别越明显
plt.colorbar()
plt.title('SOM Distance Map')

# 标记每个节点的类别
for i in range(X_normalized.shape[0]):
    wx, wy = som.winner(X_normalized[i])
    plt.text(wx, wy, f'{y[i]}', color='red', fontsize=12, ha='center', va='center')#在wx,wy处写上y[i]的值

plt.show()
#图中有多少个点，就有多少个最优节点（类别），每个点的颜色代表了它所属的类别
#这里的数据是随机生成的，所以不同的类别可能没有明显的区分