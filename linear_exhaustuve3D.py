import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#设置y=3x+4
x_data = [1.0, 2.0, 3.0]
y_data = [7.0, 10.0, 13.0]

def forward(x,w,b):
    return x*w+b
def loss(x,y,w,b):
    y_pred=forward(x,w,b)
    return(y_pred-y)**2

# 生成 w 和 b 的范围
w_range = np.arange(0.0, 4.1, 0.1)
b_range = np.arange(0.0, 4.1, 0.1)

# 创建网格 它们的形状是 (len(b_range), len(w_range))
W, B = np.meshgrid(w_range, b_range)

# 初始化 mse 矩阵
mse_list = np.zeros_like(W)  # 创建一个与 W 形状相同的二维数组

for i in range(len(w_range)):
    w=w_range[i]
    for j in range(len(b_range)):
        b=b_range[j]
        #print('w',w,'b',b)
        loss_sum=0
        for x, y in zip(x_data,y_data):
            sum=loss(x,y,w,b)
            loss_sum=loss_sum+sum
            #print('x',x,'y',y,'sum',sum)

        # print('loss_sum',loss_sum)
        mse_list[j,i]=loss_sum/len(x_data)
        #mse_list.append(mse)
        # print('mse',mse)
        #wb_list.append((w,b))
    
 

#index_min=mse_list.index(min(mse_list))
#print("Best w is:",w_list[index_min])

# plt.plot(w_list,mse_list)
# plt.ylabel('Loss')
# plt.xlabel('w')
# plt.show()

# np.unravel_index 函数将一维索引转换为多维索引。
# 数组的形状（mse_matrix.shape）。
# 例如，如果 mse_matrix 是一个 5x5 的矩阵，np.argmin(mse_matrix) 返回 12，
# 那么 np.unravel_index(12, (5, 5)) 
# 返回 (2, 2)，表示最小值位于第 3 行、第 3 列（索引从 0 开始）。
# np.argmin 函数返回数组中最小值的索引。 返回一个索引

index_min=np.unravel_index(np.argmin(mse_list),mse_list.shape)
best_w,best_b=w_range[index_min[1]],b_range[index_min[0]]
print('best_w',best_w,'best_b',best_b)
# index_min[0] 是最小值所在的行索引（对应于 b 的索引）。
# index_min[1] 是最小值所在的列索引（对应于 w 的索引）。


# 绘制 3D 图形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(W, B, mse_list, cmap='viridis')

# 设置标签
ax.set_xlabel('w')
ax.set_ylabel('b')
ax.set_zlabel('MSE')

plt.show()