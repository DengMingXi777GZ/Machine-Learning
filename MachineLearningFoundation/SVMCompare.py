#以红酒数据为训练集，比较线性核，高斯核SVM分类器，BP神经网络，与C4.5决策树的分类效果  

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier#多层感知机
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 加载鸢尾花数据集
wine=datasets.load_wine()
X=wine.data
y=wine.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 训练 SVM 模型
svm_linear = SVC(kernel='linear')
svm_linear.fit(X_train, y_train)
y_pred = svm_linear.predict(X_test)
print('SVM with linear kernel accuracy:', accuracy_score(y_test, y_pred))

# 训练 SVM 模型
svm_rbf = SVC(kernel='rbf')
svm_rbf.fit(X_train, y_train)
y_pred = svm_rbf.predict(X_test)
print('SVM with rbf kernel accuracy:', accuracy_score(y_test, y_pred))

# 训练 BP 神经网络模型
bp = MLPClassifier(hidden_layer_sizes=(100, 100,300,100), max_iter=5000)#隐藏层有两层，每层100个神经元
bp.fit(X_train, y_train)
y_pred = bp.predict(X_test)
print('BP neural network accuracy:', accuracy_score(y_test, y_pred))

# 训练 C4.5 决策树模型
dt = DecisionTreeClassifier(criterion='entropy')#
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
print('C4.5 decision tree accuracy:', accuracy_score(y_test, y_pred))

#画图比较
import matplotlib.pyplot as plt
import numpy as np

# 数据
x = np.arange(4)
y = [accuracy_score(y_test, svm_linear.predict(X_test)), accuracy_score(y_test, svm_rbf.predict(X_test)), accuracy_score(y_test, bp.predict(X_test)), accuracy_score(y_test, dt.predict(X_test))]

# 画图
plt.bar(x, y, color='skyblue')
plt.xticks(x, ['SVM with linear kernel', 'SVM with rbf kernel', 'BP neural network', 'C4.5 decision tree'])
plt.ylabel('Accuracy')
plt.title('Comparison of different classifiers')
#加上数字标签
for i, v in enumerate(y):
    plt.text(i, v + 0.01, '%.2f' % v, ha='center', va='bottom', fontsize=10)
#控制尺寸
plt.gcf().set_size_inches(10, 6)
plt.show()

