#此代码用于展示随机森林使用相对多数投票输出作为输出学习机的效果
#还展示了STacking的效果
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

# 2.导入红酒数据
df = datasets.load_wine()
data = pd.DataFrame(df.data, columns=df.feature_names)#将数据转换为DataFrame格式
data['target'] = df.target
print(data.head())

# 3.数据预处理
# 3.1 数据标准化
scaler = StandardScaler()
#print(data.iloc[:, :-1].dtypes)  # 查看数据类型
for col in data.columns[:-1]:  # 遍历除最后一列外的所有列
    data[col] = data[col].astype(float)  # 显式转换为 float64

data.iloc[:, :-1] = scaler.fit_transform(data.iloc[:, :-1])#iloc[:, :-1]表示取除最后一列外的所有列
# 3.2 数据划分

X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# 4.建立模型
# 4.1 建立决策树模型
tree = DecisionTreeClassifier(criterion='entropy', max_depth=1)
# 4.2 建立RandomForest模型

rf = RandomForestClassifier(n_estimators=50, criterion='entropy', max_depth=1)#n_estimators是指基础模型的数量，learning_rate是指学习率
#集成的个体学习器都是决策树
# 5.训练模型
start = time.time()
rf.fit(X_train, y_train)
end = time.time()
print(f"Training time: {end - start}s")

# 6.预测
y_pred = rf.predict(X_test)
# 7.评估
accuracy = accuracy_score(y_test, y_pred)

# 8.输出
print(f"Ensemble_Accuracy: {accuracy}")

print('-'*20)
#Stacking
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# 定义初级学习器列表
base_learners = [('rf', RandomForestClassifier(n_estimators=50, criterion='entropy', max_depth=1))]

# 定义次级学习器
final_learner = LogisticRegression()

# 创建Stacking模型
stacking_model = StackingClassifier(estimators=base_learners, final_estimator=final_learner)

# 训练Stacking模型
stacking_model.fit(X_train, y_train)

# 预测
y_pred = stacking_model.predict(X_test)

# 评估
accuracy = accuracy_score(y_test, y_pred)
print(f"Stacking Ensemble Accuracy: {accuracy}")#准确率明显更高