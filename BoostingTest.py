import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import time
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import pearsonr

# 1. 导入红酒数据
df = datasets.load_wine()
data = pd.DataFrame(df.data, columns=df.feature_names)
data['target'] = df.target

# 2. 数据预处理
scaler = StandardScaler()
for col in data.columns[:-1]:
    data[col] = data[col].astype(float)
data.iloc[:, :-1] = scaler.fit_transform(data.iloc[:, :-1])

# 3. 数据划分
X = data.iloc[:, :-1].values  # 转换为 NumPy 数组
y = data.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# 4. 建立模型
tree = DecisionTreeClassifier(criterion='entropy', max_depth=1)
ada = AdaBoostClassifier(estimator=tree, n_estimators=50, learning_rate=0.1)
bag = BaggingClassifier(estimator=tree, n_estimators=50)

# 5. 训练模型并获取基学习器的预测结果
def get_base_learner_predictions(ensemble, X_test):
    predictions = []
    for estimator in ensemble.estimators_:
        predictions.append(estimator.predict(X_test))
    return np.array(predictions)

# 训练AdaBoost模型
start = time.time()
ada.fit(X_train, y_train)
end = time.time()
print(f"AdaBoost Training time: {end - start}s")

# 获取AdaBoost基学习器的预测结果
ada_predictions = get_base_learner_predictions(ada, X_test)

# 计算AdaBoost基学习器之间的相关性
ada_correlations = []
for i in range(len(ada_predictions)):
    for j in range(i + 1, len(ada_predictions)):
        corr, _ = pearsonr(ada_predictions[i], ada_predictions[j])
        ada_correlations.append(corr)
print(f"AdaBoost Base Learners Correlation: {np.mean(ada_correlations)}")
print('-'*20)
# 训练Bagging模型
start = time.time()
bag.fit(X_train, y_train)
end = time.time()
print(f"Bagging Training time: {end - start}s")

# 获取Bagging基学习器的预测结果
bag_predictions = get_base_learner_predictions(bag, X_test)

# 计算Bagging基学习器之间的相关性
bag_correlations = []
for i in range(len(bag_predictions)):
    for j in range(i + 1, len(bag_predictions)):
        corr, _ = pearsonr(bag_predictions[i], bag_predictions[j])
        bag_correlations.append(corr)
print(f"Bagging Base Learners Correlation: {np.mean(bag_correlations)}")
print('-'*20)
# 6. 预测并评估
y_pred_ada = ada.predict(X_test)
y_pred_bag = bag.predict(X_test)
print(f"AdaBoost Accuracy: {accuracy_score(y_test, y_pred_ada)}")
print(f"Bagging Accuracy: {accuracy_score(y_test, y_pred_bag)}")
print('-'*20)
# 7. 与单个决策树比较
tree1 = DecisionTreeClassifier(criterion='entropy', max_depth=1)
start = time.time()
tree1.fit(X_train, y_train)
end = time.time()
print(f"Single Decision Tree Training time: {end - start}s")
y_pred_tree = tree1.predict(X_test)
print(f"Single Decision Tree Accuracy: {accuracy_score(y_test, y_pred_tree)}")

#画图
import matplotlib.pyplot as plt
from sklearn import tree
# 画图
plt.figure(figsize=(12, 6))
tree.plot_tree(tree1, filled=True, feature_names=df.feature_names, class_names=df.target_names)
#plt.show()



'''决策树的分裂过程
根节点：所有460个样本进入根节点。
分裂条件：根据Glucose <= 1.052进行分裂。
True分支：386个样本满足条件Glucose <= 1.052，进入左子节点。
False分支：74个样本不满足条件Glucose <= 1.052，进入右子节点。
决策树的分类逻辑
左子节点：大多数样本属于类别0（283个样本），因此该节点的预测类别为0。
右子节点：大多数样本属于类别1（63个样本），因此该节点的预测类别为1。
'''