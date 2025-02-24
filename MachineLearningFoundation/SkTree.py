#基于sklearn库来实现决策树
from sklearn import tree
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import graphviz #这个库用于可视化决策树
import pandas as pd

wine = load_wine()
X = wine.data#特征
y = wine.target#标签
print(wine.data.shape)#输出数据集的大小,178行13列
print(wine.target.shape)#输出标签的大小，178行

print(pd.concat([pd.DataFrame(wine.data),pd.DataFrame(wine.target)],axis=1))#第一个参数是将两个DataFrame合并，axis默认为0，是纵向合并，为1是横向合并
print(wine.feature_names)#输出特征的名字
print(wine.target_names)#输出标签的名字,1,2,3分别代表三个类别

#划分数据集
Xtrain,Xtest,Ytrain,Ytest = train_test_split(X,y,test_size=0.3)
print('Xtrain.shape:',Xtrain.shape,'Xtest.shape:',Xtest.shape,'Ytrain.shape:',Ytrain.shape,'Ytest.shape:',Ytest.shape)#输出训练集和测试集的大小

clf = tree.DecisionTreeClassifier(criterion="entropy")#实例化
clf = clf.fit(Xtrain,Ytrain)#训练模型
score = clf.score(Xtest,Ytest)#返回预测的准确accuracy
print('accuracy:\t',score)

#可视化决策树
feature_name = ['酒精','苹果酸','灰','灰的碱性','镁','总酚','类黄酮','非黄烷类酚类','原花青素','颜色强度','色调','od280/od315稀释葡萄酒','脯氨酸']
dot_data = tree.export_graphviz(clf,out_file=None,feature_names=feature_name,class_names=['琴酒','雪莉','贝尔摩德'],filled=True,rounded=True,special_characters=True,fontname='SimHei')
#每个参数的意义：clf是训练好的模型，out_file=None表示不输出文件，feature_names是特征的名字，class_names是标签的名字，filled=True表示填充颜色，rounded=True表示圆角，special_characters=True表示特殊字符，fontname='SimHei'表示字体
graph = graphviz.Source(dot_data)
#展示图片
#graph.render('wine')
#graph.view()#打开图片
#图中的values表明了每个类别的样本量，entropy表示信息熵，samples表示样本量，value表示每个类别的样本量，class表示类别
#特征重要性
clf.feature_importances_
print([*zip(feature_name,clf.feature_importances_)])#将特征名字和特征重要性一一对应,0.0表示该特征对于分类没有贡献
#每次运行结果不一样，因为每次划分数据集的时候都是随机的，所以每次训练的结果都不一样
#在每次分枝时，不从使用全部特征，而是随 机选取一部分特征，从中选取不纯度相关指标最优的作为分枝用的节点。这样，每次生成的树也就不同了。

#random_state & splitter 用于控制随机性
clf = tree.DecisionTreeClassifier(criterion="entropy",random_state=30,splitter='random')#实例化
#random_state用于控制随机性，数值越大，随机性越小
# splitter用于控制随机划分，有两个值，random和best，random表示随机划分，best表示最优划分
'''输入 ”best" ，决策树在分枝时虽然随机，但是还是会 优先选择更重要的特征进行分枝（重要性可以通过属性feature_importances_ 查看）
输入 “random" ，决策树在 分枝时会更加随机，树会因为含有更多的不必要信息而更深更大，并因这些不必要信息而降低对训练集的拟合。
这 也是防止过拟合的一种方式。当你预测到你的模型会过拟合，用这两个参数来帮助你降低树建成之后过拟合的可能 性。
'''
clf = clf.fit(Xtrain,Ytrain)#训练模型
score = clf.score(Xtest,Ytest)#返回预测的准确accuracy
print('accuracy:\t',score)

#画图
dot_data = tree.export_graphviz(clf,out_file=None,feature_names=feature_name,class_names=['琴酒','雪莉','贝尔摩德'],filled=True,rounded=True,special_characters=True,fontname='SimHei')
graph = graphviz.Source(dot_data)
#graph.render('wine_random30')#更深更大的树，因为含有更多的不必要信息而更深更大，并因这些不必要信息而降低对训练集的拟合
#graph.view()

#剪枝参数
#max_depth 用于控制树的最大深度,min_samples_leaf 用于控制分枝后的每个子节点最少样本数,min_samples_split 用于控制一个节点只有在包含至少n个样本时才允许进行分裂
clf = tree.DecisionTreeClassifier(criterion="entropy",random_state=30,splitter='random',max_depth=3,min_samples_leaf=5,min_samples_split=5)#实例化
clf = clf.fit(Xtrain,Ytrain)#训练模型
score = clf.score(Xtest,Ytest)#返回预测的准确accuracy
print('accuracy(max_depth=3,samples_leaf=5,split=5):\t',score)
#其他剪枝参数
#min_impurity_decrease 用于控制信息增益的大小，如果小于设定值，不分枝
#min_impurity_split 用于控制信息增益的大小，如果小于设定值，不分枝
#max_features 用于控制分枝时考虑的特征个数
#min_weight_fraction_leaf 用于控制叶子节点样本权重占总权重的比例
#max_leaf_nodes 用于控制叶子节点的最大个数

#树的深度不一定是越深越好或越浅越好
test = []
for i in range(10):
    clf = tree.DecisionTreeClassifier(max_depth=i+1,criterion="entropy",random_state=30,splitter="random")
    clf = clf.fit(Xtrain, Ytrain)
    score = clf.score(Xtest, Ytest)
    test.append(score)
plt.plot(range(1,11),test,color="red",label="max_depth")
plt.legend()#max_depth=3时，准确率最高,最佳深度与数据集有关
#plt.show()

#目标权重参数 class_weight & min_weight_fraction_leaf
#class_weight 用于控制每个类别的权重(给小样本大权重)，可以是字典，balanced，balanced_subsample
#min_weight_fraction_leaf 用于控制叶子节点样本权重占总权重的比例
clf = tree.DecisionTreeClassifier(criterion="entropy",random_state=30,splitter='random',class_weight={0:1,1:1,2:100})#实例化
clf = clf.fit(Xtrain,Ytrain)#训练模型 class_weight={0:1,1:1,2:10}表示类别的权重都是1,1,10
score = clf.score(Xtest,Ytest)#返回预测的准确accuracy
print('accuracy(class_weight={0:1,1:1,2:10}):\t',score)

#min_weight_fraction_leaf表示叶子节点上所有样本权重和的最小值占总样本权重和的比例。如果叶子节点上的样本权重和小于这个值，则该节点会被剪枝。这个参数的默认值是 0，
clf = tree.DecisionTreeClassifier(criterion="entropy",random_state=30,splitter='random',min_weight_fraction_leaf=0.1)#实例化
clf = clf.fit(Xtrain,Ytrain)#训练模型
score = clf.score(Xtest,Ytest)#返回预测的准确accuracy
print('accuracy(min_weight_fraction_leaf=0.1):\t',score)
#画图
dot_data = tree.export_graphviz(clf,out_file=None,feature_names=feature_name,class_names=['琴酒','雪莉','贝尔摩德'],filled=True,rounded=True,special_characters=True,fontname='SimHei')
graph = graphviz.Source(dot_data)
#graph.render('wine_weight0.1')#给小样本大权重
#graph.view()


#输入矩阵一定要是二维的，即使只有一列，也要是二维的，如果是一维，应该reshape(-1,1)

#apply返回每个测试样本所在的叶子节点的索引,direction返回每个测试样本在叶子节点的方向
print(clf.apply(Xtest))
print(clf.predict(Xtest))
#print(clf.predict_proba(Xtest))#返回每个测试样本的概率

#总结

'''参数：
criterion：特征选择标准，gini或entropy
splitter：节点分裂策略，best或random
max_depth：决策树最大深度
min_samples_split：内部节点再划分所需最小样本数
min_samples_leaf：叶子节点最少样本数
min_weight_fraction_leaf：叶子节点最小的样本权重和
max_features：划分时考虑的最大特征数
random_state：随机种子
max_leaf_nodes：最大叶子节点数
min_impurity_decrease：节点划分最小不纯度,如果小于这个值，不分枝
min_impurity_split：节点划分最小不纯度
接口函数：
fit(X,y,sample_weight=None,check_input=True,X_idx_sorted=None)
预测：
apply(X,check_input=True) 是返回每个测试样本所在的叶子节点的索引
predict(X,check_input=True)是预测类别
predict_proba(X) 是预测概率

'''
