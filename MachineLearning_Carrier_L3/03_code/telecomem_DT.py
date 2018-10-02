#!-*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn import cross_validation

data=pd.read_csv('D:\workspace\TelecomML\data\decision.csv',encoding='utf-8',sep=' ')
# print data.shape
# print data.head(5)
# print data.info()
# print data.describe()
col_dicts = {}
cols =  data.columns.values.tolist()
df=data.ix[:,cols[1:]]
# print df.shape
X=df.loc[:,cols[1:-1]]
# print X.info()
y=df[cols[-1]]
# print type(y)
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.3, random_state=0)
# print y_train.value_counts()/len(y_train)
# print y_test.value_counts()/len(y_test)
# from sklearn import tree
# from sklearn.ensemble import RandomForestClassifier
# credit_model = RandomForestClassifier(n_estimators=10)
# credit_model.fit(X_train, y_train)
# #
# credit_pred = credit_model.predict(X_test)
# from sklearn import metrics
# print metrics.classification_report(y_test, credit_pred)
# print metrics.confusion_matrix(y_test, credit_pred)
# print metrics.accuracy_score(y_test, credit_pred)

# class_weights = {1:1, 2:4}
# credit_model_cost = DecisionTreeClassifier(max_depth=6,class_weight = class_weights)
# credit_model_cost.fit(X_train, y_train)
# credit_pred_cost = credit_model_cost.predict(X_test)
# print metrics.classification_report(y_test, credit_pred_cost)
# print metrics.confusion_matrix(y_test, credit_pred_cost)
# print metrics.accuracy_score(y_test, credit_pred_cost)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
sc.fit(X_train)#计算样本的均值和标准差
X_train_std=sc.transform(X_train)
X_test_std=sc.transform(X_test)
print X_train_std.shape
print X_test_std.shape
from sklearn.linear_model.logistic import LogisticRegression
lr=LogisticRegression(C=1000.0,random_state=0)
lr.fit(X_train_std,y_train)
# 模型预测
y_pred = lr.predict_proba(X_test_std)
print (y_pred)


from matplotlib.colors import ListedColormap
# 绘制决策边界
def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    # 设置标记点和颜色
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # 绘制决策面
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    # 绘制所有样本
    X_test, y_test = X[test_idx, :], y[test_idx]
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)
    # 高亮预测样本
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1], c='', alpha=1.0, linewidths=1, marker='o', s=55, label='test set')


# X_combined_std = np.vstack((X_train_std, X_test_std))
# print X_combined_std.shape
# y_combined = np.hstack((y_train, y_test))
# plot_decision_regions(X=X_combined_std, y=y_combined, classifier=lr, test_idx=range(105, 150))
# plt.xlabel('petal length[standardized]')
# plt.ylabel('petal width[standardized]')
# plt.legend(loc='upper left')
# plt.show()

# 观察正则化参数C的作用：减少正则化参数C的值相当于增加正则化的强度
# 观察：减小参数C值，增加正则化强度，导致权重系数逐渐收缩
weights, params = [], []
for c in np.arange(-5, 5, dtype=float):
    lr = LogisticRegression(C=10 ** c, random_state=0)
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10 ** c)
weights = np.array(weights)
plt.plot(params, weights[:, 0], label='petal length')
plt.plot(params, weights[:, 1], label='petal width', linestyle='--')
plt.ylabel('weight coefficient')
plt.xlabel('C')
plt.legend(loc='upper left')
plt.xscale('log')
plt.show()



