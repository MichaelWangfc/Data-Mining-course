
# coding: utf-8

# ## 业务场景

# 判断是否潜在合约客户 vs 单卡客户：
# 

# ## 字段信息：
# 名称       | 说明   | 类型  |
# --------------|---------|--------|-----
# user_id     |用户标识 | int   |
# service_kind  |业务类型 | string | 2G\3G\4G
# call_duration|主叫时长（分）|
# called_duration|被叫时长（分）
# in_package_flux|免费流量
# out_package_flux|计费流量
# |月均上网时长（分）|
# net_duration |入网时长（天） |long
# last_recharge_value|最近一次缴费金额（元）
# total_recharge_value|总缴费金额(元)|
# total_recharge_count|缴费次数
# contractuser_flag|是否潜在合约用户
# 
# 
# silent_serv_flag|是否三无用户|int|0：否，1：是，三无：无月租费，无最低消费，无来电显示
# 
# pay_type    | 付费类型 |int   | 0：预付费，1：后付费
# 
# 

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn import datasets

from sklearn import svm,metrics
from sklearn.model_selection import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import OneHotEncoder

from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题


# ## 读取运营商数据

# In[2]:


data= pd.read_csv('data_carrier_svm.csv',encoding ='utf8')
data.head()


# ## 数据探索

# In[3]:


#不同用户的主叫时长分布情况对比
cond = data['是否潜在合约用户']==1
data[cond]['主叫时长（分）'].hist(alpha =0.5,label='潜在合约用户')
data[~cond]['主叫时长（分）'].hist(color='r',alpha = 0.5,label='非潜在合约用户')
plt.legend()


# In[4]:


#不同用户的被叫时长分布情况对比
cond = data['是否潜在合约用户']==1
data[cond]['被叫时长（分）'].hist(alpha =0.5,label='潜在合约用户')
data[~cond]['被叫时长（分）'].hist(color='r',alpha = 0.5,label='非潜在合约用户')
plt.legend()


# In[5]:


#不同用户的业务类型情况对比
grouped =data.groupby(['是否潜在合约用户','业务类型'])['用户标识'].count().unstack()
grouped.plot(kind= 'bar',alpha =1.0,rot = 0)


# ## 数据预处理

# In[6]:


#分割特征数据集和便签数据集
X = data.loc[:,'业务类型':'余额']
y= data.loc[:,'是否潜在合约用户']
print('The shape of X is {0}'.format(X.shape))
print('The shape of y is {0}'.format(y.shape))


# In[7]:


X.head()


# ### 类别特征编码

# In[8]:


#自定义转换函数
def service_mapping(cell):
    if cell=='2G':
        return 2
    elif cell=='3G':
        return 3
    elif cell=='4G':
        return 4

#将业务类型的string型值映射为整数型
service_map = X['业务类型'].map(service_mapping)
service = pd.DataFrame(service_map)

#使用OncHotEncoder转化类型特征为0/1编码的多维特征
enc = OneHotEncoder()
service_enc = enc.fit_transform(service).toarray()
service_enc

#0/1编码的多维特征的名称
service_names = enc.active_features_.tolist()
service_newname=[str(x)+'G' for x in service_names]

service_df = pd.DataFrame(service_enc,columns=service_newname)
service_df.head()
X_enc = pd.concat([X,service_df],axis = 1).drop('业务类型',axis=1)
X_enc.head()


# ### 数据归一化

# In[18]:


from sklearn.preprocessing import normalize
X_normalized = normalize(X_enc)


# In[10]:


#分割训练集和测试集
X_train,X_test,y_train,y_test = train_test_split(X_normalized,y,test_size = 0.2, random_state=112)
print('The shape of X_train is {0}'.format(X_train.shape))
print('The shape of X_test is {0}'.format(X_test.shape))


# In[11]:


#生成数据可视化
plt.scatter(X_train[:,0],X_train[:,1],c=y_train)


# ### 训练简单模型: 线性超平面模型

# In[12]:


#模型实例化
linear_clf = svm.LinearSVC()
#在训练集上训练模型
linear_clf.fit(X_train,y_train)

#在测试集上预测
y_pred = linear_clf.predict(X_test)

#计算准备；率
score = metrics.accuracy_score(y_test,y_pred)
print('The accuracy score of the model is: {0}'.format(score))

#查看混淆举证
metrics.confusion_matrix(y_test,y_pred)


# ### 训练核函数=rbf的SVM算法：调参

# In[13]:


# 设置调试参数的范围
C_range = np.logspace(-5,5,5)
gamma_range = np.logspace(-9,2,10)

clf = svm.SVC(kernel='rbf',cache_size=1000,random_state=117)
param_grid = {'C':C_range,
             'gamma':gamma_range}

# GridSearch作用在训练集上
grid = GridSearchCV(clf,param_grid=param_grid,scoring= 'accuracy',n_jobs=2,cv =5)
grid.fit(X_train,y_train)


# In[14]:


# examine the best model
print(grid.best_score_)
print(grid.best_params_)
print(grid.best_estimator_)


# ## 使用最优超参再次进行训练

# In[15]:


#instance with the best parameters
clf_best = svm.SVC(kernel="rbf",C=grid.best_params_['C'],
                   gamma = grid.best_params_['gamma'],probability=True)

#fit on the trainingt data
clf_best.fit(X_train,y_train)

#predict on the testing data
y2_pred = clf_best.predict(X_test)


# ### 模型评估

# In[16]:


#caculate the accuracy score on testing data
accuracy = metrics.accuracy_score(y_test,y2_pred)
print("The accuracy is %f"%accuracy)

#get the confusion matrics
metrics.confusion_matrix(y_test,y2_pred)


# In[17]:


# store the predicted probabilities for class 1
y2_pred_prob = clf_best.predict_proba(X_test)[:, 1]

# IMPORTANT: first argument is true values, second argument is predicted probabilities
#fpr: false positive rate (=1- specifity), tpr = true postive rate
fpr, tpr, thresholds = metrics.roc_curve(y_test, y2_pred_prob)
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC curve for diabetes classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)

