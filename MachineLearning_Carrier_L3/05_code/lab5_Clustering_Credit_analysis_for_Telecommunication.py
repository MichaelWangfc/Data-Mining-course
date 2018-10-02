
# coding: utf-8

# 本案例使用电信用户的通信行为数据集，进行用户信用分群和分析。由于是没有标注的训练样本，使用降维和聚类等无监督方法将用户进行分群。<br>
# 然后对不同群体数据进行人工分析，确定群体的信用行为特点。

# ## step1: 数据读取

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pylab as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


X = pd.read_csv('data/telecom.csv',encoding='utf-8')
print(X.shape)
X.head()


# ## step 2 : 数据标准化

# In[3]:


from sklearn import preprocessing
X_scaled = preprocessing.scale(X)
X_scaled[0:5]


# ## step 3: 进行PCA数据降维

# In[4]:


from sklearn.decomposition import PCA

#生成PCA实例
pca = PCA(n_components=3)
#进行PCA降维
X_pca = pca.fit_transform(X_scaled)
#生成降维后的dataframe
X_pca_frame = pd.DataFrame(X_pca,columns=['pca_1','pca_2','pca_3'])
X_pca_frame.head()


# In[5]:


#查看这三个维度坐标轴（主成分）与原始的7个维度坐标轴之间的关系：
pd.DataFrame(pca.components_,columns = X.columns,index =['pca_1','pca_2','pca_3'])


# ## step 4: K-means 聚类

# ### 训练简单模型

# In[6]:


from sklearn.cluster import KMeans

#KMeans算法实例化，将其设置为K=10
est = KMeans(n_clusters=10)

#作用到降维后的数据上
est.fit(X_pca)


# In[7]:


#取出聚类后的标签
kmeans_clustering_labels = pd.DataFrame(est.labels_,columns=['cluster'])

#生成有聚类后dataframe
X_pca_frame = pd.concat([X_pca_frame,kmeans_clustering_labels],axis =1)

X_pca_frame.head()


# ### 对不同的k值进行计算，筛选最优的k值

# In[23]:


from mpl_toolkits.mplot3d import Axes3D
from sklearn import metrics


# In[73]:


#KMeans算法实例化，将其设置为K=range（2,14）
d = {}
fig_reduced_data = plt.figure(figsize= (12,12))
for k in range(2,14):
    est = KMeans(n_clusters=k,random_state=111)
    #作用到降维后的数据上
    y_pred = est.fit_predict(X_pca)
    #评估不同k值聚类算法效果
    score = metrics.calinski_harabaz_score(X_pca_frame,y_pred)
    d.update({k:calinski_harabaz_score})
    print('calinski_harabaz_score with k={0} is {1}'.format(k,score))
    #生成三维图形，每个样本点的坐标分别是三个主成分的值  
    ax= plt.subplot(4,3,k-1,projection = '3d')
    ax.scatter(X_pca_frame.pca_1,X_pca_frame.pca_2,X_pca_frame.pca_3,c= y_pred)
    ax.set_xlabel('pca_1')
    ax.set_ylabel('pca_2')
    ax.set_zlabel('pca_3')


# In[62]:


#绘制不同k值对应的score，找到最优的k值
x = []
y = []
for k,score in d.iteritems():
    x.append(k)
    y.append(score)   

plt.plot(x,y)
plt.xlabel('k value')
plt.ylabel('calinski_harabaz_score')


# ### 步骤5	样本筛选

# In[8]:


X.index = X_pca_frame.index
#合并原数据和三个主成分的数据
X_full = pd.concat([X,X_pca_frame],axis = 1)
X_full.head()


# In[9]:


#按每个聚类分组
grouped = X_full.groupby('cluster')
result_data  = pd.DataFrame()
#对分组做循环，分别对每组进行去除异常值处理
for name,group in grouped:
    #每组去除异常值前的个数
    print('Group:{0}, Samples before:{1}'.format(name,group['pca_1'].count()))
    desp = group[['pca_1','pca_2','pca_3']].describe()
    for att in ['pca_1','pca_2','pca_3']:
        lower25 =desp.loc['25%',att]
        upper75 =desp.loc['75%',att]
        IQR = upper75 - lower25
        min_value = lower25-1.5*IQR
        max_value = upper75 +1.5*IQR
        #使用统计中的1.5*IQR法则，删除每个聚类中的噪音和异常点
        group = group[(group[att]>min_value)&(group[att]<max_value)]
    result_data = pd.concat([result_data,group],axis = 0)
    #每组去除异常值前的个数
    print('Group:{0}, Samples after:{1}'.format(name,group['pca_1'].count()))
print('Remain sample:',result_data['pca_1'].count())


# ### 聚类效果查看
# 

# #### 原始数据降维后的可视化

# In[10]:


from mpl_toolkits.mplot3d import Axes3D

#生成三维图形，每个样本点的坐标分别是三个主成分的值
fig_reduced_data = plt.figure()
ax_reduced_data = plt.subplot(111,projection = '3d')
ax_reduced_data.scatter(X_pca_frame.pca_1.values,X_pca_frame.pca_2.values,
                        X_pca_frame.pca_3.values)
ax_reduced_data.set_xlabel('Component_1')
ax_reduced_data.set_ylabel('Component_2')
ax_reduced_data.set_zlabel('Component_3')


# #### 聚类算法之后的不同簇数据的可视化：

# In[11]:


#设置每个簇对应的颜色
cluster_2_color = {0:'red',1:'green',2:'blue',3:'yellow',4:'cyan',
                  5:'black',6:'magenta',7:'#fff0f5',8:'#ffdab9',9:'#ffa500'}

colors_clustered_data = X_pca_frame.cluster.map(cluster_2_color)
fig_clustered_data = plt.figure()
ax_clustered_data = plt.subplot(111,projection = '3d')

#聚类算法之后的不同簇数据的映射为不同颜色：
ax_clustered_data.scatter(X_pca_frame.pca_1.values,X_pca_frame.pca_2.values,
                          X_pca_frame.pca_3.values,c=colors_clustered_data)
ax_clustered_data.set_xlabel('Component_1')
ax_clustered_data.set_ylabel('Component_2')
ax_clustered_data.set_zlabel('Component_3')


# ### 3.筛选后的数据聚类可视化
# 

# In[12]:


colors_filtered_data = result_data.cluster.map(cluster_2_color)
fig = plt.figure()
ax = plt.subplot(111,projection = '3d')
ax.scatter(result_data.pca_1.values,result_data.pca_2.values,
           result_data.pca_3.values,c=colors_filtered_data)
ax.set_xlabel('Component_1')
ax.set_ylabel('Component_2')
ax.set_zlabel('Component_3')


# In[13]:


### 4.每个聚类在pca_1,pca_2,pca_3上的均值的可视化 


# ### step 6. 用户行为分析（用户画像构建）

# In[14]:


#查看各族中的每月话费情况
monthly_Fare = result_data.groupby('cluster').describe().loc[:,u'每月话费']
monthly_Fare


# In[15]:


monthly_Fare[['mean','std']].plot(kind ='bar',rot = 0,legend = True)


# In[16]:


#查看各族中的入网时间情况
access_time = result_data.groupby('cluster').describe().loc[:,u'入网时间']
access_time


# In[17]:


access_time[['mean','std']].plot(kind ='bar',rot = 0,
                                 legend = True,title ='Access Time')


# In[63]:


#查看各族中的欠费金额情况
arrearage = result_data.groupby('cluster').describe().loc[:,u'欠费金额']
arrearage[['mean','std']].plot(kind ='bar',rot = 0,
                                 legend = True,title ='Arrearage')


# In[64]:





# In[65]:


#综合描述
new_column = [ 'Access_time',u'套餐价格',u'每月流量','Monthly_Fare',u'每月通话时长',
              'Arrearage',u'欠费月份数',u'pca_1', u'pca_2', u'pca_3', u'cluster']
result_data.columns = new_column
result_data.groupby('cluster')[['Monthly_Fare','Access_time','Arrearage']].mean().plot(kind = 'bar')

