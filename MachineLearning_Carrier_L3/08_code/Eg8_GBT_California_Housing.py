
# coding: utf-8

# ## Use-case: California Housing
# 
# This use-case study shows how to apply GBRT to a real-world dataset. The task is to predict the log median house value for census block groups in California. The dataset is based on the 1990 censues comprising roughly 20.460 groups(1990 census block groups). There are 8 features for each group including: median income, average house age, latitude, and longitude.
# 
# -  The response variable Y is the median house value in each neighborhood measured in units of $100,000.
# -  8 features : The predictor variables are demographics such as median income MedInc, housing density as reflected by the number of houses House, and the average occupancy in each house AveOccup. Also included as predictors are the location of each neighborhood (longitude and latitude), and several quantities reflecting the properties of the houses in the neighborhood: average number of rooms AveRooms and bedrooms AveBedrms. There are thus a total of eight predictors, all numeric.

# Some of the aspects that make this dataset challenging are: 
# - a) heterogenous features (different scales and distributions) and
# - b) non-linear feature interactions (specifically latitude and longitude).
# - c) Furthermore, the data contains some extreme values of the response (log median house value) -- such a dataset strongly benefits from robust regression techniques such as huberized loss functions.
# 
# Below you can see histograms for some of the features and the response. You can see that they are quite different: median income is left skewed, latitude and longitude are bi-modal, and log median house value is right skewed.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingRegressor


# In[2]:


#you can load the data from local
#get the domain or header of the cal housing data
cal_housing_domain = pd.read_csv(".\data\CaliforniaHousing\cal_housing.domain",
                                 header =None,delimiter=":")
header = list(cal_housing_domain.loc[:,0])

#build the pd dataframe from cal housing data
cal_housing = pd.read_csv(".\data\CaliforniaHousing\cal_housing.data",names=header)


# In[3]:


#show the head of the data
cal_housing.head(5)


# In[4]:


#add columns to the df
cal_housing["AveRooms"] = cal_housing["totalRooms"]/cal_housing["households"]
cal_housing["AveBedrms"] = cal_housing["totalBedrooms"]/cal_housing["households"]
cal_housing["AveOccup"] = cal_housing["population"]/cal_housing["households"]
#select the columns
cal_housing = cal_housing[['medianIncome','housingMedianAge','AveRooms',
                     'AveBedrms', 'population','AveOccup',
                    'latitude', 'longitude', 'medianHouseValue']]
#change the names of columns
cal_housing.columns = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population',
                       'AveOccup', 'Latitude', 'Longitude','MedianHouseValue']


# In[5]:


cal_housing.head()


# # Data exploring

# In[6]:


print("The shape of the data is {}".format(cal_housing.shape))
print("The features of data are:{}".format(cal_housing.columns.astype(list)[:-1]))


# In[7]:


cal_housing.info()


# In[8]:


cal_housing.describe()


# In[9]:


print(cal_housing.isnull().sum())


# 
# ## Exploring the distribution of one variables :Histogram

# In[10]:


# The distribution of housingMedianAge
_ = cal_housing.hist(column=['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms','Latitude', 
                             'Longitude','MedianHouseValue'],
                     figsize=(12,8),bins=10)


# In[11]:


cal_housing[["AveRooms",'AveBedrms']].plot.box(vert = False)


# In[12]:


np.log(cal_housing[['MedianHouseValue']]).plot.box(vert = False)


# ## Exploring the relation between two variables :Scatterplot

# In[13]:


#The relation betwen the feature and the output value
plt.scatter(cal_housing.MedInc,cal_housing.MedianHouseValue)


# In[14]:


#The relation betwen the feature and the output value
plt.scatter(cal_housing.AveRooms,cal_housing.MedianHouseValue)


# In[15]:


pd.plotting.scatter_matrix(cal_housing,diagonal="kde",figsize = (24,16))
plt.show()


# # Training a model

# In[16]:


from sklearn.metrics import mean_absolute_error


# In[17]:


#get the data and target array from dataframe
data = cal_housing.iloc[:,0:-1].as_matrix()
target= cal_housing.iloc[:,-1].as_matrix()
#split the data into two sets
X_train,X_test,y_train,y_test = train_test_split(data,np.log(target),
                                test_size = 0.2,random_state=123)


# ### Regularization
# 
# GBRT provide three knobs to control overfitting: tree structure, shrinkage, and randomization.
# 
# #### Tree Structure
# 
# The depth of the individual trees is one aspect of model complexity. The depth of the trees basically control ***the degree of feature interactions*** that your model can fit. 
# 
# For example, if you want to capture the interaction between a feature ``latitude`` and a feature ``longitude`` your trees need a depth of at least two to capture this. Unfortunately, the degree of feature interactions is not known in advance but it is usually fine to assume that it is faily low -- in practise, a depth of 4-6 usually gives the best results. In scikit-learn you can constrain the depth of the trees using the ***``max_depth``*** argument.
# 
# Another way to control the depth of the trees is by enforcing a lower bound on the number of samples in a leaf: this will avoid inbalanced splits where a leaf is formed for just one extreme data point. In scikit-learn you can do this using the argument ***``min_samples_leaf``***. This is effectively a means to introduce bias into your model with the hope to also reduce variance as shown in the example below:
# 
# #### Shrinkage
# 
# The most important regularization technique for GBRT is shrinkage: the idea is basically to do slow learning by shrinking the predictions of each individual tree by some small scalar, the ``learning_rate``. By doing so the model has to re-enforce concepts. A lower ``learning_rate`` requires a higher number of ``n_estimators`` to get to the same level of training error -- so its trading runtime against accuracy.
# 
# #### Stochastic Gradient Boosting
# 
# Similar to ``RandomForest``, introducing randomization into the tree building process can lead to higher accuracy. Scikit-learn provides two ways to introduce randomization:<br>
# a) subsampling the training set before growing each tree (***``subsample``***) and <br>
# b) subsampling the features before finding the best split node (***``max_features``***).<br>
# Experience showed that the latter works better if there is a sufficient large number of features (>30).
# One thing worth noting is that both options reduce runtime.
# 
# Below we show the effect of using ``subsample=0.5``, ie. growing each tree on 50% of the training data,  on our toy example:

# ### Hyperparameter tuning
# 
# We now have introduced a number of hyperparameters -- as usual in machine learning it is quite tedious to optimize them. Especially, since they interact with each other (``learning_rate`` and ``n_estimators``, ``learning_rate`` and ``subsample``, ``max_depth`` and ``max_features``).
# 
# We usually follow this recipe to tune the hyperparameters for a gradient boosting model:
# 
#   1. Choose ``loss`` based on your problem at hand (ie. target metric)
#   2. Pick ``n_estimators`` as large as (computationally) possible (e.g. 3000).
#   3. Tune ``max_depth``, ``learning_rate``, ``min_samples_leaf``, and ``max_features`` via grid search.
#   4. Increase ``n_estimators`` even more and tune ``learning_rate`` again holding the other parameters fixed.
#     
# Scikit-learn provides a convenient API for hyperparameter tuning and grid search:

# In[19]:


from sklearn.grid_search import GridSearchCV


# In[20]:


param_grid = {'max_depth': [4, 6],
              'min_samples_leaf': [3, 5, 9],
              'learning_rate': [0.1, 0.05, 0.02, 0.01]}

estimator = GradientBoostingRegressor(loss='huber',
                n_estimators=1000,random_state=112,subsample=0.5)

grid_cv = GridSearchCV(estimator,param_grid, n_jobs=4).fit(X_train,y_train)


# In[21]:


# best hyperparameter setting
best_params = grid_cv.best_params_
best_params


# In[22]:


#training a estimator with best hyperparameters
best_estimator  = GradientBoostingRegressor(loss ='huber',n_estimators=1000,
                                            random_state=113,subsample=0.5,
                                            learning_rate=0.05,max_depth = 6,
                                            min_samples_leaf = 5)
best_estimator.fit(X_train,y_train)


# In[23]:


#calculte the mean absolute error
y_pred = best_estimator.predict(X_test)
mae = mean_absolute_error(y_test,y_pred)
print('MAE: %.4f' % mae)


# ### Feature importance
# 
# Often features do not contribute equally to predict the target response. When interpreting a model, the first question usually is: 
# - what are those important features and how do they contributing in predicting the target response?
# 
# A GBRT model derives this information from the fitted regression trees which intrinsically perform feature selection by choosing appropriate split points. You can access this information via the instance attribute ``est.feature_importances_``.

# In[24]:


# sort importances
indices = np.argsort(best_estimator.feature_importances_)
names = cal_housing.columns.tolist()[0:-1]
# plot as bar chart
plt.barh(np.arange(len(names)), best_estimator.feature_importances_[indices])
plt.yticks(np.arange(len(names)) + 0.25, np.array(names)[indices])
_ = plt.xlabel('Relative importance')

