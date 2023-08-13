#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# # Storing the dataset into a variable and Exploratory data analysis

# In[2]:


dataset = pd.read_csv('temperature_data.CSV')


# In[3]:


dataset.head()


# In[4]:


# shape of the data
dataset.shape


# In[5]:


dataset.describe()


# In[6]:


# To see the data type of each of the variable, number of values entered in each of the variables
dataset.info()


# In[7]:


# Checking For Null Values
dataset.isnull().sum()


# In[8]:


dataset['motor_speed'].mode()


# In[9]:


# Checking the stats of the data
dataset.describe().transpose()


# # From the above analysis
# 
#  Minimum value of motor_speed is -1.371529
#  25% of the data lie below motor_speed = -0.951892
#  50% of the data lie below motor_speed = -0.140246
#  75% of the data lie below motor_speed = 0.853584
#  Max value of motor_speed is 2.024164

# In[10]:


# Create boxplot for column="motor_speed"
dataset.boxplot(column="motor_speed",return_type='axes',figsize=(9,8))
plt.text(x=0.78, y=0.853, s="3rd Quartile")
plt.text(x=0.8, y=-0.140, s="Median")
plt.text(x=0.78, y=-0.951, s="1st Quartile")
plt.text(x=0.9, y=-1.371, s="Min")
plt.text(x=0.9, y=2.02, s="Max")

plt.show()


# In[11]:


## As profile_id is not important because it tells the uniqueness of session only,not so important in prediction.
## so we can drop it
dataset.drop('profile_id',axis=1,inplace=True)


# In[12]:


plt.figure(figsize=(15,7))
dataset.motor_speed.hist(bins=100)
plt.show()


# ## Separating the features and labels and Motor_speed IS TEMPERATURE CONSIDERED AS FEATURE VARIABLE

# In[13]:


x = dataset.drop('motor_speed',axis=1)   ## independent feature
y = dataset['motor_speed']


# ## Train_Test Split

# In[14]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=100)


# In[15]:


## Scaling the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


# In[16]:


## Training the dataset
from sklearn.linear_model import  Ridge, Lasso, LinearRegression, SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor,  AdaBoostRegressor, BaggingRegressor
from xgboost import XGBRegressor, XGBRFRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error


# In[17]:


def predict(model):
    # Define Models Name
    print('Model: {}'.format(model))
    ## fitting the model
    model.fit(x_train_scaled,y_train)
    ## predicting the value
    y_pred = model.predict(x_test_scaled)
    print('R2 score',r2_score(y_test,y_pred))
    print('MAE',mean_absolute_error(y_test,y_pred))
    print('RMSE:{}'.format(np.sqrt(mean_squared_error(y_test,y_pred))))
    
    sns.distplot(y_test-y_pred)
    plt.show()


# In[18]:


predict(LinearRegression())
plt.show()


# In[19]:


predict(Ridge())
plt.show()


# In[20]:


predict(Lasso())
plt.show()


# In[21]:


predict(KNeighborsRegressor())
plt.show()


# In[22]:


predict(DecisionTreeRegressor())
plt.show()


# In[23]:


predict(XGBRegressor())
plt.show()


# ## XgbRegressor gives the r2 score of 0.9995 Selecting this as our final model

# In[24]:


model = XGBRegressor()
model.fit(x_train_scaled,y_train)


# In[25]:


y_pred = model.predict(x_test)


# In[26]:


y_pred


# In[27]:


x_test['predicted_temperature'] = y_pred


# In[28]:


x_test.head()


# In[29]:


dataset.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8);
plt.show()


# In[30]:


sns.pairplot(dataset)
plt.show()


# In[31]:


dataset.corr()


# In[32]:


sns.heatmap(dataset.corr())
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




