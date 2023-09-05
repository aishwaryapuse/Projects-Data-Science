#!/usr/bin/env python
# coding: utf-8

# ### Importing the Libraries

# In[1]:


import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense , LSTM
from sklearn.metrics import mean_squared_error


# In[2]:


# load the data
Dataset_link='https://raw.githubusercontent.com/mwitiderrick/stockprice/master/NSE-TATAGLOBAL.csv'


# In[3]:


Stockprice = pd.read_csv(Dataset_link, parse_dates=True)
Stockprice.reset_index()


# In[4]:


Stockprice.head(10)


# In[5]:


Stockprice.columns


# In[6]:


Stockprice.shape


# In[7]:


Stockprice.info()


# In[8]:


Stockprice.sample(5)


# In[9]:


Stockprice.describe()


# In[10]:


Stockprice.isnull().sum()


# In[11]:


plt.figure(figsize=(8,4))
Stockprice['Open'].plot(kind='line',figsize=(9,6),color='y',label="Opening Price")
plt.ylabel("Price")
plt.legend(loc="upper left")
plt.title("Change in opening price over the years")
plt.grid()


# In[12]:


# 
plt.figure(figsize=(10,6))
Stockprice['Close'].plot(kind='line',figsize=(10,4),color='r',label="Closing Price")
plt.ylabel("Price")
plt.legend(loc="upper right")
plt.title("Change in closing price over the years")
plt.grid()


# In[13]:


sp1=Stockprice.reset_index()['Close']
sp1


# In[14]:


plt.figure(figsize=(5,4))
sns.heatmap(Stockprice.corr(),annot=True,cmap='coolwarm')


# In[15]:


plt.figure(figsize=(11,5))
plt.subplot(1,2,1)
sns.boxplot(data=Stockprice,y='Total Trade Quantity',color='yellow')
plt.subplot(1,2,2)
sns.boxplot(data=Stockprice,y='Turnover (Lacs)',color='blue')


# In[16]:


fig=plt.figure(figsize=(7,6))
plt.scatter(Stockprice['Total Trade Quantity'],Stockprice['Turnover (Lacs)'], alpha=0.5, edgecolor='b', color='orange')
plt.xlabel("Trade Quantity (in 100000)")
plt.ylabel("Turnover (in lacs)")
plt.title(" Selling Units Vs Turnover")
plt.show()


# In[17]:


training_set= Stockprice[['Open']]
training_set=pd.DataFrame(training_set)
training_set


# In[18]:


scaler=MinMaxScaler(feature_range=(0,1))
training_set_scaler=scaler.fit_transform(np.array(sp1).reshape(-1,1))


# In[19]:


training_set_scaler


# In[22]:


train_size1= int(len(training_set_scaler)*0.65)
test_size1=int(len(training_set_scaler))-train_size1
train_data1,test_data1=training_set_scaler[0:train_size1,:],training_set_scaler[train_size1:len(Stockprice),:1] 


# In[23]:


train_size1


# In[24]:


def create_dataset(dataset,time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)


# In[25]:


time_step=100
x_train, y_train=create_dataset(train_data1, time_step)
x_test, y_test= create_dataset(test_data1, time_step)


# In[26]:


print(x_train.shape,y_train.shape)


# In[27]:


x_test.shape


# In[28]:


y_test.shape


# In[29]:


x_train = x_train.reshape(x_train.shape[0],x_train.shape[1] , 1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1] , 1)


# In[30]:


model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(100,1)))
model.add(LSTM(50, return_sequences=True, input_shape=(100,1)))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam', metrics='acc')


# In[31]:


model.summary()


# In[ ]:


model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs = 75, batch_size = 64, verbose = 1)


# In[32]:


train_predict1=model.predict(x_train)
test_predict1=model.predict(x_test)
#Transformback to original form
train_predict1=scaler.inverse_transform(train_predict1)
test_predict1=scaler.inverse_transform(test_predict1)


# In[33]:


math.sqrt(mean_squared_error(y_train,train_predict1))


# In[34]:


math.sqrt(mean_squared_error(y_test,test_predict1))


# In[37]:


### Plotting 
# shift train predictions for plotting
look_back=100
trainPredictPlot = np.empty_like(training_set_scaler)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict1)+look_back, :] = train_predict1

# shift test predictions for plotting
testPredictPlot = np.empty_like(training_set_scaler)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict1)+(look_back*2)+1:len(sp1)-1, :] = test_predict1

# plot baseline and predictions
plt.figure(figsize=(12,5))
plt.plot(scaler.inverse_transform(training_set_scaler))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()


# In[ ]:




