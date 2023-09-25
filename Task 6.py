#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Loading Libraries
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# Load the Data
df = pd.read_csv("C:\\Users\\aishw\\Downloads\\Iris (1).csv")


# In[3]:


df.head()


# In[4]:


# delete a column
df=df.drop(columns=['Id'])


# In[5]:


df.head()


# In[6]:


df.describe()


# In[7]:


df.info()


# In[8]:


df.columns


# In[9]:


# to display no. of sample on each class
df['Species'].value_counts()


# In[10]:


df.isnull().sum()


# In[11]:


df.corr()


# In[12]:


corr = df.corr()
fig, ax = plt.subplots(figsize=(5,4))
sns.heatmap(corr, annot=True, ax=ax,cmap='coolwarm')


# ### Data Preprocessing

# In[13]:


from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()


# In[14]:


df['Species']= le.fit_transform(df['Species'])
df.head()


# ### Model Evaluation
# 
#         *Training & Testing Data

# In[15]:


from sklearn.model_selection import train_test_split


# In[16]:


x=df.drop(columns=['Species'])
y= df['Species']


# In[17]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.30)


# In[18]:


x_train


# In[19]:


y_train


# In[20]:


x_test


# In[21]:


y_test


# In[22]:


print(len(x_train))
print(len(x_test))
print(len(y_train))
print(len(y_test))


# ### Decision Tree Algorithm

# In[23]:


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()


# In[24]:


model.fit(x_train,y_train)


# In[25]:


# print metric to get performance
print("Accuracy:",model.score(x_test,y_test)*100)


# In[ ]:





# In[26]:


df.plot.density()


# In[27]:


# histogram
df.hist(figsize=(20,12),color="orange",edgecolor='black')
plt.show()


# In[28]:


# Display an correlation between each features
sns.pairplot(df, hue='Species', size=3)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[24]:


# Logistic Regression
from sklearn.linear_model import LogisticRegression
model= LogisticRegression()


# In[25]:


model.fit(x_train,y_train)


# In[26]:


print("Accuracy:",model.score(x_test,y_test)*100)


# In[30]:


# knn
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()


# In[31]:


model.fit(x_train,y_train)


# In[32]:


# print metric to get performance
print("Accuracy:",model.score(x_test,y_test)*100)


# In[ ]:




