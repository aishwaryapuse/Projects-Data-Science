#!/usr/bin/env python
# coding: utf-8

# ### Importing the Libraries

# In[1]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


cn = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm','Species']


# ### Loading the dataset

# In[3]:


iris = pd.read_csv("C:\\Users\\aishw\\Downloads\\iris\\iris.data", names= cn)


# In[4]:


# view the data
iris.head()


# In[5]:


# view the columns
iris.columns


# In[6]:


# view the shape
iris.shape


# In[7]:


# describe the data
iris.describe()


# In[8]:


# basic information
iris.info()


# In[9]:


# correlation plot-EDA
iris.corr()


# In[10]:


iris['Species'].value_counts()


# In[11]:


type(iris)


# In[12]:


# check for null values
iris.isnull().sum()


# ### Exploratory Data Analysis (EDA)

# In[13]:


# Histogram
iris['SepalLengthCm'].hist()


# In[14]:


iris['SepalWidthCm'].hist(color='green')


# In[15]:


iris['PetalLengthCm'].hist(color='red')


# In[16]:


iris['PetalWidthCm'].hist(color='orange')


# In[17]:


# Relationship between Species and Sepal length
# boxplot
plt.figure(figsize=(15,8))
sns.boxplot(x='Species',y='SepalLengthCm',data=iris.sort_values('SepalLengthCm',ascending=False))


# In[18]:


#  Relationship between sepal width and sepal length
#jointplot
sns.jointplot(x='SepalLengthCm',y='SepalWidthCm',data=iris,size=5)


# In[19]:


# Scatter plot
colors = ['red','orange','blue']
species =['Iris-setosa','Iris-versicolor','Iris-virginica']


# In[20]:


for i in range(3):
    x = iris[iris['Species']== species[i]]
    plt.scatter(x['SepalLengthCm'],x['SepalWidthCm'],c=colors[i],label=species[i])
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.legend()


# In[21]:


for i in range(3):
    x = iris[iris['Species']== species[i]]
    plt.scatter(x['PetalLengthCm'],x['PetalWidthCm'],c=colors[i],label=species[i])
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.legend()


# In[22]:


for i in range(3):
    x = iris[iris['Species']== species[i]]
    plt.scatter(x['SepalLengthCm'],x['PetalLengthCm'],c=colors[i],label=species[i])
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.legend()


# In[23]:


for i in range(3):
    x = iris[iris['Species']== species[i]]
    plt.scatter(x['SepalWidthCm'],x['PetalWidthCm'],c=colors[i],label=species[i])
plt.xlabel("Sepal Width")
plt.ylabel("PetalWidth")
plt.legend()


# In[24]:


# correlation
corr = iris.corr()
fig, ax = plt.subplots(figsize=(5,4))
sns.heatmap(corr, annot=True, ax=ax,cmap='coolwarm')


# In[25]:


# histogram
iris.hist(figsize=(20,14),color="y",edgecolor='black')
plt.show()


# In[26]:


# Display an correlation between each features
sns.pairplot(iris, hue='Species', size=3)
plt.show()


# In[27]:


iris.plot.density()


# ### Data Preprocessing

# In[28]:


from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()


# In[29]:


iris['Species']= le.fit_transform(iris['Species'])
iris.head()


# ### Model Evaluation
#             *Training & Testing Data

# In[30]:


from sklearn.model_selection import train_test_split
x=iris.drop(columns=['Species'])
y= iris['Species']


# In[31]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.30,random_state=0)


# In[32]:


x_train


# In[33]:


y_train


# In[34]:


x_test


# In[35]:


y_test


# In[36]:


print(len(x_train))
print(len(x_test))
print(len(y_train))
print(len(y_test))


# ### Models

# In[37]:


# KneighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
knc = KNeighborsClassifier()
knc.fit(x_train, y_train)
acc_nc = knc.score(x_test, y_test) * 100
print("Accuracy (KneighborsClassifier): ", acc_nc)


# In[38]:


# KNeighborsRegressor
from sklearn.neighbors import KNeighborsRegressor
knr = KNeighborsRegressor(n_neighbors=7)
knr.fit(x_train,y_train)
acc_nr = knr.score(x_test, y_test) * 100
print("Accuracy (KneighborsRegressor): ", acc_nr)


# In[39]:


# Logistic Regression 
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train, y_train)
acc_lr = lr.score(x_test, y_test) * 100
print("Accuracy (Logistic Regression): ", acc_lr)


# In[40]:


# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rfc =RandomForestClassifier()
rfc.fit(x_train,y_train)
acc_rfc=rfc.score(x_test,y_test)*100
print("Accuracy (Random Forest Classifier): ",acc_rfc)


# In[41]:


# Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
dtc =DecisionTreeClassifier()
dtc.fit(x_train,y_train)
acc_dtc=dtc.score(x_test,y_test)*100
print("Accuracy (Decision Tree): ",acc_dtc)


# In[42]:


# Support Vector machine
from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train,y_train)
acc_svc=svc.score(x_test,y_test)*100
print("Accuracy (SVC): ",acc_svc)


# In[43]:


# Naive bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(x_train,y_train)
acc_gnb=gnb.score(x_test,y_test)*100
print("Accuracy (Naive Bayes): ",acc_gnb)


# In[44]:


# Visualising the accuracy

plt.figure(figsize=(12,6))
model_acc = [acc_nc,acc_nr,acc_lr,acc_rfc,acc_dtc,acc_svc,acc_gnb]
model_name = ['KneighborsClassifier','KneighborsRegressor','Logistic Regression','Random Forest Classifier','Decision Tree','SVC','Naive Bayes']
plt.xlabel("Accuracy")
plt.ylabel("Models")
sns.barplot(x=model_acc, y=model_name, palette='plasma')


# In[ ]:





# In[ ]:




