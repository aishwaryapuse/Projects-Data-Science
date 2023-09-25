#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Loading Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
import matplotlib as mpl
import sklearn

from sklearn.model_selection import train_test_split


# In[2]:


# Data preparation
import os
os.getcwd()


# In[3]:


train = pd.read_csv("C:\\Users\\aishw\\Downloads\\train.csv\\train.csv")


# In[4]:


test = pd.read_csv("C:\\Users\\aishw\\Downloads\\test.csv\\test.csv")


# In[5]:


members = pd.read_csv("C:\\Users\\aishw\\Downloads\\members.csv~\\members.csv")


# In[6]:


songs = pd.read_csv("C:\\Users\\aishw\\Downloads\\songs.csv\\songs.csv")


# In[7]:


print('train',train.shape)
print('test',test.shape)
print('members',members.shape)
print('songs',songs.shape)


# In[8]:


# Understanding Data
train.head()


# In[9]:


test.head()


# In[10]:


members.head()


# In[11]:


songs.head()


# In[12]:


# Data Preprocessing


# In[13]:


#2% sample of items
train = train.sample(frac=0.5)


# In[14]:


train.shape


# In[15]:


train= pd.merge(train,songs,on='song_id',how='left')
del songs


# In[16]:


train.shape


# In[17]:


train.head()


# In[18]:



train = pd.merge(train,members,on='msno',how='left')


# In[19]:


train.shape


# In[20]:


train.head()


# In[21]:


train.info()


# In[22]:


train.describe()


# In[23]:


# count
train.isnull().sum()/train.isnull().count()*100


# In[24]:


dtypes = pd.DataFrame(train.dtypes,columns=["Data Type"])

dtypes["Unique Values"]=train.nunique().sort_values(ascending=True)

dtypes["Null Values"]=train.isnull().sum()

dtypes["% null Values"]=train.isnull().sum()/len(train)

dtypes.sort_values(by="Null Values" , ascending=False).style.background_gradient(cmap='YlOrRd',axis=0)


# In[25]:


# Visualizing null values
plt.figure(figsize=(25,10))
sns.heatmap(train.isnull(),cbar=False,cmap='viridis')
plt.tick_params(axis='x', labelsize=25)


# In[26]:


# Replace NA
for i in train.select_dtypes(include=['object']).columns:
    train[i]=train[i].fillna(value = 'unknown')

    # Numerics with mean    
for i in train.select_dtypes(exclude=['object']).columns:
    train[i] = train[i].fillna(value = train[i].mean())


# In[27]:


train.isna().sum()


# In[28]:


## Changing Data Format

# Registration_init_time
train['registration_init_time'] = pd.to_datetime(train['registration_init_time'],format='%Y%m%d',errors = 'ignore')

train['registration_init_time_year'] = train['registration_init_time'].dt.year
train['registration_init_time_month'] = train['registration_init_time'].dt.month
train['registration_init_time_day']  = train['registration_init_time'].dt.day

# exploration_date
train['expiration_date'] = pd.to_datetime(train['expiration_date'] ,format='%Y%m%d',errors = 'ignore')
train['expiration_date_year'] = train['expiration_date'].dt.year
train['expiration_date_month'] = train['expiration_date'].dt.month
train['expiration_date_day'] = train['expiration_date'].dt.day


# In[29]:


train.head()


# In[30]:


# Changing dates to category
train['registration_init_time']=train['registration_init_time'].astype('category')
train['expiration_date']= train['expiration_date'].astype('category')


# In[31]:


# Encoding & categorizating Columns

# Object data to category
for col in train.select_dtypes(include=['object']).columns:
    train[col] = train[col].astype('category')
    
# Encoding categorical features
for col in train.select_dtypes(include=['category']).columns:
    train[col] = train[col].cat.codes


# In[32]:


train.info()


# In[33]:


train.describe()


# In[34]:


train.head()


# In[35]:


# Сorrelation matrix
plt.figure(figsize=[20,10])
sns.heatmap(train.corr(), annot=True)
plt.show()


# In[36]:


# Drop Column
train = train.drop(['expiration_date','lyricist'] , 1)


# In[37]:


train.info()


# In[38]:


train.shape


# In[39]:


## Building models
# random Forest
train_1=train.copy()


# In[40]:


x_train=train_1.drop(['target'],axis=1).values
y_train=train_1['target'].values


# In[41]:


x_train.shape


# In[42]:


y_train.shape


# In[43]:


x_train,x_test,y_train,y_test=train_test_split(x_train,y_train,test_size=0.25)


# In[44]:


from sklearn.ensemble import RandomForestClassifier


# In[47]:


# Selected columns
print(train.shape)
train.columns


# In[48]:


# Сorrelation matrix
plt.figure(figsize=[20,10])
sns.heatmap(train.corr(), annot=True)
plt.show()


# In[54]:


# Create model
XGB = xgb.XGBClassifier(learning_rate=0.1, max_depth=15, min_child_weight=5)
XGB.fit()

XGB_TrainScore = XGB.score(x_train,y_train)
XGB_TrainScore


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


XGB_TestScore = XGB.score(x_test,y_test)
XGB_TestScore


# In[ ]:


y_pred_xgb =XGB.predict(test_data)


# In[ ]:


cm_lgbm = confusion_matrix(test_labels, y_pred_xgb)
sns.heatmap(cm_lgbm, annot=True, fmt='g')


# In[ ]:


print(classification_report(test_labels, y_pred_xgb))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


RF = RandomForestClassifier()
RF.fit(x_train,y_train)


# In[ ]:


clf = RandomForestClassifier(n_estimators=250, max_depth=25,random_state= 0)
clf.fit(x_train,y_train)


# In[ ]:


train_2=train.copy()


# In[ ]:


# Drop columns with importances < 0.04
train_2 = train_2.drop(train_plot.features[train_plot.importances < 0.04].tolist(), 1)


# In[ ]:


# Selected Columns
train_2.columns


# In[ ]:


## XGBoost


# In[ ]:


# Train And Test Split
train3=train.copy()


# In[ ]:


x_train1=train3.drop(['target'],axis=1).values
y_train1=train3['target'].values


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(x_train_1,y_train_1,test_size=0.25)


# In[ ]:


#create model
model2 = xgb.XGBClassifier(learning_rate=0.1, max_depth=15, min_child_weight=5)
model2.fit(x_train_1,y_train_1)


# In[ ]:


x_train_1.shape


# In[ ]:


# predicting
Prediction =model2.predict(x_test)


# In[ ]:


from skelearn import metrics


# In[ ]:


print(metrics.classification_report(y_test,Prediction))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


#Confusion matrix
import seaborn as sns 
from sklearn.metrics import confusion_matrix

cm_rf = confusion_matrix(x_train,y_train)
sns.heatmap(cm_rf, annot=True, fmt='g')


# In[ ]:


from sklearn.metrics import classification_report


# In[ ]:


print(classification_report(test_labels, y_pred_rf))


# In[ ]:


# Feature Selection


# In[ ]:


# Drop columns with importances < 0.04
df = df.drop(df_plot.features[df_plot.importances < 0.04].tolist(), 1)


# In[ ]:


# Selected columns
print(df.shape)
df.columns


# In[ ]:


# Сorrelation matrix
plt.figure(figsize=[20,10])
sns.heatmap(df.corr(), annot=True)
plt.show()


# In[ ]:


# XGBoosting


# In[ ]:


# Create model
XGB = xgb.XGBClassifier(learning_rate=0.1, max_depth=15, min_child_weight=5)
XGB.fit(train_data, train_labels)

XGB_TrainScore = XGB.score(train_data, train_labels)
XGB_TrainScore


# In[ ]:


XGB_TestScore = XGB.score(test_data, test_labels)
XGB_TestScore


# In[ ]:


y_pred_xgb =XGB.predict(test_data)


# In[ ]:


cm_lgbm = confusion_matrix(test_labels, y_pred_xgb)
sns.heatmap(cm_lgbm, annot=True, fmt='g')


# In[ ]:


print(classification_report(test_labels, y_pred_xgb))


# In[ ]:


## Light LGBM
import lightgbm as lgb
d_train = lgb.Dataset(train_data, label=train_labels)


# In[ ]:


params = {
        'objective': 'binary',
        'boosting': 'dart',
        'learning_rate': 0.2 ,
        'verbose': 0,
        'num_leaves': 100,
        'bagging_fraction': 0.95,
        'bagging_freq': 1,
        'bagging_seed': 1,
        'feature_fraction': 0.9,
        'feature_fraction_seed': 1,
        'max_bin': 256,
        'num_rounds': 100,
        'metric' : 'auc'
    }


# In[ ]:


clf = lgb.train(params, d_train, 100)


# In[ ]:


y_predtrain_lgbm=clf.predict(train_data)

y_predtest_lgbm=clf.predict(test_data)


# In[ ]:


# Making Prediction for training and test sets
for i in range(0, train_data.shape[0]):
    if y_predtrain_lgbm[i]>=.5: 
        y_predtrain_lgbm[i]=1
    else:  
        y_predtrain_lgbm[i]=0
        
for i in range(0, test_data.shape[0]):
    if y_predtest_lgbm[i]>=.5: 
        y_predtest_lgbm[i]=1
    else:  
        y_predtest_lgbm[i]=0      


# In[ ]:


from sklearn.metrics import accuracy_score


LGBM_TrainScore = accuracy_score(y_predtrain_lgbm,train_labels)
LGBM_TestScore = accuracy_score(y_predtest_lgbm,test_labels)
#Print accuracy
print ("Test Accuracy with LGBM = ", LGBM_TrainScore)
print ("Test Accuracy with LGBM = ", LGBM_TestScore)


# In[ ]:


#Confusion matrix
import seaborn as sns 
from sklearn.metrics import confusion_matrix

cm_lgbm = confusion_matrix(test_labels, y_predtest_lgbm)
sns.heatmap(cm_lgbm, annot=True, fmt='g')


# In[ ]:


print(classification_report(test_labels, y_predtest_lgbm))


# In[ ]:


from catboost import CatBoostClassifier


# In[ ]:


CatBoost = CatBoostClassifier(learning_rate=0.1, depth=10, iterations=300)
CatBoost.fit(train_data, train_labels)


CatBoost_TrainScore = CatBoost.score(train_data, train_labels)


# In[ ]:


CatBoost_TrainScore


# In[ ]:


CatBoost_TestScore = CatBoost.score(test_data, test_labels)
CatBoost_TestScore


# In[ ]:


# Predicting
y_pred_catboost = CatBoost.predict(test_data)


# In[ ]:


cm_catboost = confusion_matrix(test_labels, y_pred_catboost)
sns.heatmap(cm_lgbm, annot=True, fmt='g')


# In[ ]:


print(classification_report(test_labels, y_pred_catboost))


# In[ ]:


# Comparing Boosting Results
results = pd.DataFrame( [["Random Forest", RF_TrainScore, RF_TestScore ],
                       ["XGBoost", XGB_TrainScore ,XGB_TestScore ],
                        ["Light LGBM", LGBM_TrainScore ,LGBM_TestScore ],
                        ["CatBoost", CatBoost_TrainScore ,CatBoost_TestScore ]],
                       columns = ["Model","Training Accuracy %","Test Evaluation %"]).sort_values(by="Test Evaluation %",ascending=False)
results.style.background_gradient(cmap='BuPu')


# In[ ]:





# In[ ]:




