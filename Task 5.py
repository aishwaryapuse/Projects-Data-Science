#!/usr/bin/env python
# coding: utf-8

# ### Global Terrorism Dataset

# In[1]:


# Loading libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings("ignore")


# In[2]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[3]:


DF = pd.read_csv("C:\\Users\\aishw\\Downloads\\Global Terrorism - START data\globalterrorismdb_0718dist.csv",encoding='latin1')


# In[4]:


DF.head()


# In[5]:


DF.columns.values


# In[6]:


DF.rename(columns={'iyear':'Year','imonth':'Month','iday':"day",'gname':'Group','country_txt':'Country','region_txt':'Region','provstate':'State','city':'City','latitude':'latitude',
    'longitude':'longitude','summary':'summary','attacktype1_txt':'Attacktype','targtype1_txt':'Targettype','weaptype1_txt':'Weapon','nkill':'kill',
     'nwound':'Wound'},inplace=True)


# In[7]:


DF = DF[['Year','Month','day','Country','State','Region','City','latitude','longitude',"Attacktype",'kill',
               'Wound','target1','summary','Group','Targettype','Weapon','motive']]


# In[8]:


DF.head()


# In[9]:


DF.shape


# In[10]:


DF.info


# In[11]:


DF.isnull().sum()


# In[12]:


DF.isna().sum()


# In[13]:


DF.describe()


# In[14]:


DF.head


# In[15]:


year=DF['Year'].unique()
years_count = DF['Year'].value_counts(dropna = False).sort_index()
plt.figure(figsize = (28,18))
sns.barplot(x = year,y = years_count,palette = "viridis")
plt.xticks(rotation = 50)
plt.title('Attacks Taking Place In Years',fontsize=40)
plt.xlabel('Attacking Year',fontsize=30)
plt.ylabel('Number of Attacks Per Year',fontsize=30)

plt.show()


# In[16]:


# Top 5 years with most Attack happened from 1971 to 2014 [2014,2015,2016,2013,2017,2012]
DF['Year'].value_counts().plot(kind='bar')


# In[17]:


# Year wise Terrorist Activities
pd.crosstab(DF.Year, DF.Region).plot(kind='area',stacked=True,figsize=(20,10))
plt.ylabel('No:of Attacks',fontsize=25)
plt.xlabel("Years",fontsize=25)
plt.title('Terrorist Activities (Region) In Each Year',fontsize=30)
plt.show()


# In[18]:


attack = DF.Country.value_counts()[:10]
attack


# In[19]:


sns.displot(data = DF, x="Year", kind="hist",multiple="stack")


# In[20]:


DF['Month'].value_counts().plot(kind='bar') 


# In[21]:


DF.Group.value_counts()[1:10]


# In[22]:


df = DF[['Year','kill']].groupby(['Year']).sum()
fig, ax4 = plt.subplots(figsize=(20,10))
df.plot(kind='bar',alpha=0.7,ax=ax4)
plt.xticks(rotation = 50)
plt.title("People who died because of Attack",fontsize=25)
plt.ylabel("No of killed people",fontsize=20)
plt.xlabel('Year',fontsize=20)
top_side = ax4.spines["top"]
top_side.set_visible(False)
right_side = ax4.spines["right"]
right_side.set_visible(False)


# In[23]:


DF['City'].value_counts().to_frame().sort_values('City',axis=0,ascending=False).head(10).plot(kind='bar',figsize=(20,10),color='yellow')
plt.xticks(rotation = 50)
plt.xlabel("City",fontsize=15)
plt.ylabel("No of attack",fontsize=15)
plt.title("Top 10  cities",fontsize=20)
plt.show()


# In[24]:


# Top 5 countries with most attack happened
DF['Country'].value_counts().head().plot(kind='line',color ='black') 


# In[25]:


DF['Country'].value_counts().head()


# In[26]:


# Top 5 regions with most attack happened 
DF['Region'].value_counts().head().plot(kind='bar',color='m') 


# In[27]:


DF['Region'].value_counts().head()


# In[28]:


DF['Month'].value_counts().head()


# In[29]:


DF['Attacktype'].value_counts().plot(kind='bar',figsize=(20,10),color='g')
plt.xticks(rotation = 50)
plt.xlabel("Attacktype",fontsize=15)
plt.ylabel("Number of attack",fontsize=15)
plt.title("Name - attacktype",fontsize=20)
plt.show()


# In[30]:


DF[['Attacktype','kill']].groupby(["Attacktype"],axis=0).sum().plot(kind='bar',figsize=(20,10),color=['darkslateblue'])
plt.xlabel('Attack type',fontsize=15)
plt.title("Number of killed ",fontsize=20)
plt.ylabel('Number of people',fontsize=15)
plt.xticks(rotation=50)
plt.show()


# In[31]:


DF[['Attacktype','Wound']].groupby(["Attacktype"],axis=0).sum().plot(kind='bar',figsize=(20,10),color=['orange'])
plt.xticks(rotation=50)
plt.title("Number of wounded  ",fontsize=20)
plt.ylabel('Number of people',fontsize=15)
plt.xlabel('Attack type',fontsize=15)
plt.show()


# In[32]:


DF['Group'].value_counts().to_frame().drop('Unknown').head(10).plot(kind='bar',color='brown',figsize=(20,10))
plt.title("Top 10 terrorist group attack",fontsize=20)
plt.xlabel("terrorist group name",fontsize=15)
plt.ylabel("Attack number",fontsize=15)
plt.show()


# In[33]:


DF[['Group','kill']].groupby(['Group'],axis=0).sum().drop('Unknown').sort_values('kill',ascending=False).head(10).plot(kind='bar',color='c',figsize=(20,10))
plt.title("Top 10 terrorist group attack",fontsize=20)
plt.xlabel("terrorist group name",fontsize=15)
plt.ylabel("No of killed people",fontsize=15)
plt.show()


# In[34]:


plt.subplots(figsize=(8,6))
sns.barplot(y=DF['Group'].value_counts()[1:12].index,x=DF['Group'].value_counts()[1:12].values,palette='deep')
plt.title('Most Active Terrorist Organizations')
plt.show()


# In[35]:


df=DF[['Group','Country','kill']]
df=df.groupby(['Group','Country'],axis=0).sum().sort_values('kill',ascending=False).drop('Unknown').reset_index().head(10)
df


# In[36]:


typeKill = DF.pivot_table(columns='Attacktype', values='kill', aggfunc='sum')
typeKill


# In[37]:


kill = DF.loc[:,'kill']
print('Number of people killed by terror attack:', int(sum(kill.dropna())))


# In[38]:


countryKill = DF.pivot_table(columns='Country', values='kill', aggfunc='sum')
countryKill


# In[39]:


typeKill = DF.pivot_table(columns='Attacktype', values='kill', aggfunc='sum')
typeKill


# In[40]:


DF.head()


# In[41]:


# Country wise Terrorist attacks
fig,axes = plt.subplots(figsize=(10,8),nrows=1,ncols=2)
sns.barplot(x = DF['Country'].value_counts()[:20].values, y = DF['Country'].value_counts()[:20].index,ax=axes[0],palette = 'coolwarm_r');
axes[0].set_title('Country Wise Terrorist Attacks')
sns.barplot(x=DF['Region'].value_counts().values,y=DF['Region'].value_counts().index,ax=axes[1])
axes[1].set_title('Region Wise Terrorist Attacks')
fig.tight_layout()
plt.show()


# In[42]:


Corr_data = DF[['Year','Month','day','Country','State','Region','City','latitude','longitude','Attacktype','kill','Wound','target1','summary','Group','Targettype','Weapon','motive']]


# In[43]:


plt.figure(figsize=(8,6))
sns.heatmap(np.round(Corr_data.corr(),2),annot=True,cmap='PuRd')


# In[44]:


attacks_by_country_region = DF.groupby(['Country', 'Region']).size().reset_index(name='Total_Attacks_Count')

# Find the row with the maximum total number of attacks
max_attacks_row = attacks_by_country_region.loc[attacks_by_country_region['Total_Attacks_Count'].idxmax()]
min_attacks_row = attacks_by_country_region.loc[attacks_by_country_region['Total_Attacks_Count'].idxmin()]

print("Country with the maximum attacks:", max_attacks_row['Country'])
print("Region with the maximum attacks:", max_attacks_row['Region'])
print("Region with the minimum attacks:", min_attacks_row['Region'])
print("Total number of attacks in that region:", max_attacks_row['Total_Attacks_Count'])


# #CONCLUDING RESULTS FOUND
# 
# Month with most attackes - MAY Most attacking group- TALIBAN Most Attack Types: Bombing/Explosion Country with the most attacks: Iraq Year with the most attacks: 2014
# 
# 
# ### Number of attacks have increased from 1970 to 2017 by 94.0%
# Country Iraq is found to be highly targetted and has the most attacks and hence is considered as the hot zone of terrorism.
# The Middle East and North Africa regions are again the most taregeted ones.
# Bombing/Explosion is the most frequent method of attack by terrorists.
# Taliban and ISIL are ont the top of the list of most active terrorist organisations.

# In[ ]:




