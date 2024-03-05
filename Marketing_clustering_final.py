#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install kmodes


# In[2]:


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from kmodes.kprototypes import KPrototypes


# In[3]:


data_set =pd.read_csv("C:/Users/ADMIN/Desktop/bankmarketing.csv")


# In[4]:


data_set


# In[5]:


data_set.describe()


# In[6]:


data_set.isnull()


# In[7]:


data_set.columns


# In[8]:


data_set=data_set.drop(['campaign','previous','emp.var.rate', 'cons.price.idx',
       'cons.conf.idx', 'euribor3m', 'nr.employed', 'y'],axis=1)


# In[9]:


data_set


# In[10]:


import seaborn as sns
sns.boxplot(x=data_set['age'])
plt.show()


# In[11]:


A_Q1=data_set['age'].quantile(0.25)
A_Q3=data_set['age'].quantile(0.75)
IQR=A_Q3-A_Q1
print(A_Q1)
print(A_Q3)
print(IQR)
Lower_Whisker = A_Q1-1.5*IQR
Upper_Whisker = A_Q3+1.5*IQR
print(Lower_Whisker, Upper_Whisker)


# In[12]:


data_set = data_set[data_set['age']< Upper_Whisker]


# In[13]:


sns.boxplot(x=data_set['duration'])
plt.show()


# In[14]:


data_set.describe()


# In[15]:


B_Q1=data_set['duration'].quantile(0.25)
B_Q3=data_set['duration'].quantile(0.75)
IQR=B_Q3-B_Q1
print(B_Q1)
print(B_Q3)
print(IQR)
DLower_Whisker = B_Q1-1.5*IQR
DUpper_Whisker = B_Q3+1.5*IQR
print(DLower_Whisker, DUpper_Whisker)


# In[16]:


data_set = data_set[data_set['duration']< DUpper_Whisker]


# In[17]:


data_set.describe()


# In[18]:


sns.boxplot(x=data_set['pdays'])
plt.show()


# In[19]:


C_Q1=data_set['pdays'].quantile(0.25)
C_Q3=data_set['pdays'].quantile(0.75)
IQR=C_Q3-C_Q1
print(C_Q1)
print(C_Q3)
print(IQR)
CLower_Whisker = C_Q1-1.5*IQR
CUpper_Whisker = C_Q3+1.5*IQR
print(CLower_Whisker, CUpper_Whisker)


# In[20]:


data_set = data_set[data_set['pdays']< CUpper_Whisker]


# In[21]:


data_set.describe()


# In[22]:


sns.boxplot(x=data_set['age'])
plt.show()


# In[23]:


data_array=data_set.values


# In[24]:


data_array[:,0] =data_array[:,0].astype(float)
data_array[:,10]=data_array[:,10].astype(float)
data_array[:,11]=data_array[:,11].astype(float)


# In[25]:


data_array


# In[26]:


Kproto =KPrototypes(n_clusters=3,verbose=2,max_iter=10)
clusters=Kproto.fit_predict(data_array,categorical = [1,2,3,4,5,6,7,8,9,12])


# In[27]:


print(Kproto.cluster_centroids_)


# In[28]:


cluster_dict=[]
for c in clusters:
    cluster_dict.append(c)
    


# In[29]:


cluster_dict


# In[30]:


data_set['cluster'] =cluster_dict


# In[31]:


data_set


# In[32]:


plt.scatter(x=data_set['age'],y=data_set['job'],c=data_set['cluster'])
plt.show()


# In[33]:


plt.scatter(x=data_set['age'],y=data_set['pdays'],c=data_set['cluster'])
plt.xlabel('age')
plt.ylabel('pdays')
plt.legend()
plt.show()


# In[34]:


plt.scatter(x=data_set['duration'],y=data_set['poutcome'],c=data_set['cluster'])
plt.xlabel('duration')
plt.ylabel('poutcome')


# In[35]:


data_set[data_set['cluster']==0]


# In[36]:


data_set[data_set['cluster']==1]


# In[37]:


data_set[data_set['cluster']==2]


# In[38]:


####Clustering has been doon on the basis of duration.

