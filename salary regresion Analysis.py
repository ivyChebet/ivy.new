#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np


# In[3]:


data=pd.read_csv("C:/Users/ADMIN/Desktop/Salary_dataset.csv")
data


# In[4]:


from sklearn.linear_model import LinearRegression


# In[5]:


model= LinearRegression()


# In[6]:


model.fit(data[['YearsExperience']],data[['Salary']])


# In[7]:


Years=[16.3,12.5,11.6,13.8]


# In[8]:


for Year in Years:
    Salaries=model.predict([[Year]])
    print(Salaries)


# In[ ]:




