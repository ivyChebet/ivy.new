#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np


# In[2]:


S_data=pd.read_csv("C:/Users/ADMIN/Desktop/supermarket_sales.csv")
S_data


# In[3]:


##### most sold item per city


# In[4]:


max_value=S_data['Total'].max()
max_value


# In[5]:


p=S_data["Product line"]
product_list=list(p)
product_list

c=S_data["cogs"]
cogs_list=list(c)
cogs_list


# In[6]:


cd=S_data['City']
city_list=list(cd)
city_list


# In[7]:


cogsmax=max(cogs_list)
cogsmax


# In[50]:


cogsproduct=list(zip(city_list,product_list,cogs_list))
print(cogsproduct)


# In[9]:


for x,y,z in cogsproduct:
    if x=='':
        q=[x,y,z]
print(q)


# In[10]:


J=q[0]
J


# In[11]:


K=q[1]
K


# In[12]:


L=q[2]
L


# In[13]:


import pymysql


# In[79]:


# Connect to MySQL server
connection = pymysql.connect(
    host='localhost',
    user='root',
    password='*****',
    database='********'
)
cursor2=connection.cursor()
cursor2.execute("CREATE DATABASE IF NOT EXISTS supermarketreports0")


# In[80]:


# create a table
create_table_query = """
CREATE TABLE IF NOT EXISTS `reports2` (
    `City` VARCHAR(200),
    `Product` VARCHAR(200),
    `Quantity` VARCHAR(200)
    
)
"""
# Execute the CREATE TABLE query
cursor2.execute(create_table_query)


# In[81]:


import pandas as pd


# In[82]:


####insert into query
insert_table_query="""
INSERT INTO reports2(
   city,Product,Quantity
)VALUES(%s,%s,%s)
"""
   


# In[83]:


#####dump into data


# In[84]:


data_to_insert = (J,K,L)

# Execute the INSERT INTO query with data
cursor2.execute(insert_table_query, data_to_insert)

# Commit the changes
connection.commit()


# In[85]:


######most sold item per city.


# In[86]:


most_sold_item=[]
for x,y,z in cogsproduct:
    if  x== 'Yangon':
        if z== np.max(z):
            z=[x,y,z]
    most_sold_item.append(z)
most_sold_item       


# In[47]:


payment=S_data['Payment']
payment_list=list(payment)
payment_list


# In[90]:


paymentmax=max(payment_list)
paymentmax


# In[ ]:


####the most used payment method generally is Ewallet


# In[91]:


citypayment=S_data[['City','Payment']]
citypayment


# In[92]:


city_payment=list(zip(city_list,payment_list))
city_payment

