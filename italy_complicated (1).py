#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


italy_data=pd.read_excel("C:/Users/ADMIN/Desktop/italy.xlsx")
italy_data


# In[3]:


italy_data.info()


# In[4]:


italy=pd.DataFrame(italy_data)
italy


# In[5]:


#####Descriptive Statistics:
###Calculate the mean, median, and standard deviation of the number of charging points


# In[6]:


italy.describe()


# In[7]:


###Number of charging points has a mean of 5.215669 and a standrd deviaion of 35.188089


# In[8]:


####Population Analysis.


# In[9]:


#######Population Analysis:
##Explore the relationship between the number of charging points and the population of each city


# In[10]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[11]:


sns.scatterplot(data=italy.iloc[0:50],x="population",y="num_charging_points_reprojected")


# In[12]:


sns.barplot(x="population",y="num_charging_points_reprojected",data = italy.iloc[0:20])
plt.show()


# In[13]:


from matplotlib.pyplot import figure
figure(num=None, figsize=(50,100), dpi=256, facecolor='w', edgecolor='r')
plt.title("relationship between population and number of charging points")
sns.barplot(x="population",y="num_charging_points_reprojected",data = italy[0:50])
plt.show()


# In[14]:


italy[['population','num_charging_points_reprojected']].corr()


# In[15]:


####From the corr() function one can tell that there is a strong positive correlation between the variable population
####and number of charging points.


# In[16]:


######from the boxplot one can tell that there is a positive linear relationship between the number of charging oints 


# In[17]:


sns.barplot( x="city",y="num_charging_points_reprojected",data=italy.iloc[0:10])
plt.show()


# In[18]:


italy.corr()


# In[19]:


import plotly.express as px
fig = px.bar(italy[0:10], x='num_charging_points_reprojected', y='city')
fig.show()


# In[20]:


#######Regional Comparisons:
####3Compare the number of charging points in northern, central, and southern regions of Italy


# In[ ]:





# In[21]:


######Outliers Detection:
###Identify any cities with an unusually high or low number of charging points and investigate the reasons


# In[22]:


plt.boxplot(data=italy,x=italy['num_charging_points_reprojected'])
plt.show()


# In[23]:


db1=italy[['city','num_charging_points_reprojected','population']]
db1


# In[24]:


db2=db1.sort_values(["num_charging_points_reprojected","city"])
print(db2)


# In[25]:


AB=db2.head(20)
pd.DataFrame(AB)
print(AB)


# In[26]:


CD=db2.tail(10)
pd.DataFrame(CD)
print(CD)


# In[27]:


city1=db1['city']
citylist=list(city1)
citylist


# In[28]:


points=db1['num_charging_points_reprojected']
pointslist=list(points)
pointslist


# In[29]:


pointscity=list(zip(citylist,pointslist))
pointscity


# In[30]:


#min1=[]
for x,y in pointscity:
    if  y <1:
        data3=[x,y]
        print(data3)
        
        

   


# In[31]:


db2


# In[32]:


db2.rename({'num_charging_points_reprojected':'chargingpoints','population':'pop'},axis='columns')


# In[33]:


sns.barplot(data=CD,x="population",y="num_charging_points_reprojected")
plt.show()


# In[34]:


########the number of charging points in everycity depends on the popultion.The higher the population the higher the number of charging points.


# In[35]:


######7.Population Density Impact:
#####Analyze how population density (pop_ISTAT/M2) relates to the number of charging points.


# In[36]:


sns.scatterplot(data=italy,x=italy["population"],y=italy["M2"])
plt.show()


# In[37]:


x_values=italy["population"]
x_values


# In[38]:


y_values=italy["M2"]
y_values


# In[39]:


import random


# In[40]:





plt.figure(figsize=(6, 5))
ax = plt.axes()
ax.scatter(x=x_values,y=y_values)

ax.set_xlabel('population')
ax.set_ylabel('M2')

plt.show()


# In[41]:


######Plotting the best line of fit.


# In[42]:


plt.scatter(y_values, x_values)
plt.xlabel('M2')
plt.ylabel('population')



z = np.polyfit(y_values, x_values, 1)
p = np.poly1d(z)
plt.plot(y_values,p(y_values),"r--")

# Show the plot
plt.show()


# In[43]:


####From the best line of fit one can see that we have a  negative linear regression


# In[44]:


####3From the corr() function M2 and Poulation have a negative correlation of -0.011323
####This shows that as one variable increases the other one decreases.


# In[45]:


#####Score vs. Charging Points:
#####Investigate the relationship between the M5 score and the number of charging points
#####Where M5 is the percentage EV cars.


# In[ ]:





# In[46]:


plt.scatter(CD['num_charging_points_reprojected'], CD['population'])
plt.xlabel('num_charging_points_reprojected')
plt.ylabel('population')



z = np.polyfit(CD['num_charging_points_reprojected'],italy['population'] ,1)
p = np.poly1d(z)
plt.plot(CD['num_charging_points_reprojected'],p(CD['num_charging_points_reprojected']) ,"r--")

# Show the plot
plt.show()


# In[47]:


######there is a negative linear regression between number of charging points and M5(percentage of MV cars)


# In[ ]:


italy[]


# In[48]:


######Score Distribution:
#####Visualize the distribution of M5 scores across different cities.


# In[49]:


import plotly.express as px
fig = px.bar(italy[0:500],x='city',y='M5')
fig.show()


# In[50]:


import plotly.express as px
fig = px.bar(italy[499:1001],x='city',y='M5')
fig.show()


# In[51]:


#######M2 Score Comparison:
#######Compare the M2 score of each city and identify any patterns or trends
#######


# In[52]:


#Population vs. M5 Score:
#Analyze if there's any correlation between population size and the M5 score


# In[53]:


italy.corr()


# In[54]:


####there is a weak postive correlation between
####population and percentage of MV cars
####as one variable increases the other also increases although there is no strong relationship between the two.


# In[55]:


k_values=italy['M5']
k_values


# In[56]:


plt.scatter(x_values,k_values )
plt.xlabel('population')
plt.ylabel('M5')



z = np.polyfit(x_values,k_values ,1)
p = np.poly1d(z)
plt.plot(x_values,p(x_values),"r--")

# Show the plot
plt.show()


# In[57]:


####the scatter plot shows that there is a weak positive linear regression between 
#####population and the percentage of MV cars.


# In[58]:


#####Top Charging Cities:
###Identify the top 5 cities with the highest number of charging points.


# In[59]:


city_point4=italy[["city","num_charging_points_reprojected"]]
city_point4


# In[60]:


sorted_df = city_point4.sort_values(by='num_charging_points_reprojected',ascending=False)
sorted_df


# In[61]:


sorted_df.head()


# ### Predictive Modeling:
# ####Build a predictive model to estimate the number of charging points based on ####available features.

# In[62]:


###for this linear regression would be effective.
####The first step would be to identify the dependent and independent variables.


# In[63]:


print(italy.columns)


# In[64]:


italyfinal=italy.drop(['PRO_COM_T','M2','M2_score'],axis=1)
italyfinal


# In[65]:


#####the dependent variable is number of charging points while the independent variables are 
#####population and prcentage of V cars


# In[66]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score 


# In[67]:


y=italyfinal['num_charging_points_reprojected'].values.reshape(-1,1)
y


# In[68]:


Variables=italyfinal[['population','M5']]
Variables
x=Variables


# In[69]:


x


# In[70]:


X_train, X_test, y_train, y_test = train_test_split(x, y,test_size=0.2,random_state=0)


# In[71]:


print(X_train)


# In[72]:


print(X_test) 


# In[73]:


print(y_train)


# In[74]:


print( y_test)


# In[75]:


regressor = LinearRegression()
regressor.fit(X_train, y_train)


# In[76]:


######Intercept


# In[77]:


print(regressor.intercept_)


# In[78]:


#####retrieving the slope (which is also the coefficient of x)


# In[79]:


print(regressor.coef_)


# In[80]:


#####Make Predictions


# In[81]:


y_pred = regressor.predict(X_test)


# In[82]:


y_pred


# In[83]:


Predicted=pd.DataFrame(y_pred)
Predicted


# In[84]:


italy_predicted=pd.concat([italyfinal,Predicted])
italy_predicted


# In[85]:


italy_predicted.iloc[7902:9482]


# In[86]:


########cheking if our model is okay,we use the mean square error 


# In[87]:


mae=mean_absolute_error(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
rmse=np.sqrt(mse)


# In[88]:


print(mae)


# In[89]:


print(mse)


# In[90]:


print(rmse)


# In[92]:


#our mae is close to 0 hence effective.


# In[ ]:




