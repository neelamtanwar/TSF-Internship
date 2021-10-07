#!/usr/bin/env python
# coding: utf-8

# # TSF Internship Task 1

# 
# AIM : Predict the percentage of a student based on the number of study hours
# LANGUAGE USED : Python 3.7
# IDE : Jupyter Notebook
# AUTHOR : Neelam Tanwar
# 

# In[1]:


#importing the libraries 
import warnings
warnings.filterwarnings("ignore")



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#reading the dataset
url = "https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv"
data = pd.read_csv(url)
print("Displaying Dataset")
data.head()


# In[3]:


data.isna() #finding the null values 


# There is no null value

# In[4]:


print("INFORMATION REGARDING DATA SET:_ ")

data.info()   #getting the information about dataset


# In[5]:


print("Desccribing the data")
data.describe()


# 1. here we could see the important decriptive statistcs regarding our dataset.

# # VISUALISATION

# In[15]:



data.plot(x='Hours', y='Scores',style = 'o',color = 'blue')
ax = plt.axes()
plt.title("Relationship between hours of study and scores",fontsize=30)
plt.xlabel("Hours Of Study", fontsize=20)
plt.ylabel("Scores Obtained",fontsize=20)
ax.set_facecolor("black")
plt.figure(figsize=[40,40])
plt.show()


# graph shows that relationship between hours of study and score obtained is somehow linear , it means with increase of hours of study , score obtained is also increasing.

# # PREPARING THE DATA

# In[19]:


#now we divide the data into attributes and labels
x = data.iloc[:,:-1] #independent data
y = data.iloc[:,1] #dependent data


# # training the data 

# In[27]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[29]:


from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x_train,y_train)
print("Data is Trained")


# In[31]:


print (reg.coef_)
print(reg.intercept_) 


# In[37]:


ax = plt.axes()
line = x*reg.coef_+reg.intercept_
plt.scatter(x,y,color='brown')
plt.plot(x,line,color = 'orange')
ax.set_facecolor("grey")
plt.show()


# # PREDICTING THE VLAUE AND  COMPARING WITH ACTUAL VALUE

# In[42]:


print(x_test)
y_pred = reg.predict(x_test)


# In[48]:


#comparing with actual value
df = pd.DataFrame({'Actual': y_test,'Predicted': y_pred})
df


# In[54]:


df.plot(kind='bar',figsize=[10,10],color = 'bg')


# this shows the difference between predicted value and actual value which is very small

# # testing our own data

# In[64]:


hours = 8.5

arr = np.array([hours])
arr = arr.reshape(-1,1)
own_pred = reg.predict(arr)
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))


# # evaluating model

# In[73]:


from sklearn import metrics
print('Mean Absolute Error=', metrics.mean_absolute_error(y_test,y_pred))
print('Mean Square Error =',metrics.mean_squared_error(y_test,y_pred))
print("R^2  =", metrics.r2_score(y_test,y_pred))


# R^2 is coffecient of determination which shows that precision of prediction is around 94%.

# # THANK YOU

# In[ ]:




