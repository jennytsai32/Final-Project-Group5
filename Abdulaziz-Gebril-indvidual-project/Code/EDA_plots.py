#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 


# In[37]:


us_data= pd.read_csv('final_data.csv')


# In[38]:


us_data


# In[ ]:





# In[39]:


plt.figure(figsize=(9,8))
plt.hist(us_data['Temperature(F)'], color = 'red', edgecolor = 'black',
         bins = int(180/5))
plt.xlabel('Temperature(F)',fontsize=12)
plt.title('Temperature Distribution',fontsize=20)
plt.show()


# In[ ]:





# In[ ]:





# In[29]:


plt.figure(figsize=(9,8))
plt.hist(us_data['Humidity(%)'], color = 'yellow', edgecolor = 'black',
         bins = int(180/5))

plt.xlabel('Humidity(%)',fontsize=12)
plt.title('Humidity Distribution',fontsize=20)
plt.show()


# In[ ]:





# In[28]:


plt.figure(figsize=(9,8))
plt.hist(us_data['Wind_Speed(mph)'], color = 'blue', edgecolor = 'black',
         bins = int(180/5))
plt.xlabel('Wind_Speed(mph)',fontsize=12)
plt.title('Wind Speed Distribution',fontsize=20)
plt.show()


# In[ ]:





# In[26]:


plt.figure(figsize=(9,8))
plt.hist(us_data['Pressure(in)'], color = 'blue', edgecolor = 'black',
         bins = int(180/5))
plt.xlabel('Pressure(in)',fontsize=12)
plt.title('Pressure Distribution',fontsize=20)

plt.show()


# In[40]:


fig, ax=plt.subplots(figsize=(16,7))
us_data['Weather_Condition'].value_counts().sort_values(ascending=False).plot.bar(width=0.5,edgecolor='k',align='center',linewidth=2)
plt.xlabel('Weather_Condition',fontsize=20)
plt.ylabel('Number of Accidents',fontsize=20)
ax.tick_params(labelsize=20)
plt.title('Weather Condition for accidents',fontsize=25)
plt.grid()
plt.ioff()


# In[42]:


f,ax=plt.subplots(1,2,figsize=(18,8))
us_data['Severity'].value_counts().plot.pie(autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Percentage Severity Distribution')
ax[0].set_ylabel('Count')
sns.countplot('Severity',data=us_data,ax=ax[1],order=us_data['Severity'].value_counts().index)
ax[1].set_title('Count of Severity')
plt.show()


# In[36]:





# In[ ]:




