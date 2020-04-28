#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.utils import resample


# In[2]:


# import dataset
data = pd.read_csv('us_data_criteria_1.csv')


# In[3]:


# Subsetting
data = data[(data.Year==2019)]
data.groupby(['Severity', 'Year'])['Severity'].count()


# In[4]:


# Missing Values
print("Sum of NULL values in each column. ")
print(data.isnull().sum())


# In[5]:


# Separate majority and minority classes
majority = data[data.Severity=='Low']
minority = data[data.Severity=='High']

# Downsample
majority_down = resample(majority, replace=True, n_samples=1000, random_state=100)
minority_down = resample(minority, replace=True, n_samples=1000, random_state=100)
data = pd.concat([majority_down, minority_down])
 
# Display new class counts
data.Severity.value_counts()


# In[6]:


data.to_csv('us_data_sample_for_gui.csv')


# In[ ]:




