#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 


# In[2]:


#data_1= pd.read_csv("us_data_2019_Criteria_1.csv")
#data_2=pd.read_csv("us_data_2018_Criteria_1.csv")


# In[3]:


#columns= ["ID","Source","TMC","Severity","Start_Time","End_Time","Start_Lat","Start_Lng","End_Lat","End_Lng",
 #         "Distance(mi)","Description","Number","Street","Side","City","County","State",
  #        "Zipcode","Country","Timezone","Airport_Code","Weather_Timestamp",
   #       "Temperature(F)","Wind_Chill(F)","Humidity(%)","Pressure(in)","Visibility(mi)",
    #      "Wind_Direction","Wind_Speed(mph)","Precipitation(in)","Weather_Condition","Amenity","Bump",
     #     "Crossing","Give_Way","Junction","No_Exit","Railway","Roundabout","Station","Stop",
      #    "Traffic_Calming","Traffic_Signal","Turning_Loop","Sunrise_Sunset","Civil_Twilight","Nautical_Twilight",
       #   "Astronomical_Twilight","Year","Date","Month","Day","Hour","Weekday","Time_Duration(min)"]

#data_1=data_1[columns]
#data_2=data_2[columns]


# In[4]:


#data=pd.concat([data_1,data_2], axis=0)
#data.to_csv("final_data.csv", encoding='utf-8', index=False)


# In[66]:


data=pd.read_csv("final_data.csv")
data


# In[42]:


missing_df = data.isnull().sum(axis=0).reset_index()
missing_df.columns = ['column_name','missing_count']
missing_df = missing_df.ix[missing_df['missing_count']>0]

missing_df = missing_df.sort_values(by='missing_count')
print(missing_df)
ind = np.arange(missing_df.shape[0])
width = 0.5
fig,ax = plt.subplots(figsize=(12,18))
rects = ax.barh(ind,missing_df.missing_count.values,color='blue')
ax.set_yticks(ind)
ax.set_yticklabels(missing_df.column_name.values, rotation='horizontal')
ax.set_xlabel("Count of missing values")
ax.set_title("Number of missing values in each column")
plt.show()


# In[59]:


cols=["Wind_Speed(mph)","Visibility(mi)","Temperature(F)","Pressure(in)","Humidity(%)","Wind_Chill(F)"]
data_2=data[cols]
data_2=data_2.dropna()
features=["Wind_Speed(mph)","Visibility(mi)","Temperature(F)","Pressure(in)","Humidity(%)"]
target=["Wind_Chill(F)"]
x=data_2[features]
y=data_2[target]


# In[ ]:





# In[54]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=10) 


# In[30]:


# training model and checking for R2

from sklearn.metrics import mean_squared_error, r2_score


reg = LinearRegression().fit(X_train, y_train)
y_pred = reg.predict(X_test)
print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))


# In[31]:


adj_r2 = 1-(1-0.98)*(len(y_test)-1)/(len(y_test)-len(features)-1)
adj_r2


# In[60]:


data= data.dropna(subset=["Wind_Speed(mph)"])
data= data.dropna(subset=["Visibility(mi)"])
data= data.dropna(subset=["Temperature(F)"])
data= data.dropna(subset=["Pressure(in)"])
data= data.dropna(subset=["Humidity(%)"])


# In[33]:


new_wind_chill=reg.predict(data[features])[data["Wind_Chill(F)"].isnull()]


# In[34]:


chill= new_wind_chill.ravel()


# In[35]:


chill= new_wind_chill.tolist()


# In[36]:


from itertools import chain 
chill_list = [j for sub in chill for j in sub] 
print(chill_list)


# In[37]:





# In[67]:


data= data.dropna(subset=["Wind_Speed(mph)"])
data= data.dropna(subset=["Visibility(mi)"])
data= data.dropna(subset=["Temperature(F)"])
data= data.dropna(subset=["Pressure(in)"])
data= data.dropna(subset=["Humidity(%)"])


# In[68]:


missing_columns=["Wind_Chill(F)"]

# filling missing values with random number of the observed values as a starting point

def random_imputation(df, feature):

    number_missing = df[feature].isnull().sum()
    observed_values = df.loc[df[feature].notnull(), feature]
    df.loc[df[feature].isnull(), feature + '_imp'] = np.random.choice(observed_values, number_missing, replace = True)
    
    return df


# In[69]:


for feature in missing_columns:
    data[feature + '_imp'] = data[feature]
    data = random_imputation(data, feature)


# In[70]:



# using regression to impute missing values 
deter_data = pd.DataFrame(columns = ["Det" + name for name in missing_columns])

for feature in missing_columns:
        
    deter_data["Det" + feature] = data[feature+ '_imp']
    parameters = ["Wind_Speed(mph)","Visibility(mi)","Temperature(F)","Pressure(in)","Humidity(%)"]
    
    #Create a Linear Regression model to estimate the missing data
    model = LinearRegression().fit(X = data[parameters], y = data[feature + '_imp'])
    

    #keep the index of the missing data from the original dataframe when creating the new column
    deter_data.loc[data[feature].isnull(), "Det" + feature] = model.predict(data[parameters])[data[feature].isnull()]


# In[72]:


deter_data


# In[85]:


new_data=pd.merge(data, deter_data, left_index=True, right_index=True)


# In[80]:


new_data["Wind_Chill(F)"]=new_data["Wind_Chill(F)"].fillna(new_data["DetWind_Chill(F)"], inplace=True)


# In[ ]:





# In[91]:


new_data.to_csv("wind_chill_regression_2019&2018_data.csv", encoding='utf-8', index=False)

