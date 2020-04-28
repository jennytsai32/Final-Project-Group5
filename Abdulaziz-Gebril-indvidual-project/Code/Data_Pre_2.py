#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
#pd.options.mode.chained_assignment = None
#pd.options.display.max_columns = 999
import missingno as msno 


# In[2]:


# reading U.S accidents csv file
us_data= pd.read_csv('US_Accidents_Dec19.csv')


# In[3]:


us_data


# In[ ]:





# In[4]:


# Severity Classification
# 1,2 = Low
#3,4 =High
us_data["Severity"]=us_data["Severity"].replace({1: "Low", 2: "Low", 3: "High", 4:"High"})


# In[5]:




us_data['Start_Time'] = pd.to_datetime(us_data['Start_Time'], errors='coerce')
us_data['End_Time'] = pd.to_datetime(us_data['End_Time'], errors='coerce')

# Extract year, month, day, hour and weekday and create new columns 
us_data["Date"]=us_data["Start_Time"].dt.date
us_data['Year']=us_data['Start_Time'].dt.year
us_data['Month']=us_data['Start_Time'].dt.strftime('%b')
us_data['Day']=us_data['Start_Time'].dt.day
us_data['Hour']=us_data['Start_Time'].dt.hour
us_data['Weekday']=us_data['Start_Time'].dt.strftime('%a')

# Extract the amount of time in the unit of minutes for each accident, round to the nearest integer
td='Time_Duration(min)'
us_data[td]=round((us_data['End_Time']-us_data['Start_Time'])/np.timedelta64(1,'m'))


# In[15]:


# Subsetting 2019 data.

us_data_2017=us_data.loc[us_data['Year'] == 2016]
#us_data_2017=us_data_2017.loc[us_data_2017['Severity'].isin([3,4])]

us_data_2017


# In[9]:



# Severity Distribution (%)

f,ax=plt.subplots(1,2,figsize=(18,8))
us_data_2017['Severity'].value_counts().plot.pie(autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Percentage Severity Distribution')
ax[0].set_ylabel('Count')
sns.countplot('Severity',data=us_data_2017,ax=ax[1],order=us_data_2017['Severity'].value_counts().index)
ax[1].set_title('Count of Severity')
plt.show()


# In[14]:


# Missing data Pattern
msno.matrix(us_data_2017) 


# In[ ]:





# In[50]:


# Missing data dataframe

missing_df = us_data_2017.isnull().sum(axis=0).reset_index()
missing_df.columns = ['columns_name','missing_count']
missing_df['missing_ratio'] = missing_df['missing_count'] /us_data_2017.shape[0]
#missing_df.loc[missing_df['missing_ratio']>0.1]
missing_df


# In[28]:


# Missing data visual


missing_df = us_data_2017.isnull().sum(axis=0).reset_index()
missing_df.columns = ['column_name','missing_count']
missing_df = missing_df.ix[missing_df['missing_count']>0]

missing_df = missing_df.sort_values(by='missing_count')
print(missing_df)
ind = np.arange(missing_df.shape[0])
width = 0.5
fig,ax = plt.subplots(figsize=(6,10))
rects = ax.barh(ind,missing_df.missing_count.values,color='blue')
ax.set_yticks(ind)
ax.set_yticklabels(missing_df.column_name.values, rotation='horizontal')
ax.set_xlabel("Count of missing values")
ax.set_title("Number of missing values in each column")
plt.show()


# In[9]:


# Function that calculates distance betwwen two points using Lat and Lng.



def haversine(lat1, lon1, lat2, lon2, to_radians=True, earth_radius=6371):
   
    if to_radians:
        lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])

    a = np.sin((lat2-lat1)/2.0)**2 +         np.cos(lat1) * np.cos(lat2) * np.sin((lon2-lon1)/2.0)**2

    return earth_radius * 2 * np.arcsin(np.sqrt(a))


#df['dist'] = \
    #haversine(df.LAT.shift(), df.LONG.shift(),
                 #df.loc[1:, 'LAT'], df.loc[1:, 'LONG'])


# In[13]:


col=["Date","Month","Start_Lat","Start_Lng", "City", "State","Temperature(F)","Wind_Chill(F)","Humidity(%)","Pressure(in)","Visibility(mi)","Wind_Direction","Wind_Speed(mph)","Precipitation(in)","Weather_Condition"]
us_data_201=us_data_2017[col]


# In[14]:


# Null data dataframe

null_data = us_data_201[us_data_201.isnull().any(axis=1)]
null_data


# In[11]:


# State counts in null_data
null_data["State"].value_counts()


# In[15]:


null_data["Month"].value_counts()


# In[16]:


# Missing value in Percipitation colummn
col_2=["Month","Date","State","Precipitation(in)"]
perc=us_data_2017[col_2]
perc["Month"].value_counts()


# In[25]:


# Using California to check distances between data points 

distance_CA=us_data_2017.loc[us_data_2017['State'] == "CA"]

#distance_CA=distance_CA.sort_values("Date")

distance_CA['dist'] =     haversine(distance_CA.Start_Lat.shift(), distance_CA.Start_Lng.shift(),
                 distance_CA.loc[1:, 'Start_Lat'], distance_CA.loc[1:, 'Start_Lng'])

col_3=["City","dist","Date","Precipitation(in)"]

print(distance_CA[col_3])


# In[20]:


distance_CA["dist"].max()


# In[7]:


# First Imputation

us_data_2017['Wind_Chill(F)'] = us_data_2017['Wind_Chill(F)'].fillna(us_data_2017.groupby(["Date","City","State"])['Wind_Chill(F)'].transform('mean'))
us_data_2017['Precipitation(in)'] = us_data_2017['Precipitation(in)'].fillna(us_data_2017.groupby(["Date","City","State"])['Precipitation(in)'].transform('mean'))
us_data_2017['Pressure(in)'] = us_data_2017['Pressure(in)'].fillna(us_data_2017.groupby(["Date","City","State"])['Pressure(in)'].transform('mean'))
us_data_2017['Temperature(F)'] = us_data_2017['Temperature(F)'].fillna(us_data_2017.groupby(["Date","City","State"])['Temperature(F)'].transform('mean'))
us_data_2017['Humidity(%)'] = us_data_2017['Humidity(%)'].fillna(us_data_2017.groupby(["Date","City","State"])['Humidity(%)'].transform('mean'))
us_data_2017['Visibility(mi)'] = us_data_2017['Visibility(mi)'].fillna(us_data_2017.groupby(["Date","City","State"])['Visibility(mi)'].transform('mean'))
us_data_2017['Wind_Speed(mph)'] = us_data_2017['Wind_Speed(mph)'].fillna(us_data_2017.groupby(["Date","City","State"])['Wind_Speed(mph)'].transform('mean'))


# In[112]:


# Second Imputation

us_data_2017['Wind_Chill(F)'] = us_data_2017['Wind_Chill(F)'].fillna(us_data_2017.groupby(["Date","State"])['Wind_Chill(F)'].transform('mean'))
us_data_2017['Precipitation(in)'] = us_data_2017['Precipitation(in)'].fillna(us_data_2017.groupby(["Date","State"])['Precipitation(in)'].transform('mean'))
us_data_2017['Pressure(in)'] = us_data_2017['Pressure(in)'].fillna(us_data_2017.groupby(["Date","State"])['Pressure(in)'].transform('mean'))
us_data_2017['Temperature(F)'] = us_data_2017['Temperature(F)'].fillna(us_data_2017.groupby(["Date","State"])['Temperature(F)'].transform('mean'))
us_data_2017['Humidity(%)'] = us_data_2017['Humidity(%)'].fillna(us_data_2017.groupby(["Date","State"])['Humidity(%)'].transform('mean'))
us_data_2017['Visibility(mi)'] = us_data_2017['Visibility(mi)'].fillna(us_data_2017.groupby(["Date","State"])['Visibility(mi)'].transform('mean'))
us_data_2017['Wind_Speed(mph)'] = us_data_2017['Wind_Speed(mph)'].fillna(us_data_2017.groupby(["Date","State"])['Wind_Speed(mph)'].transform('mean'))


# In[13]:


us_data_2017["Month"].value_counts()


# In[14]:


# Third Imputation

us_data_2017['Wind_Chill(F)'] = us_data_2017['Wind_Chill(F)'].fillna(us_data_2017.groupby(["Month","State"])['Wind_Chill(F)'].transform('mean'))
us_data_2017['Precipitation(in)'] = us_data_2017['Precipitation(in)'].fillna(us_data_2017.groupby(["Month","State"])['Precipitation(in)'].transform('mean'))
us_data_2017['Pressure(in)'] = us_data_2017['Pressure(in)'].fillna(us_data_2017.groupby(["Month","State"])['Pressure(in)'].transform('mean'))
us_data_2017['Temperature(F)'] = us_data_2017['Temperature(F)'].fillna(us_data_2017.groupby(["Month","State"])['Temperature(F)'].transform('mean'))
us_data_2017['Humidity(%)'] = us_data_2017['Humidity(%)'].fillna(us_data_2017.groupby(["Month","State"])['Humidity(%)'].transform('mean'))
us_data_2017['Visibility(mi)'] = us_data_2017['Visibility(mi)'].fillna(us_data_2017.groupby(["Month","State"])['Visibility(mi)'].transform('mean'))
us_data_2017['Wind_Speed(mph)'] = us_data_2017['Wind_Speed(mph)'].fillna(us_data_2017.groupby(["Month","State"])['Wind_Speed(mph)'].transform('mean'))


# In[96]:





# In[15]:


# Checking for Missing values Pattern
msno.matrix(us_data_2017) 


# In[8]:


missing_df = us_data_2017.isnull().sum(axis=0).reset_index()
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


# In[13]:


#cols_4=["Wind_Speed(mph)","Visibility(mi)","Temperature(F)","Pressure(in)","Humidity(%)"."Wind_Chill(F)"]
#linear_data=us_data_2017[cols_4]


#from sklearn import linear_model

#missing_columns = ["Wind_Chill(F)"]
#deter_data = pd.DataFrame(columns = ["Det" + name for name in missing_columns])

#for feature in missing_columns:
        
    #deter_data["Det" + feature] = us_data_2017[feature ]
    #parameters = ["Wind_Speed(mph)","Visibility(mi)","Temperature(F)","Pressure(in)","Humidity(%)"]
    
    #Create a Linear Regression model to estimate the missing data
    #model = linear_model.LinearRegression()
    #model.fit(X = linear_data[parameters], y = linear_data[feature])
    
    #observe that I preserve the index of the missing data from the original dataframe
    #deter_data.loc[us_data_2017[feature].isnull(), "Det" + feature] = model.predict(linear_data[parameters])[linear_data[feature].isnull()]


# In[92]:


# Data points per state count


states = us_data_2017.State.unique()
count_by_state=[]
for i in us_data_2017.State.unique():
    count_by_state.append(us_data_2017[us_data_2017['State']==i].count()['ID'])

fig,ax = plt.subplots(figsize=(16,10))
sns.barplot(states,count_by_state)


# In[116]:


# Minmizing Weather_Condition Categories 



us_data_2017["Weather_Condition"]=us_data_2017["Weather_Condition"].replace(
    dict.fromkeys(['Mostly Cloudy','Partly Cloudy','Funnel Cloud', 'Scattered Clouds', 'Cloudy / Windy'], 'Cloudy'), 
    regex=True
)

us_data_2017["Weather_Condition"]= us_data_2017["Weather_Condition"].replace(
    dict.fromkeys(['Patches of Fog','Shallow Fog','Drizzle and Fog', "Partial Fog", "Light Freezing Fog", 'Fog / Windy','Light Fog','Mist'], 'Fog'), 
    regex=True
)

us_data_2017["Weather_Condition"]= us_data_2017["Weather_Condition"].replace(
    dict.fromkeys(['Light Rain','Heavy Rain','Heavy Rain / Windy', 'Light Freezing Rain', 'Showers in the Vicinity',
                   'Light Rain Shower','Freezing Rain', 'Rain / Windy', 
                   'Light Freezing Rain / Windy', 
                   'Light Rain Shower / Windy', 'Rain Shower', 'Light Rain Showers','Rains','Rain / Windy', 'N/A Precipitation', 'Rain / Windy', 'Rain / Windy' ,
                   'Rains','Heavy Rain','Light T-Stormstorms and Rain','Rains','Heavy Rain','Rains','Heavy Rain' ], 'Rain'), 
    regex=True
)

us_data_2017["Weather_Condition"]= us_data_2017["Weather_Condition"].replace(
    dict.fromkeys(['Light Snow','Heavy Snow','Snow / Windy', 'Light Snow / Windy', 'Snow and Sleet',
                   'Light Snow Shower','Light Snow and Sleet', 'Light Snow Grains',
                 'Light Snow and Sleet / Windy', 'Heavy Snow with Thunder',
                   'Snow and Thunder', 'Snow and Sleet / Windy', 'Light Snow Showers',
                   'Heavy Blowing Snow','Ice Pellets', 'Low Drifting Snow', 'Snow / Windy',
                   'Blowing Snow','Light Snow', 'Light Snow', "Hail", "Small Hail", 'Snow Shower',
                   'Snow Grains', 'Small Snow', 'Snow with T-Storm', 'Snows', 'Small Snow' , 'Snow with T-Storm', 'Snow / Windy', 'Light Snow', 'Light T-Storms and Snow',
                   'Small Snow','Snow with T-Storm','Light Snow','Snows','T-Storm and Snow / Windy','Light T-Storms and Snow',
                   'Light T-Stormstorms and Snow', 'Light Snow','Snows','Heavy Snow','Small Snow','Snow / Windy', 'Snow with T-Storm',
                   'T-Storm and Snow / Windy','Light Snow','T-Storm and Snow / Windy','Heavy Snow',
                   'Snows','Small Snow','Snow / Windy','Snow with T-Storm','Light T-Stormstorms and Snow','Light Snow','Snows','Heavy Snow',
                   'Small Snow','Snow / Windy','Snow with T-Storm', 'Small Snow','Light Snow','Light T-Stormstorms and Snow','Snows','T-Storms with Snow' ], 'Snow'), 
    regex=True
)


us_data_2017["Weather_Condition"]= us_data_2017["Weather_Condition"].replace(
    dict.fromkeys(['Fair / Windy','Smoke / Windy', 'Wintry Mix / Windy',
                'Light Drizzle / Windy', 'Squalls / Windy', "Squalls" ], 'Windy'), 
    regex=True
)


us_data_2017["Weather_Condition"]= us_data_2017["Weather_Condition"].replace(
    dict.fromkeys(['Drizzle','Light Drizzle', 'Heavy Drizzle',
                'Drizzle / Windy', 'Light Freezing Drizzle', 'Heavy Freezing Drizzle'], 'Drizzle'), 
    regex=True
)


us_data_2017["Weather_Condition"]= us_data_2017["Weather_Condition"].replace(
    dict.fromkeys(['Heavy T-Storm','Rain with Thunder', 'Thunder',
                'Thunder in the Vicinity', 'Thunder / Windy', 'Heavy T-Storm / Windy',
                   'Light Thunderstorms and Rain', 'Heavy Thunderstorms and Rain',
                   'Thunder and Hail / Windy', 'Light Thunderstorms and Snow' , 
                  'Light T-Stormstorms and Rain', 'T-Storm in the Vicinity',
       'Heavy T-Stormstorms and Rain', 'T-Stormstorms and Rain', 'T-Storm / Windy', 'T-Storm in the Vicinity', 'Snow with T-Storm'
                  'T-Storm and Snow / Windy', 'T-Stormstorm', 'T-Storm and Snow', 'T-Storm in the Vicinity', 'T-Storm / Windy','T-Stormstorm', 'Light T-Stormstorms and Rain', 'Heavy T-Stormstorms and Rain','Light T-Stormstorms and Rain','T-Stormstorms and Rain','T-Stormstorm',
                   'T-Storm in the Vicinity','Heavy T-Stormstorms and Rain','Heavy T-Stormstorms and Snow','T-Storm / Windy',
                   'Heavy T-Stormstorms with Small Snow', 'Light T-Stormstorm','T-Stormstorms and Snow','T-Storms with Snow',
                   'Light T-Stormstorms and Rain','T-Stormstorm','T-Storm in the Vicinity',
                   'Heavy T-Stormstorms and Rain','Light T-Stormstorms and Snow',
                   'Heavy T-Stormstorms and Snow','T-Storm / Windy','Heavy T-Stormstorms with Small Snow','Light T-Stormstorm',
                   'T-Stormstorms and Snow','T-Stormstorms and Rain','T-Stormstorm',
                   'T-Storm in the Vicinity','Heavy T-Stormstorms and Rain','T-Storm','T-Storm / Windy',
                   'Heavy T-Stormstorms with Small Snow','T-Stormstorms and Snow','T-Storm and Snow / Windy',
                   'T-Storms and Snow','T-Storms with Snow',
                   'Light T-Storm', 'T-Stormstorm','T-Stormstorms and Rain', 'Light T-Stormstorms and Rain', 'Heavy T-Stormstorms and Rain','T-Storm','Heavy T-Stormstorms with Small Snow',
                   'Light T-Stormstorm' ], 'T-Storm'), 
    regex=True
)



us_data_2017["Weather_Condition"]= us_data_2017["Weather_Condition"].replace(
    dict.fromkeys(['Sleet', 'Light Sleet', 'Heavy Sleet'], 'Sleet'), 
    regex=True
)


us_data_2017["Weather_Condition"]= us_data_2017["Weather_Condition"].replace(
    dict.fromkeys(['Sand / Dust Whirlwinds', 'Blowing Dust', 'Widespread Dust / Windy', 
                   'Sand / Dust Whirlwinds / Windy', 'Widespread Dust', 'Blowing Dust / Windy', 'Dust / Windy','Dust / Windy','Blowing Sand'], 'Dust'), 
    regex=True
)


us_data_2017["Weather_Condition"]= us_data_2017["Weather_Condition"].replace(
    dict.fromkeys(['Haze / Windy','Light Haze'], 'Haze'), 
    regex=True
)


# In[117]:


us_data_2017.Weather_Condition.unique()


# In[98]:


# Accident sper weather condition category

fig, ax=plt.subplots(figsize=(16,7))
us_data_2017['Weather_Condition'].value_counts().sort_values(ascending=False).plot.bar(width=0.5,edgecolor='k',align='center',linewidth=2)
plt.xlabel('Weather_Condition',fontsize=20)
plt.ylabel('Number of Accidents',fontsize=20)
ax.tick_params(labelsize=20)
plt.title('Weather Condition for accidents',fontsize=25)
plt.grid()
plt.ioff()


# In[118]:


# Wind_Direction Category Minimizing 
us_data_2017["Wind_Direction"]=us_data_2017["Wind_Direction"].replace({"E": "East", "W": "West", "N": "North", "S":"South","CALM":"Calm","VAR":"Variable"})


# In[119]:


us_data_2017.Wind_Direction.unique()


# In[120]:


# Imputing missing values for Weather_Condition and Wind_Direction
us_data_2017['Weather_Condition'] = us_data_2017['Weather_Condition'].fillna(us_data_2017.groupby(["Date","City"])['Weather_Condition'].ffill())
us_data_2017['Wind_Direction'] = us_data_2017['Wind_Direction'].fillna(us_data_2017.groupby(["Date","City"])['Wind_Direction'].ffill())


# In[121]:


us_data_2017['Weather_Condition'] = us_data_2017['Weather_Condition'].fillna(us_data_2017.groupby(["Date","State"])['Weather_Condition'].ffill())
us_data_2017['Wind_Direction'] = us_data_2017['Wind_Direction'].fillna(us_data_2017.groupby(["Date","State"])['Wind_Direction'].ffill())


# In[108]:


missing_df = us_data_2017.isnull().sum(axis=0).reset_index()
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


# In[ ]:





# In[30]:


# Wind direction counts 

f, ax = plt.subplots(figsize=(18, 8))
sns.countplot(y="Wind_Direction", data=us_data_2017, color="c");


# In[115]:


#us_data_2018.to_csv("us_data_2018_draft_2.csv", encoding='utf-8', index=False)
#us_data_high=us_data_2018.loc[us_data['Severity'] == 4]
#us_data_high


# In[122]:


#us_data_2017["Severity"]=us_data_2017["Severity"].replace({4:"High", 3:"High"})
#us_data_2017


# In[34]:


#us_data_2017["Severity"]=us_data_2017["Severity"].replace({1: "Low", 2: "Low", 3: "High", 4:"High"})


# In[123]:


#us_data_2017.to_csv("us_data_2017_Criteria_2.csv", encoding='utf-8', index=False)


# In[130]:


#us_data_2019= pd.read_csv('us_data_2019_Criteria_2.csv')
#us_data_2018=pd.read_csv('us_data_2018_Criteria_2.csv')
#us_data_2017=pd.read_csv('us_data_2017_Criteria_2.csv')
#us_data_2016=pd.read_csv('us_data_2016_Criteria_2.csv')


# In[131]:


columns= ["ID","Source","TMC","Severity","Start_Time","End_Time","Start_Lat","Start_Lng","End_Lat","End_Lng",
          "Distance(mi)","Description","Number","Street","Side","City","County","State",
          "Zipcode","Country","Timezone","Airport_Code","Weather_Timestamp",
          "Temperature(F)","Wind_Chill(F)","Humidity(%)","Pressure(in)","Visibility(mi)",
          "Wind_Direction","Wind_Speed(mph)","Precipitation(in)","Weather_Condition","Amenity","Bump",
          "Crossing","Give_Way","Junction","No_Exit","Railway","Roundabout","Station","Stop",
          "Traffic_Calming","Traffic_Signal","Turning_Loop","Sunrise_Sunset","Civil_Twilight","Nautical_Twilight",
          "Astronomical_Twilight","Year","Date","Month","Day","Hour","Weekday","Time_Duration(min)"]

us_data_2019=us_data_2019[columns]
us_data_2018=us_data_2018[columns]
us_data_2017=us_data_2017[columns]
us_data_2016=us_data_2016[columns]


# In[132]:


us_data_final=pd.concat([us_data_2019,us_data_2018,us_data_2017,us_data_2016], axis=0)
us_data_final


# In[133]:


us_data_final.to_csv("us_data_criteria_2.csv", encoding='utf-8', index=False)


# In[135]:


missing_df = us_data_final.isnull().sum(axis=0).reset_index()
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


# In[134]:


f,ax=plt.subplots(1,2,figsize=(18,8))
us_data_final['Severity'].value_counts().plot.pie(autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Percentage Severity Distribution')
ax[0].set_ylabel('Count')
sns.countplot('Severity',data=us_data_final,ax=ax[1],order=us_data_2018['Severity'].value_counts().index)
ax[1].set_title('Count of Severity')
plt.show()

