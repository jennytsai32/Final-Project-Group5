


#%pip install missingno




import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999
import missingno as msno 




us_data_2019= pd.read_csv('US_Accidents_2019.csv')
us_data_2019.head()




us_data_2019['Start_Time'] = pd.to_datetime(us_data_2019['Start_Time'], errors='coerce')
us_data_2019['End_Time'] = pd.to_datetime(us_data_2019['End_Time'], errors='coerce')

# Extract year, month, day, hour and weekday
us_data_2019["Date"]=us_data_2019["Start_Time"].dt.date
us_data_2019['Year']=us_data_2019['Start_Time'].dt.year
us_data_2019['Month']=us_data_2019['Start_Time'].dt.strftime('%b')
us_data_2019['Day']=us_data_2019['Start_Time'].dt.day
us_data_2019['Hour']=us_data_2019['Start_Time'].dt.hour
us_data_2019['Weekday']=us_data_2019['Start_Time'].dt.strftime('%a')

# Extract the amount of time in the unit of minutes for each accident, round to the nearest integer
td='Time_Duration(min)'
us_data_2019[td]=round((us_data_2019['End_Time']-us_data_2019['Start_Time'])/np.timedelta64(1,'m'))





#l= [ 'TMC', 'Severity', 'Start_Time',"ID",
       #'End_Time',
       #'Distance(mi)','City',
       #'Weather_Timestamp', 'Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)',
       #'Pressure(in)', 'Visibility(mi)', 'Wind_Direction', 'Wind_Speed(mph)',"State",
       #'Precipitation(in)', 'Weather_Condition',"Date", 'Hour','Weekday',"Month", 'Time_Duration(min)'
       #]




#us_data_2019=us_data_2019[l]




us_data_2019["Severity"]=us_data_2019["Severity"].replace({1: "Low", 2: "Low", 3: "High", 4:"High"})
us_data_2019




f,ax=plt.subplots(1,2,figsize=(18,8))
us_data_2019['Severity'].value_counts().plot.pie(autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Percentage Severity Distribution')
ax[0].set_ylabel('Count')
sns.countplot('Severity',data=us_data_2019,ax=ax[1],order=us_data_2019['Severity'].value_counts().index)
ax[1].set_title('Count of Severity')
plt.show()




msno.matrix(us_data_2019) 



msno.heatmap(us_data_2019) 




missing_df = us_data_2019.isnull().sum(axis=0).reset_index()
missing_df.columns = ['columns_name','missing_count']
missing_df['missing_ratio'] = missing_df['missing_count'] /us_data_2019.shape[0]
missing_df.loc[missing_df['missing_ratio']>0.1]




missing_df = us_data_2019.isnull().sum(axis=0).reset_index()
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



us_data_2019['Wind_Chill(F)'] = us_data_2019['Wind_Chill(F)'].fillna(us_data_2019.groupby(["Date","City"])['Wind_Chill(F)'].transform('mean'))
us_data_2019['Precipitation(in)'] = us_data_2019['Precipitation(in)'].fillna(us_data_2019.groupby(["Date","City"])['Precipitation(in)'].transform('mean'))
us_data_2019['Pressure(in)'] = us_data_2019['Pressure(in)'].fillna(us_data_2019.groupby(["Date","City"])['Pressure(in)'].transform('mean'))
us_data_2019['Temperature(F)'] = us_data_2019['Temperature(F)'].fillna(us_data_2019.groupby(["Date","City"])['Temperature(F)'].transform('mean'))
us_data_2019['Humidity(%)'] = us_data_2019['Humidity(%)'].fillna(us_data_2019.groupby(["Date","City"])['Humidity(%)'].transform('mean'))
us_data_2019['Visibility(mi)'] = us_data_2019['Visibility(mi)'].fillna(us_data_2019.groupby(["Date","City"])['Visibility(mi)'].transform('mean'))
us_data_2019['Wind_Speed(mph)'] = us_data_2019['Wind_Speed(mph)'].fillna(us_data_2019.groupby(["Date","City"])['Wind_Speed(mph)'].transform('mean'))



us_data_2019['Wind_Chill(F)'] = us_data_2019['Wind_Chill(F)'].fillna(us_data_2019.groupby(["Date","State"])['Wind_Chill(F)'].transform('mean'))
us_data_2019['Precipitation(in)'] = us_data_2019['Precipitation(in)'].fillna(us_data_2019.groupby(["Date","State"])['Precipitation(in)'].transform('mean'))
us_data_2019['Pressure(in)'] = us_data_2019['Pressure(in)'].fillna(us_data_2019.groupby(["Date","State"])['Pressure(in)'].transform('mean'))
us_data_2019['Temperature(F)'] = us_data_2019['Temperature(F)'].fillna(us_data_2019.groupby(["Date","State"])['Temperature(F)'].transform('mean'))
us_data_2019['Humidity(%)'] = us_data_2019['Humidity(%)'].fillna(us_data_2019.groupby(["Date","State"])['Humidity(%)'].transform('mean'))
us_data_2019['Visibility(mi)'] = us_data_2019['Visibility(mi)'].fillna(us_data_2019.groupby(["Date","State"])['Visibility(mi)'].transform('mean'))
us_data_2019['Wind_Speed(mph)'] = us_data_2019['Wind_Speed(mph)'].fillna(us_data_2019.groupby(["Date","State"])['Wind_Speed(mph)'].transform('mean'))



us_data_2019['Wind_Chill(F)'] = us_data_2019['Wind_Chill(F)'].fillna(us_data_2019.groupby(["Month","State"])['Wind_Chill(F)'].transform('mean'))
us_data_2019['Precipitation(in)'] = us_data_2019['Precipitation(in)'].fillna(us_data_2019.groupby(["Month","State"])['Precipitation(in)'].transform('mean'))
us_data_2019['Pressure(in)'] = us_data_2019['Pressure(in)'].fillna(us_data_2019.groupby(["Month","State"])['Pressure(in)'].transform('mean'))
us_data_2019['Temperature(F)'] = us_data_2019['Temperature(F)'].fillna(us_data_2019.groupby(["Month","State"])['Temperature(F)'].transform('mean'))
us_data_2019['Humidity(%)'] = us_data_2019['Humidity(%)'].fillna(us_data_2019.groupby(["Month","State"])['Humidity(%)'].transform('mean'))
us_data_2019['Visibility(mi)'] = us_data_2019['Visibility(mi)'].fillna(us_data_2019.groupby(["Month","State"])['Visibility(mi)'].transform('mean'))
us_data_2019['Wind_Speed(mph)'] = us_data_2019['Wind_Speed(mph)'].fillna(us_data_2019.groupby(["Month","State"])['Wind_Speed(mph)'].transform('mean'))




msno.matrix(us_data_2019) 




missing_df = us_data_2019.isnull().sum(axis=0).reset_index()
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




states = us_data_2019.State.unique()
count_by_state=[]
for i in us_data_2019.State.unique():
    count_by_state.append(us_data_2019[us_data_2019['State']==i].count()['ID'])

fig,ax = plt.subplots(figsize=(16,10))
sns.barplot(states,count_by_state)








lst=list(us_data_2019.City.unique())
print(lst)
print(len(lst))




us_data_2019.Weather_Condition.unique()




us_data_2019["Weather_Condition"]=us_data_2019["Weather_Condition"].replace(
    dict.fromkeys(['Mostly Cloudy','Partly Cloudy','Funnel Cloud', 'Scattered Clouds', 'Cloudy / Windy'], 'Cloudy'), 
    regex=True
)

us_data_2019["Weather_Condition"]= us_data_2019["Weather_Condition"].replace(
    dict.fromkeys(['Patches of Fog','Shallow Fog','Drizzle and Fog', "Partial Fog", "Light Freezing Fog", 'Fog / Windy'], 'Fog'), 
    regex=True
)

us_data_2019["Weather_Condition"]= us_data_2019["Weather_Condition"].replace(
    dict.fromkeys(['Light Rain','Heavy Rain','Heavy Rain / Windy', 'Light Freezing Rain', 'Showers in the Vicinity',
                   'Light Rain Shower','Freezing Rain', 'Rain / Windy', 
                   'Light Freezing Rain / Windy', 
                   'Light Rain Shower / Windy', 'Rain Shower', 'Light Rain Showers','Rains','Rain / Windy', 'N/A Precipitation', 'Rain / Windy', 'Rain / Windy'  ], 'Rain'), 
    regex=True
)

us_data_2019["Weather_Condition"]= us_data_2019["Weather_Condition"].replace(
    dict.fromkeys(['Light Snow','Heavy Snow','Snow / Windy', 'Light Snow / Windy', 'Snow and Sleet',
                   'Light Snow Shower','Light Snow and Sleet', 'Light Snow Grains',
                 'Light Snow and Sleet / Windy', 'Heavy Snow with Thunder',
                   'Snow and Thunder', 'Snow and Sleet / Windy', 'Light Snow Showers',
                   'Heavy Blowing Snow','Ice Pellets', 'Low Drifting Snow', 'Snow / Windy',
                   'Blowing Snow','Light Snow', 'Light Snow', "Hail", "Small Hail", 'Snow Shower',
                   'Snow Grains', 'Small Snow', 'Snow with T-Storm', 'Snows', 'Small Snow' , 'Snow with T-Storm', 'Snow / Windy', 'Light Snow', 'Light T-Storms and Snow','Small Snow','Snow with T-Storm','Light Snow','Snows','T-Storm and Snow / Windy','Light T-Storms and Snow' ], 'Snow'), 
    regex=True
)


us_data_2019["Weather_Condition"]= us_data_2019["Weather_Condition"].replace(
    dict.fromkeys(['Fair / Windy','Smoke / Windy', 'Wintry Mix / Windy',
                'Light Drizzle / Windy', 'Squalls / Windy', "Squalls" ], 'Windy'), 
    regex=True
)


us_data_2019["Weather_Condition"]= us_data_2019["Weather_Condition"].replace(
    dict.fromkeys(['Drizzle','Light Drizzle', 'Heavy Drizzle',
                'Drizzle / Windy', 'Light Freezing Drizzle', 'Heavy Freezing Drizzle'], 'Drizzle'), 
    regex=True
)


us_data_2019["Weather_Condition"]= us_data_2019["Weather_Condition"].replace(
    dict.fromkeys(['Heavy T-Storm','Rain with Thunder', 'Thunder',
                'Thunder in the Vicinity', 'Thunder / Windy', 'Heavy T-Storm / Windy',
                   'Light Thunderstorms and Rain', 'Heavy Thunderstorms and Rain',
                   'Thunder and Hail / Windy', 'Light Thunderstorms and Snow' , 
                  'Light T-Stormstorms and Rain', 'T-Storm in the Vicinity',
       'Heavy T-Stormstorms and Rain', 'T-Stormstorms and Rain', 'T-Storm / Windy', 'T-Storm in the Vicinity', 'Snow with T-Storm'
                  'T-Storm and Snow / Windy', 'T-Stormstorm', 'T-Storm and Snow', 'T-Storm in the Vicinity', 'T-Storm / Windy','T-Stormstorm', 'Light T-Stormstorms and Rain', 'Heavy T-Stormstorms and Rain' ], 'T-Storm'), 
    regex=True
)



us_data_2019["Weather_Condition"]= us_data_2019["Weather_Condition"].replace(
    dict.fromkeys(['Sleet', 'Light Sleet', 'Heavy Sleet'], 'Sleet'), 
    regex=True
)


us_data_2019["Weather_Condition"]= us_data_2019["Weather_Condition"].replace(
    dict.fromkeys(['Sand / Dust Whirlwinds', 'Blowing Dust', 'Widespread Dust / Windy', 
                   'Sand / Dust Whirlwinds / Windy', 'Widespread Dust', 'Blowing Dust / Windy', 'Dust / Windy','Dust / Windy'], 'Dust'), 
    regex=True
)


us_data_2019["Weather_Condition"]= us_data_2019["Weather_Condition"].replace(
    dict.fromkeys(['Haze / Windy'], 'Haze'), 
    regex=True
)




us_data_2019.Weather_Condition.unique()




fig, ax=plt.subplots(figsize=(16,7))
us_data_2019['Weather_Condition'].value_counts().sort_values(ascending=False).plot.bar(width=0.5,edgecolor='k',align='center',linewidth=2)
plt.xlabel('Weather_Condition',fontsize=20)
plt.ylabel('Number of Accidents',fontsize=20)
ax.tick_params(labelsize=20)
plt.title('Weather Condition for accidents',fontsize=25)
plt.grid()
plt.ioff()




us_data_2019.Wind_Direction.unique()




us_data_2019.to_csv("us_data_2019_draft.csv", encoding='utf-8', index=False)




us_data_2019= pd.read_csv('us_data_2019_draft.csv')
us_data_2019.head()




#sns.catplot(x="Date", y="", hue="Wind_Direction", kind="swarm", data=us_data_2019);
f, ax = plt.subplots(figsize=(18, 8))
sns.countplot(y="Wind_Direction", data=us_data_2019, color="c");




us_data_2019["Wind_Direction"]=us_data_2019["Wind_Direction"].replace({"E": "East", "W": "West", "N": "North", "S":"South","CALM":"Calm","VAR":"Variable"})




f, ax = plt.subplots(figsize=(18, 8))
sns.countplot(y="Wind_Direction", data=us_data_2019, color="c");









us_data_2019['Weather_Condition'] = us_data_2019['Weather_Condition'].fillna(us_data_2019.groupby(["Date","City"])['Weather_Condition'].ffill())
us_data_2019['Wind_Direction'] = us_data_2019['Wind_Direction'].fillna(us_data_2019.groupby(["Date","City"])['Wind_Direction'].ffill())




us_data_2019['Weather_Condition'] = us_data_2019['Weather_Condition'].fillna(us_data_2019.groupby(["Date","State"])['Weather_Condition'].ffill())
us_data_2019['Wind_Direction'] = us_data_2019['Wind_Direction'].fillna(us_data_2019.groupby(["Date","State"])['Wind_Direction'].ffill())




us_data_2019["Weather_Timestamp"]=us_data_2019["Weather_Timestamp"].ffill()




missing_df = us_data_2019.isnull().sum(axis=0).reset_index()
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




fig, ax=plt.subplots(figsize=(16,7))
us_data_2019['Weather_Condition'].value_counts().sort_values(ascending=False).plot.bar(width=0.5,edgecolor='k',align='center',linewidth=2)
plt.xlabel('Weather_Condition',fontsize=20)
plt.ylabel('Number of Accidents',fontsize=20)
ax.tick_params(labelsize=20)
plt.title('Weather Condition for accidents',fontsize=25)
plt.grid()
plt.ioff()




f, ax = plt.subplots(figsize=(18, 8))
sns.countplot(y="Wind_Direction", data=us_data_2019, color="c");

