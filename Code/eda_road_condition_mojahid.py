import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plot
from matplotlib import style
import seaborn as sns
from sklearn import svm
from sklearn import preprocessing

df1=pd.read_csv('./data/US_Accidents_Dec19.csv')
# select road condition features
df=df1[['Description','Start_Time', 'End_Time','City','State', 'Zipcode','Distance(mi)',
        'Side','Weather_Condition','Sunrise_Sunset','Amenity', 'Bump', 'Crossing','Give_Way','Junction',
        'No_Exit','Railway', 'Roundabout', 'Station', 'Stop','Traffic_Calming',
        'Traffic_Signal', 'Turning_Loop','Severity']]

print(df.columns)
print(df.count)

# select all accidents after 2019
#df=df[df['Start_Time']>'2018-31-12 00:00:00']


print(df.count)
print(df1.count)

# create new file for road conditions features 2019
#df.to_csv('./data/us_accid_19.csv')


#df=pd.read_csv('./data/us_accid_19.csv')
print(df.count)
print(df.describe())
print(df.info())
print(df.index)

# convert the features to binaries


label_encoder = preprocessing.LabelEncoder()
df['Amenity']= label_encoder.fit_transform(df['Amenity'])
df['Bump']= label_encoder.fit_transform(df['Bump'])
df['Crossing']= label_encoder.fit_transform(df['Crossing'])
df['Give_Way']= label_encoder.fit_transform(df['Give_Way'])
df['Junction']= label_encoder.fit_transform(df['Junction'])
df['No_Exit']= label_encoder.fit_transform(df['No_Exit'])
df['Railway']= label_encoder.fit_transform(df['Railway'])
df['Roundabout']= label_encoder.fit_transform(df['Roundabout'])
df['Station']= label_encoder.fit_transform(df['Station'])
df['Stop']= label_encoder.fit_transform(df['Stop'])
df['Traffic_Calming']= label_encoder.fit_transform(df['Traffic_Calming'])
df['Traffic_Signal']= label_encoder.fit_transform(df['Traffic_Signal'])
df['Turning_Loop']= label_encoder.fit_transform(df['Turning_Loop'])

#divid severity to 'Low' & 'High'
#df['Severity']=df['Severity'].apply(lambda x: 'Low' if x==2 else 'High')
#print(df['Severity'].head(3))

print(df.head(5))
print(' Plot the Severity High/Low')
sns.countplot(x='Severity', data=df)
plot.show()

data=['Amenity','Bump','Crossing','Give_Way','Junction','No_Exit','Railway','Roundabout','Station','Stop','Traffic_Calming','Traffic_Signal','Turning_Loop']
print('######################################')


#dataFrame.plot.bar(rot=15, title="Car Price vs Car Weight comparision for Sedans made by a Car Company");

plot.show(block=True);




#sns.lmplot('Junction','No_Exit',data=df, hue='Type', palette='Set1', scatter_kws={'s':70})

feature_cols = ['Amenity','Bump','Crossing','Give_Way','Junction','No_Exit','Railway','Roundabout','Station','Stop','Traffic_Calming','Traffic_Signal','Turning_Loop']
target_col =['Severity']
print(df[feature_cols])
data=df[feature_cols]
data['target']=df[target_col]

print(data.info())
print(data.head(3))
print(data['target'].unique())

data1 =data[data.target=='High']
data2 =data[data.target=='Low']

print(data1['Junction'])
plt.scatter(data1['Junction'],data1['No_Exit'],color='green',marker='+')
#plt.scatter(data2['Junction'],data2['No_Exit'],color='blue',marker='+')
plt.draw()
plt.show()
