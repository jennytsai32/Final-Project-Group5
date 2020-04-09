

import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt




us_data_2019= pd.read_csv('us_data_2019_draft_2.csv')
us_data_2019.head()




us_data_2019["Weather_Condition"]= us_data_2019["Weather_Condition"].astype(str) 
us_data_2019["Wind_Direction"]= us_data_2019["Wind_Direction"].astype(str) 
us_data_2019["Severity"]= us_data_2019["Severity"].astype(str) 
us_data_2019["Month"]= us_data_2019["Month"].astype(str) 




from sklearn import preprocessing 
label_encoder = preprocessing.LabelEncoder() 
  
us_data_2019['Weather_Condition']= label_encoder.fit_transform(us_data_2019['Weather_Condition'])
us_data_2019['Wind_Direction']= label_encoder.fit_transform(us_data_2019['Wind_Direction']) 
us_data_2019['Severity']= label_encoder.fit_transform(us_data_2019['Severity']) 


us_data_2019.head()




W= [ "Start_Lat","Start_Lng","Severity",
        'Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)',
       'Pressure(in)', 'Visibility(mi)', 'Wind_Direction', 'Wind_Speed(mph)',
       'Precipitation(in)', 'Weather_Condition'
       ]
        
weather_data=us_data_2019[W]





weather_data=weather_data.dropna()




features=[ "Start_Lat","Start_Lng",
        'Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)',
       'Pressure(in)', 'Visibility(mi)', 'Wind_Direction', 'Wind_Speed(mph)',
       'Precipitation(in)', 'Weather_Condition'
       ]

X=weather_data[features]
target=["Severity"]
Y=weather_data[target]




from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1) # 70% training and 30% test




from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression()
logmodel.fit(X_train,y_train)




y_pred=logmodel.predict(X_test)




print("Accuracy:",logmodel.score(X_test, y_test))




from sklearn.metrics import confusion_matrix
confusion_matrix=confusion_matrix(y_test,y_pred)
print(confusion_matrix)




fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(confusion_matrix)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, confusion_matrix[i, j], ha='center', va='center', color='red')
plt.show()






