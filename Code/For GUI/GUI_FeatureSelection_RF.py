# importing the required packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

# import dataset
# data = pd.read_csv('us_data_criteria_1.csv') # 15 minutes
data = pd.read_csv('us_data_sample_for_gui.csv') # 3 seconds

#### Part I: Pre-processing before modeling ++++++++++++++++++++++++++++++++++++####
# subsetting - selecting 2018 & 2019
data = data[(data.Year>=2018)]

# drop unnecessary columns
data.drop(['ID', 'Source', 'TMC', 'Start_Time','End_Time',
           'Start_Lat', 'Start_Lng', 'End_Lat', 'End_Lng',
           'Distance(mi)', 'Description', 'Number', 'Street',
           'Side', 'City','County', 'State', 'Zipcode', 'Country',
           'Timezone', 'Airport_Code','Weather_Timestamp',
           'Civil_Twilight', 'Nautical_Twilight', 'Astronomical_Twilight',
           'Turning_Loop','Wind_Direction', 'Weather_Condition',
           'Year','Date', 'Month', 'Day', 'Hour', 'Weekday','Time_Duration(min)',
           'Precipitation(in)','Wind_Chill(F)'], axis=1, inplace=True)

# drop nans
data = data.dropna()

# recode variables before modeling
data['Severity'] = data['Severity'].apply(lambda x:'0' if x =='Low' else '1')
data.iloc[:, 6:19] = data.iloc[:, 6:19].replace({True:"1", False:"0"})
data['Sunrise_Sunset'] = data['Sunrise_Sunset'].replace({"Day":"1", "Night":"0"})
print(data.head())

# define X and Y
Y = data.values[:, 0]
X = data.values[:, 1:19]

# split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=100)


#### Part II: Feature Selection using Random Forest ++++++++++++++++++++++++++++++++++++####
# random forest model - using all features
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

## feature importance plot
# get feature importance
importances = clf.feature_importances_

# convert the importance into one-dimensional array with corresponding df column names as axis labels
f_importances = pd.Series(importances, data.iloc[:, 1:19].columns)

# sort the array in descending order of the importance
f_importances.sort_values(ascending=False, inplace=True)

# make the bar plot from f_importance
f_importances.plot(x='Features', y='Importance', kind='bar', figsize=(16, 9), rot=90, fontsize=15)

# show the plot
plt.tight_layout()
plt.show()


#### Part III: Random Forest using Top 10 Features ++++++++++++++++++++++++++++++++++++####
# select the training and test set using top 10 features
newX_train = X_train[:, clf.feature_importances_.argsort()[::-1][:10]]
newX_test = X_test[:, clf.feature_importances_.argsort()[::-1][:10]]

# random forest model - Top 10 features
clf_k_features = RandomForestClassifier(n_estimators=100)
clf_k_features.fit(newX_train, y_train)

# prediction on test using top 10 features
y_pred_k_features = clf_k_features.predict(newX_test)
y_pred_k_features_score = clf_k_features.predict_proba(newX_test)

# classification report - top 10 features
print("\n")
print("Results Using Top 10 Features: \n")
print("Classification Report: ")
print(classification_report(y_test,y_pred_k_features))
print("\n")
print("Accuracy : ", accuracy_score(y_test, y_pred_k_features) * 100)
print("\n")
print("ROC_AUC : ", roc_auc_score(y_test,y_pred_k_features_score[:,1]) * 100)

# confusion matrix - top 10 features
conf_matrix = confusion_matrix(y_test, y_pred_k_features)
class_names = data['Severity'].unique()
df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names )
plt.figure(figsize=(5,5))
hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=df_cm.columns, xticklabels=df_cm.columns)
hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
plt.ylabel('True label',fontsize=20)
plt.xlabel('Predicted label',fontsize=20)
plt.tight_layout()
plt.show()





