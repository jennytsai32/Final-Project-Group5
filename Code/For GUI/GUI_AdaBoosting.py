
# importing the required packages
from sklearn.ensemble import AdaBoostClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

# import dataset
# data = pd.read_csv('us_data_criteria_1.csv')  # 12.5 minutes
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

#### Part II: Random Forest using Top 10 Features ++++++++++++++++++++++++++++++++++++####
# AdaBoost - using all features
clf_ab = AdaBoostClassifier(n_estimators=100)
clf_ab.fit(X_train, y_train)

# select the training and test set using top 10 features
newX_train_ab = X_train[:, clf_ab.feature_importances_.argsort()[::-1][:10]]
newX_test_ab = X_test[:, clf_ab.feature_importances_.argsort()[::-1][:10]]

# random forest model - Top 10 features
clf_k_features_ab = AdaBoostClassifier(n_estimators=100)
clf_k_features_ab.fit(newX_train_ab, y_train)

# prediction on test using top 10 features
y_pred_k_features_ab = clf_k_features_ab.predict(newX_test_ab)
y_pred_k_features_score_ab = clf_k_features_ab.predict_proba(newX_test_ab)

# classification report - top 10 features
print("\n")
print("Results Using Top 10 Features: \n")
print("Classification Report: ")
print(classification_report(y_test,y_pred_k_features_ab))
print("\n")
print("Accuracy : ", accuracy_score(y_test, y_pred_k_features_ab) * 100)
print("\n")
print("ROC_AUC : ", roc_auc_score(y_test,y_pred_k_features_score_ab[:,1]) * 100)

# confusion matrix - top 10 features
conf_matrix_ab = confusion_matrix(y_test, y_pred_k_features_ab)
class_names = data['Severity'].unique()
df_cm_ab = pd.DataFrame(conf_matrix_ab, index=class_names, columns=class_names )
plt.figure(figsize=(5,5))
hm_ab = sns.heatmap(df_cm_ab, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=df_cm_ab.columns, xticklabels=df_cm_ab.columns)
hm_ab.yaxis.set_ticklabels(hm_ab.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
hm_ab.xaxis.set_ticklabels(hm_ab.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
plt.ylabel('True label',fontsize=20)
plt.xlabel('Predicted label',fontsize=20)
plt.tight_layout()
plt.show()






