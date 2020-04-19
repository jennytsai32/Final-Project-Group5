#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing the required packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score


# In[2]:


# import Dataset
# read data as panda dataframe
data = pd.read_csv('us_data_final_1.csv')

data.columns


# In[3]:


# drop unnecessary columns
data.drop(['ID', 'Source', 'TMC', 'Start_Time',
       'End_Time', 'Start_Lat', 'Start_Lng', 'End_Lat', 'End_Lng',
       'Distance(mi)', 'Description', 'Number', 'Street', 'Side', 'City',
       'County', 'State', 'Zipcode', 'Country', 'Timezone', 'Airport_Code',
       'Weather_Timestamp','Civil_Twilight', 'Nautical_Twilight', 'Astronomical_Twilight', 'Turning_Loop'], axis=1, inplace=True)

data.columns


# In[4]:


# printing the dataset rows and columns
print("Dataset No. of Rows: ", data.shape[0])
print("Dataset No. of Columns: ", data.shape[1])
print("")
# printing the dataset obseravtions
print("Dataset first few rows:\n ")
print(data.head(5))


# In[5]:


# printing the struture of the dataset
print("Dataset info:\n ")
print(data.info())


# In[6]:


# printing the summary statistics of the dataset
print(data.describe(include='all'))


# In[7]:


# Missing Values
print("Sum of NULL values in each column. ")
print(data.isnull().sum())


# In[8]:


# drop unnecessary columns
data.drop(['Wind_Direction', 'Weather_Condition'], axis=1, inplace=True)

var_lst = np.array(list(zip(data.index, data.columns)))
print(var_lst)


# In[9]:


# Check frequency
severity_freq = data.groupby('Severity').Severity.count()
print(severity_freq)
print()

for c in data.iloc[:, 8:22].columns:
    freq = data.groupby(c)[c].count()
    print(freq)
    print()


# In[10]:


# Drop nans
data = data.dropna()


# In[11]:


# Recode variables
data['Severity'] = data['Severity'].apply(lambda x:'0' if x =='Low' else '1')
data.iloc[:, 8:20] = data.iloc[:, 8:20].replace({True:"1", False:"0"})
data['Sunrise_Sunset'] = data['Sunrise_Sunset'].replace({"Day":"1", "Night":"0"})
print(data.head())


# In[12]:


# Set X and Y variable
Y = data.values[:, 0]
X = data.values[:, 1:22]


# In[13]:


# split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=100)


# In[14]:


# perform training with random forest with all columns
# specify random forest classifier
clf = RandomForestClassifier(n_estimators=100)

# perform training
clf.fit(X_train, y_train)


# In[15]:


# plot feature importances
# get feature importances
importances = clf.feature_importances_

# convert the importances into one-dimensional 1darray with corresponding df column names as axis labels
f_importances = pd.Series(importances, data.iloc[:, 1:22].columns)

# sort the array in descending order of the importances
f_importances.sort_values(ascending=False, inplace=True)

# make the bar Plot from f_importances
f_importances.plot(x='Features', y='Importance', kind='bar', figsize=(16, 9), rot=90, fontsize=15)

# show the plot
plt.tight_layout()
plt.show()


# In[16]:


# select features to perform training with random forest with k columns
# select the training dataset on k-features
newX_train = X_train[:, clf.feature_importances_.argsort()[::-1][:10]]

# select the testing dataset on k-features
newX_test = X_test[:, clf.feature_importances_.argsort()[::-1][:10]]


# In[17]:


# perform training with random forest with k columns
# specify random forest classifier
# n_estimators=100: The number of trees in the forest.
# bootstrap: boolean, optional (default=True)
clf_k_features = RandomForestClassifier(n_estimators=100)

# train the model
clf_k_features.fit(newX_train, y_train)


# In[18]:


# make predictions

# predicton on test using all features
y_pred = clf.predict(X_test)
y_pred_score = clf.predict_proba(X_test)

# prediction on test using k features
y_pred_k_features = clf_k_features.predict(newX_test)
y_pred_k_features_score = clf_k_features.predict_proba(newX_test)


# In[19]:


# Classification report - all features

print("\n")
print("Results Using All Features: \n")

print("Classification Report: ")
print(classification_report(y_test,y_pred))
print("\n")

print("Accuracy : ", accuracy_score(y_test, y_pred) * 100)
print("\n")

print("ROC_AUC : ", roc_auc_score(y_test,y_pred_score[:,1]) * 100)

# Classification report - k features
print("\n")
print("Results Using K features: \n")
print("Classification Report: ")
print(classification_report(y_test,y_pred_k_features))
print("\n")
print("Accuracy : ", accuracy_score(y_test, y_pred_k_features) * 100)
print("\n")
print("ROC_AUC : ", roc_auc_score(y_test,y_pred_k_features_score[:,1]) * 100)


# In[20]:


# confusion matrix - all features
conf_matrix = confusion_matrix(y_test, y_pred)
class_names = data['Severity'].unique()


df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names )

plt.figure(figsize=(5,5))

hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=df_cm.columns, xticklabels=df_cm.columns)

hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
plt.ylabel('True label',fontsize=20)
plt.xlabel('Predicted label',fontsize=20)
# Show heat map
plt.tight_layout()


# In[21]:


# confusion matrix - k features

conf_matrix = confusion_matrix(y_test, y_pred_k_features)
class_names = data['Severity'].unique()


df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names )

plt.figure(figsize=(5,5))

hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=df_cm.columns, xticklabels=df_cm.columns)

hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
plt.ylabel('True label',fontsize=20)
plt.xlabel('Predicted label',fontsize=20)
# Show heat map
plt.tight_layout()
plt.show()


# In[22]:


data.to_csv('df_new.csv')


# In[ ]:




