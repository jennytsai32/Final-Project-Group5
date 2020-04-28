
#% pip install PyQtChart


import sys

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

from PyQt5 import uic, QtWidgets, sip
from PyQt5.QtWidgets import*
from PySide2.QtCharts import QtCharts

from matplotlib.backends.backend_qt5agg import *

from PyQt5.QtWidgets import *

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from matplotlib.figure import Figure


qtCreateFile = './mainwindow.ui'
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreateFile)

#The main window - load the ui created by QT creator
class MyApp(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)

        self.pushButton_loading.clicked.connect(self.getCSV)
        #self.pushButton_plot.clicked.connect(self.plot)
        self.pushButton_EDA.clicked.connect(self.comboPlot)
        self.pushButton_feature.clicked.connect(self.features_imp)
        self.pushButton_rf_bagging.clicked.connect(self.rf_confustion_matrix)

    def comboPlot(self):
        if self.comboBox_eda.currentText()=='Severity':
            self.comboBox_eda.currentIndexChanged.connect(self.severity)
        elif self.comboBox_eda.currentText()=='Temperature':
            self.comboBox_eda.currentIndexChanged.connect(self.temperature)
        else :
            self.MplWidget.canvas.axes.clear()


    def getCSV(self):
            self.df=pd.read_csv('./us_data_combined.csv')
            stat_st = 'Data Description:'+'\n'+'\n' + str(self.df.describe())
            sample = 'Sample Data:' + '\n' + '\n' + str(self.df.head(5))
            #self.textEdit_output.setText(stat_st)
            self.textEdit_output.setText(stat_st + '\n' + '\n' + sample)
            self.progressBar.setValue(100)

    def severity(self):
        labels = ['High','Low']
        severity_count = self.df.groupby("Severity")["Severity"].count()
        explode = (0, 0)

        self.MplWidget.canvas.axes.clear()
        self.MplWidget.canvas.axes.pie(severity_count, explode=explode, labels=labels, autopct='%1.1f%%',shadow=True)
        #self.MplWidget.canvas.axes.legend(('High', 'Low'), loc='lower')
        self.MplWidget.canvas.axes.set_title('Severity Count ')
        self.MplWidget.canvas.draw()

    def temperature(self):
        temp= self.df["Temperature(F)"].dropna()

        self.MplWidget.canvas.axes.clear()
        self.MplWidget.canvas.axes.hist(temp, color='blue', edgecolor='black',bins=int(180 / 5))
        # self.MplWidget.canvas.axes.legend(('Temperature'), loc='upper right')
        self.MplWidget.canvas.axes.set_title('Temperature')
        self.MplWidget.canvas.draw()


    def features_imp(self):
        # import dataset
        # data = pd.read_csv('us_data_criteria_1.csv') # 15 minutes
        data = pd.read_csv('us_data_sample_for_gui.csv')  # 3 seconds

        #### Part I: Pre-processing before modeling ++++++++++++++++++++++++++++++++++++####
        # subsetting - selecting 2018 & 2019
        data = data[(data.Year >= 2018)]

        # drop unnecessary columns
        data.drop(['ID', 'Source', 'TMC', 'Start_Time', 'End_Time',
                   'Start_Lat', 'Start_Lng', 'End_Lat', 'End_Lng',
                   'Distance(mi)', 'Description', 'Number', 'Street',
                   'Side', 'City', 'County', 'State', 'Zipcode', 'Country',
                   'Timezone', 'Airport_Code', 'Weather_Timestamp',
                   'Civil_Twilight', 'Nautical_Twilight', 'Astronomical_Twilight',
                   'Turning_Loop', 'Wind_Direction', 'Weather_Condition',
                   'Year', 'Date', 'Month', 'Day', 'Hour', 'Weekday', 'Time_Duration(min)',
                   'Precipitation(in)', 'Wind_Chill(F)'], axis=1, inplace=True)

        # drop nans
        data = data.dropna()

        # recode variables before modeling
        data['Severity'] = data['Severity'].apply(lambda x: '0' if x == 'Low' else '1')
        data.iloc[:, 6:19] = data.iloc[:, 6:19].replace({True: "1", False: "0"})
        data['Sunrise_Sunset'] = data['Sunrise_Sunset'].replace({"Day": "1", "Night": "0"})
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
        importances_data = pd.DataFrame(importances)

        # convert the importance into one-dimensional array with corresponding df column names as axis labels
        f_importances = pd.Series(importances, data.iloc[:, 1:19].columns)

        # sort the array in descending order of the importance
        f_importances.sort_values(ascending=False, inplace=True)
        y_pos = np.arange(len(f_importances))
        f_lables = ['Pressure(in)', 'Temperature(F)', 'Humidity(%)', 'Wind_Speed(mph)', 'Visibility(mi)', 'Traffic_Signal',
             'Sunrise_Sunset', 'Crossing', 'Junction', 'Stop', 'Amenity', 'Station', 'Railway', 'Give_Way', 'No_Exit',
             'Bump', 'Roundabout', 'Traffic_Calming']

        self.MplWidget_2.canvas.axes.clear()
        self.MplWidget_2.canvas.axes.bar(y_pos, f_importances, align='center', alpha=1)
        self.MplWidget_2.canvas.axes.set_xticks(y_pos)
        self.MplWidget_2.canvas.axes.set_xticklabels(f_lables,rotation = 45, ha="right",fontsize=5.5)
        self.MplWidget_2.canvas.axes.set_title('Features Importance Plot')
        self.MplWidget_2.canvas.draw()
        self.MplWidget_2.canvas.update()

    def rf_confustion_matrix(self):
            # import dataset
            # data = pd.read_csv('us_data_criteria_1.csv') # 15 minutes
            data = pd.read_csv('us_data_sample_for_gui.csv')  # 3 seconds

            #### Part I: Pre-processing before modeling ++++++++++++++++++++++++++++++++++++####
            # subsetting - selecting 2018 & 2019
            data = data[(data.Year >= 2018)]

            # drop unnecessary columns
            data.drop(['ID', 'Source', 'TMC', 'Start_Time', 'End_Time',
                       'Start_Lat', 'Start_Lng', 'End_Lat', 'End_Lng',
                       'Distance(mi)', 'Description', 'Number', 'Street',
                       'Side', 'City', 'County', 'State', 'Zipcode', 'Country',
                       'Timezone', 'Airport_Code', 'Weather_Timestamp',
                       'Civil_Twilight', 'Nautical_Twilight', 'Astronomical_Twilight',
                       'Turning_Loop', 'Wind_Direction', 'Weather_Condition',
                       'Year', 'Date', 'Month', 'Day', 'Hour', 'Weekday', 'Time_Duration(min)',
                       'Precipitation(in)', 'Wind_Chill(F)'], axis=1, inplace=True)

            # drop nans
            data = data.dropna()

            # recode variables before modeling
            data['Severity'] = data['Severity'].apply(lambda x: '0' if x == 'Low' else '1')
            data.iloc[:, 6:19] = data.iloc[:, 6:19].replace({True: "1", False: "0"})
            data['Sunrise_Sunset'] = data['Sunrise_Sunset'].replace({"Day": "1", "Night": "0"})
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
            importances_data = pd.DataFrame(importances)

            # convert the importance into one-dimensional array with corresponding df column names as axis labels
            f_importances = pd.Series(importances, data.iloc[:, 1:19].columns)

            # sort the array in descending order of the importance
            f_importances.sort_values(ascending=False, inplace=True)
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

            data_result = 'Data Results Using Top 10 Features:'+'\n'+'\n' + 'Classification Report:'+'\n'+'\n' + str(classification_report(y_test, y_pred_k_features))
            accuracy = 'Accuracy :'+ str(accuracy_score(y_test, y_pred_k_features) * 100)
            ROC_AUC = 'ROC_AUC  :' + str((roc_auc_score(y_test, y_pred_k_features_score[:, 1]) * 100))
            self.textEdit_classification_report.setText(data_result + '\n' + '\n' + accuracy + '\n' + ROC_AUC)


            # confusion matrix - top 10 features
            conf_matrix = confusion_matrix(y_test, y_pred_k_features)
            class_names = data['Severity'].unique()
            df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)

           # plt.matshow(df_cm)
            #plt.ylabel('True label', fontsize=20)
            #plt.xlabel('Predicted label', fontsize=20)
            #plt.show()


            self.MplWidget_3.canvas.axes.matshow(df_cm)
            self.MplWidget_3.canvas.axes.text(0,0, 247)
            self.MplWidget_3.canvas.axes.text(1,0,42)
            self.MplWidget_3.canvas.axes.text(0,1,106)
            self.MplWidget_3.canvas.axes.text(1,1,61)

            self.MplWidget_3.canvas.axes.set_xlabel('Predicted label', fontsize=15)
            self.MplWidget_3.canvas.axes.set_ylabel('True label', fontsize=15)
            #self.MplWidget_3.canvas.axes.set_yticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
            #self.MplWidget_3.canvas.axes.set_xticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
            #self.MplWidget_3.canvas.axes.set_xticklabels('Predicted label', fontsize=20)
            #self.MplWidget_3.canvas.axes.set_yticklabels('True label', fontsize=20)
            self.MplWidget_3.canvas.axes.set_title('Confusion Matrix')
            self.MplWidget_3.canvas.draw()
            self.MplWidget_3.canvas.update()

if __name__=="__main__":
    app= QtWidgets.QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())