


import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from PyQt5 import uic, QtWidgets, sip, QtCore
from PyQt5.QtCore import Qt



class TableModel(QtCore.QAbstractTableModel):

    def __init__(self, data):
        super(TableModel, self).__init__()
        self._data = data

    def data(self, index, role):
        if role == Qt.DisplayRole:
            value = self._data.iloc[index.row(), index.column()]
            return str(value)

    def rowCount(self, index):
        return self._data.shape[0]

    def columnCount(self, index):
        return self._data.shape[1]

    def headerData(self, section, orientation, role):
        # section is the index of the column/row.
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return str(self._data.columns[section])

            if orientation == Qt.Vertical:
                return str(self._data.index[section])

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
        #self.pushButton_EDA.clicked.connect(self.comboPlot)
        self.pushButton_EDA_sev.clicked.connect(self.severity)
        self.pushButton_EDA_sun.clicked.connect(self.sunrise_sunset)
        self.pushButton_EDA_hum.clicked.connect(self.humidity)
        self.pushButton_EDA_temp.clicked.connect(self.temperature)
        self.pushButton_EDA_pre.clicked.connect(self.pressure)
        self.pushButton_feature.clicked.connect(self.features_imp)
        self.pushButton_rf_bagging.clicked.connect(self.rf_confustion_matrix_rf)
        self.pushButton_ad.clicked.connect(self.rf_confustion_matrix_ad)



    def getCSV(self):
            self.df=pd.read_csv('./final_data.csv')
            #self.df = pd.read_csv('us_data_sample_for_gui.csv')
            stat_st = 'Data Description:'+'\n'+'\n' + str(self.df.describe())
            sample = 'Sample Data:' + '\n' + '\n' + str(self.df.head(5))
            #self.textEdit_output.setText(stat_st)
            self.textEdit_output.setText(stat_st + '\n' + '\n' + sample)
            self.progressBar.setValue(100)
            #use tableWidget to show the data

            self.tableWidget_load.setRowCount(3)
            self.tableWidget_load.setColumnCount(3)

            self.tableWidget_load.setItem(0, 0, QtWidgets.QTableWidgetItem('Data Shape'))
            self.tableWidget_load.setItem(1, 0, QtWidgets.QTableWidgetItem(str(self.df.shape)))

            self.tableWidget_load.setItem(0, 1, QtWidgets.QTableWidgetItem('Features'))
            self.tableWidget_load.setItem(1, 1, QtWidgets.QTableWidgetItem(str(self.df.shape[1])))
            self.tableWidget_load.setItem(0, 2, QtWidgets.QTableWidgetItem('Record count'))
            self.tableWidget_load.setItem(1, 2, QtWidgets.QTableWidgetItem(str(self.df.shape[0])))


    def severity(self):
        labels = ['Low','High']
        severity_count = self.df.groupby("Severity")["Severity"].count()


        self.MplWidget.canvas.axes.clear()
        self.MplWidget.canvas.draw()
        self.MplWidget.canvas.axes.bar(labels,severity_count,color=('orange','blue'),edgecolor='black',width=0.4)
        self.MplWidget.canvas.axes.set_title('Percentage Severity Distribution')
        self.MplWidget.canvas.draw()

    def sunrise_sunset(self):
        labels = ['Day','Night']
        Sunrise_Sunset_Day = self.df[self.df['Sunrise_Sunset'] == 'Day'][['Sunrise_Sunset', 'Severity']].groupby('Severity').count().to_numpy()
        Sunrise_Sunset_Night = self.df[self.df['Sunrise_Sunset'] == 'Night'][['Sunrise_Sunset', 'Severity']].groupby('Severity').count().to_numpy()
        a = Sunrise_Sunset_Day[0, 0]
        b = Sunrise_Sunset_Day[1, 0]
        Sunrise_Sunset_Day = [a, b]
        c = Sunrise_Sunset_Night[0, 0]
        d = Sunrise_Sunset_Night[1, 0]
        Sunrise_Sunset_Night = [c, d]

        x = np.arange(len(labels))  # the label locations
        width = 0.35  # the width of the bars

        self.MplWidget.canvas.axes.clear()
        self.MplWidget.canvas.draw()
        self.MplWidget.canvas.axes.bar(x - width / 2, Sunrise_Sunset_Day, edgecolor='black',width=width,label='Day')
        self.MplWidget.canvas.axes.bar(x + width / 2, Sunrise_Sunset_Night, edgecolor='black', width=width,label='Night')
        #self.MplWidget.canvas.axes.bar(labels,severity_count,color=('orange','blue'),edgecolor='black',width=0.4)
        self.MplWidget.canvas.axes.set_xticklabels(labels)
        self.MplWidget.canvas.axes.legend(('High', 'Low'), loc='center')
        self.MplWidget.canvas.axes.set_title('Sunrise Sunset Per Severity')
        self.MplWidget.canvas.draw()



    def pressure(self):
        temp= self.df["Pressure(in)"].dropna()

        #self.MplWidget.setEnabled(False)
        self.MplWidget.canvas.axes.clear()
        self.MplWidget.canvas.axes.cla()
        self.MplWidget.canvas.draw()
        self.MplWidget.canvas.axes.hist(temp, color='yellow', edgecolor='black',bins=int(180 / 5))
        # self.MplWidget.canvas.axes.legend(('Temperature'), loc='upper right')
        self.MplWidget.canvas.axes.set_title('Pressure(in) Distribution')
        self.MplWidget.canvas.draw()


    def temperature(self):
        temp= self.df["Temperature(F)"].dropna()

        #self.MplWidget.setEnabled(False)
        self.MplWidget.canvas.axes.clear()
        self.MplWidget.canvas.axes.cla()
        self.MplWidget.canvas.draw()
        self.MplWidget.canvas.axes.hist(temp, color='blue', edgecolor='black',bins=int(180 / 5))
        # self.MplWidget.canvas.axes.legend(('Temperature'), loc='upper right')
        self.MplWidget.canvas.axes.set_title('Temperature Distribution')
        self.MplWidget.canvas.draw()

    def humidity(self):
        temp= self.df['Humidity(%)'].dropna()

        #self.MplWidget.setEnabled(False)
        self.MplWidget.canvas.axes.clear()
        self.MplWidget.canvas.axes.cla()
        self.MplWidget.canvas.draw()
        self.MplWidget.canvas.axes.hist(temp, color='green', edgecolor='black',bins=int(180 / 5))
        # self.MplWidget.canvas.axes.legend(('Temperature'), loc='upper right')
        self.MplWidget.canvas.axes.set_title('Humidity Distribution')
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

    def rf_confustion_matrix_rf(self):
            # import dataset
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
            self.textEdit_classification_report.setText(data_result + '\n' + '\n'+'\n' + accuracy + '\n'  + '\n'+ ROC_AUC)


            # confusion matrix - top 10 features
            conf_matrix = confusion_matrix(y_test, y_pred_k_features)
            class_names = data['Severity'].unique()
            df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)

            self.MplWidget_3.canvas.axes.matshow(df_cm)
            self.MplWidget_3.canvas.axes.text(0,0, 243)
            self.MplWidget_3.canvas.axes.text(1,0,46)
            self.MplWidget_3.canvas.axes.text(0,1,105)
            self.MplWidget_3.canvas.axes.text(1,1,62)

            self.MplWidget_3.canvas.axes.set_xlabel('Predicted label', fontsize=15)
            self.MplWidget_3.canvas.axes.set_ylabel('True label', fontsize=15)
            self.MplWidget_3.canvas.axes.set_title('Confusion Matrix')
            self.MplWidget_3.canvas.draw()
            self.MplWidget_3.canvas.update()


    def rf_confustion_matrix_ad(self):
        # import dataset
        # data = pd.read_csv('us_data_criteria_1.csv')  # 12.5 minutes
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


        data_result = 'Data Results Using Top 10 Features:'+'\n'+'\n' + 'Classification Report:'+'\n'+'\n' + str(classification_report(y_test,y_pred_k_features_ab))
        accuracy = 'Accuracy :'+ str(accuracy_score(y_test, y_pred_k_features_ab) * 100)
        ROC_AUC = 'ROC_AUC  :' + str(roc_auc_score(y_test,y_pred_k_features_score_ab[:,1]) * 100)
        self.textEdit_classification_report_ad.setText(data_result + '\n' + '\n' + '\n'+accuracy + '\n'+ '\n' + ROC_AUC)


        # confusion matrix - top 10 features
        conf_matrix = confusion_matrix(y_test, y_pred_k_features_ab)
        class_names = data['Severity'].unique()
        df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)


        self.MplWidget_4.canvas.axes.matshow(df_cm)
        self.MplWidget_4.canvas.axes.text(0,0, 247)
        self.MplWidget_4.canvas.axes.text(1,0,42)
        self.MplWidget_4.canvas.axes.text(0,1,97)
        self.MplWidget_4.canvas.axes.text(1,1,70)

        self.MplWidget_4.canvas.axes.set_xlabel('Predicted label', fontsize=15)
        self.MplWidget_4.canvas.axes.set_ylabel('True label', fontsize=15)
        self.MplWidget_4.canvas.axes.set_title('Confusion Matrix')
        self.MplWidget_4.canvas.draw()
        self.MplWidget_4.canvas.update()

if __name__=="__main__":
    app= QtWidgets.QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())