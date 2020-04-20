# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(820, 600)
        MainWindow.setUnifiedTitleAndToolBarOnMac(False)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(40, 10, 631, 541))
        self.tabWidget.setObjectName("tabWidget")
        self.tab_loading = QtWidgets.QWidget()
        self.tab_loading.setObjectName("tab_loading")
        self.pushButton_loading = QtWidgets.QPushButton(self.tab_loading)
        self.pushButton_loading.setGeometry(QtCore.QRect(200, 30, 191, 41))
        self.pushButton_loading.setObjectName("pushButton_loading")
        self.textEdit_output = QtWidgets.QTextEdit(self.tab_loading)
        self.textEdit_output.setGeometry(QtCore.QRect(30, 110, 591, 391))
        self.textEdit_output.setObjectName("textEdit_output")
        self.tabWidget.addTab(self.tab_loading, "")
        self.tab_eda = QtWidgets.QWidget()
        self.tab_eda.setObjectName("tab_eda")
        self.pushButton_EDA = QtWidgets.QPushButton(self.tab_eda)
        self.pushButton_EDA.setGeometry(QtCore.QRect(110, 10, 112, 32))
        self.pushButton_EDA.setObjectName("pushButton_EDA")
        self.MplWidget = MplWidget(self.tab_eda)
        self.MplWidget.setGeometry(QtCore.QRect(69, 70, 611, 361))
        self.MplWidget.setMinimumSize(QtCore.QSize(400, 300))
        self.MplWidget.setObjectName("MplWidget")
        self.pushButton_EDA2 = QtWidgets.QPushButton(self.tab_eda)
        self.pushButton_EDA2.setGeometry(QtCore.QRect(260, 10, 112, 32))
        self.pushButton_EDA2.setObjectName("pushButton_EDA2")
        self.tabWidget.addTab(self.tab_eda, "")
        self.tab_preprocessing = QtWidgets.QWidget()
        self.tab_preprocessing.setObjectName("tab_preprocessing")
        self.tableView = QtWidgets.QTableView(self.tab_preprocessing)
        self.tableView.setGeometry(QtCore.QRect(25, 31, 611, 451))
        self.tableView.setObjectName("tableView")
        self.tabWidget.addTab(self.tab_preprocessing, "")
        self.tab_features = QtWidgets.QWidget()
        self.tab_features.setObjectName("tab_features")
        self.tabWidget.addTab(self.tab_features, "")
        self.tab_modeling = QtWidgets.QWidget()
        self.tab_modeling.setObjectName("tab_modeling")
        self.tabWidget.addTab(self.tab_modeling, "")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 820, 22))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionOpen = QtWidgets.QAction(MainWindow)
        self.actionOpen.setObjectName("actionOpen")
        self.actionSave = QtWidgets.QAction(MainWindow)
        self.actionSave.setObjectName("actionSave")
        self.actionExit = QtWidgets.QAction(MainWindow)
        self.actionExit.setObjectName("actionExit")
        self.actionExit_2 = QtWidgets.QAction(MainWindow)
        self.actionExit_2.setObjectName("actionExit_2")
        self.menuFile.addAction(self.actionOpen)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionSave)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionExit_2)
        self.menubar.addAction(self.menuFile.menuAction())

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(1)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Traffic Accident in USA  Predictor"))
        self.pushButton_loading.setText(_translate("MainWindow", "Loading Data"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_loading), _translate("MainWindow", "Loading"))
        self.pushButton_EDA.setText(_translate("MainWindow", "EDA Plot"))
        self.pushButton_EDA2.setText(_translate("MainWindow", "EDA Plot 2"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_eda), _translate("MainWindow", "EDA"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_preprocessing), _translate("MainWindow", "Preprocessing"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_features), _translate("MainWindow", "Feature Selection"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_modeling), _translate("MainWindow", "Modeling"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.actionOpen.setText(_translate("MainWindow", "Open"))
        self.actionSave.setText(_translate("MainWindow", "Save"))
        self.actionExit.setText(_translate("MainWindow", "Exit"))
        self.actionExit_2.setText(_translate("MainWindow", "Exit"))
from mplwidget import MplWidget


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
