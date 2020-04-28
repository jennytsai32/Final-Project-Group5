
from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(735, 485)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(10, 0, 711, 431))
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.tabWidget.addTab(self.tab_2, "")
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")
        self.pushButton = QtWidgets.QPushButton(self.tab_3)
        self.pushButton.setGeometry(QtCore.QRect(280, 360, 141, 31))
        self.pushButton.setObjectName("pushButton")
        self.graphicsView = QtWidgets.QGraphicsView(self.tab_3)
        self.graphicsView.setGeometry(QtCore.QRect(40, 60, 641, 281))
        self.graphicsView.setObjectName("graphicsView")
        self.label_4 = QtWidgets.QLabel(self.tab_3)
        self.label_4.setGeometry(QtCore.QRect(290, 30, 131, 21))
        self.label_4.setObjectName("label_4")
        self.tabWidget.addTab(self.tab_3, "")
        self.tab_5 = QtWidgets.QWidget()
        self.tab_5.setObjectName("tab_5")
        self.comboBox_3 = QtWidgets.QComboBox(self.tab_5)
        self.comboBox_3.setGeometry(QtCore.QRect(210, 310, 131, 31))
        self.comboBox_3.setObjectName("comboBox_3")
        self.comboBox_3.addItem("")
        self.comboBox_3.addItem("")
        self.comboBox_3.addItem("")
        self.label_3 = QtWidgets.QLabel(self.tab_5)
        self.label_3.setGeometry(QtCore.QRect(220, 290, 111, 21))
        self.label_3.setObjectName("label_3")
        self.comboBox_2 = QtWidgets.QComboBox(self.tab_5)
        self.comboBox_2.setGeometry(QtCore.QRect(360, 310, 131, 31))
        self.comboBox_2.setObjectName("comboBox_2")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.label_5 = QtWidgets.QLabel(self.tab_5)
        self.label_5.setGeometry(QtCore.QRect(370, 290, 181, 21))
        self.label_5.setObjectName("label_5")
        self.pushButton_2 = QtWidgets.QPushButton(self.tab_5)
        self.pushButton_2.setGeometry(QtCore.QRect(280, 360, 141, 31))
        self.pushButton_2.setObjectName("pushButton_2")
        self.textEdit = QtWidgets.QTextEdit(self.tab_5)
        self.textEdit.setGeometry(QtCore.QRect(350, 50, 351, 201))
        self.textEdit.setObjectName("textEdit")
        self.graphicsView_2 = QtWidgets.QGraphicsView(self.tab_5)
        self.graphicsView_2.setGeometry(QtCore.QRect(30, 50, 261, 201))
        self.graphicsView_2.setObjectName("graphicsView_2")
        self.label_6 = QtWidgets.QLabel(self.tab_5)
        self.label_6.setGeometry(QtCore.QRect(30, 20, 181, 21))
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(self.tab_5)
        self.label_7.setGeometry(QtCore.QRect(350, 20, 181, 21))
        self.label_7.setObjectName("label_7")
        self.tabWidget.addTab(self.tab_5, "")
        self.tab_4 = QtWidgets.QWidget()
        self.tab_4.setObjectName("tab_4")
        self.label_8 = QtWidgets.QLabel(self.tab_4)
        self.label_8.setGeometry(QtCore.QRect(40, 40, 181, 21))
        self.label_8.setObjectName("label_8")
        self.graphicsView_3 = QtWidgets.QGraphicsView(self.tab_4)
        self.graphicsView_3.setGeometry(QtCore.QRect(40, 70, 261, 201))
        self.graphicsView_3.setObjectName("graphicsView_3")
        self.label_9 = QtWidgets.QLabel(self.tab_4)
        self.label_9.setGeometry(QtCore.QRect(370, 40, 181, 21))
        self.label_9.setObjectName("label_9")
        self.textEdit_2 = QtWidgets.QTextEdit(self.tab_4)
        self.textEdit_2.setGeometry(QtCore.QRect(370, 70, 351, 201))
        self.textEdit_2.setObjectName("textEdit_2")
        self.label_10 = QtWidgets.QLabel(self.tab_4)
        self.label_10.setGeometry(QtCore.QRect(280, 290, 111, 21))
        self.label_10.setObjectName("label_10")
        self.comboBox_4 = QtWidgets.QComboBox(self.tab_4)
        self.comboBox_4.setGeometry(QtCore.QRect(280, 310, 131, 31))
        self.comboBox_4.setObjectName("comboBox_4")
        self.comboBox_4.addItem("")
        self.comboBox_4.addItem("")
        self.pushButton_3 = QtWidgets.QPushButton(self.tab_4)
        self.pushButton_3.setGeometry(QtCore.QRect(280, 350, 141, 31))
        self.pushButton_3.setObjectName("pushButton_3")
        self.tabWidget.addTab(self.tab_4, "")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 735, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(4)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "Loading"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "EDA"))
        self.pushButton.setText(_translate("MainWindow", "Run Random Forest"))
        self.label_4.setText(_translate("MainWindow", "Feature Importance"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), _translate("MainWindow", "Feature Selection"))
        self.comboBox_3.setItemText(0, _translate("MainWindow", "None"))
        self.comboBox_3.setItemText(1, _translate("MainWindow", "Upsampled"))
        self.comboBox_3.setItemText(2, _translate("MainWindow", "Downsampled"))
        self.label_3.setText(_translate("MainWindow", "Sample Weight"))
        self.comboBox_2.setItemText(0, _translate("MainWindow", "All features"))
        self.comboBox_2.setItemText(1, _translate("MainWindow", "Top 10 features"))
        self.label_5.setText(_translate("MainWindow", "Feature Selection"))
        self.pushButton_2.setText(_translate("MainWindow", "Run Model"))
        self.label_6.setText(_translate("MainWindow", "Confustion Matrix"))
        self.label_7.setText(_translate("MainWindow", "Classification Report"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_5), _translate("MainWindow", "RF - Bagging"))
        self.label_8.setText(_translate("MainWindow", "Confustion Matrix"))
        self.label_9.setText(_translate("MainWindow", "Classification Report"))
        self.label_10.setText(_translate("MainWindow", "Feature Selection"))
        self.comboBox_4.setItemText(0, _translate("MainWindow", "All features"))
        self.comboBox_4.setItemText(1, _translate("MainWindow", "Top 10 features"))
        self.pushButton_3.setText(_translate("MainWindow", "Run Model"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_4), _translate("MainWindow", "Adaptive Boosting Trees"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
