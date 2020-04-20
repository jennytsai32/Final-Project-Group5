import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

from PyQt5 import uic, QtWidgets
from PyQt5.QtWidgets import*

from matplotlib.backends.backend_qt5agg import *

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from matplotlib.figure import Figure


qtCreateFile = './gui/Main/mainwindow.ui'
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreateFile)

class MyApp(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)


        self.pushButton_loading.clicked.connect(self.getCSV)
        #self.pushButton_plot.clicked.connect(self.plot)
        self.pushButton_EDA.clicked.connect(self.temp_humidity)
        self.pushButton_EDA2.clicked.connect(self.temp_windy)

    def getCSV(self):
        filePath, _ = QtWidgets.QFileDialog.getOpenFileName(self,'Open file', '/home')
        if filePath !='':
            print('The file path is:',filePath)
            self.df=pd.read_csv('./data/road_condition_2019_draft.csv')
            self.progressBar.setValue(100)
            result=self.df.describe()
            #print(result)
            #print(stat_st)
            stat_st = 'Data Description:'+'\n'+'\n' + str(self.df.describe())
            self.textEdit_output.setText(stat_st)

    def Features(self):
         button = QPushButton("Click Me", self)
         button.move(100,450)

         button2 = QPushButton("Click Me Two", self)
         button2.move(100, 450)

    def temp_humidity(self):
        x = self.df['temp']
        y = self.df['humidity']
        z = self.df['windy']

        self.MplWidget.canvas.axes.clear()
        self.MplWidget.canvas.axes.plot(x,y)
        self.MplWidget.canvas.axes.legend(('temp', 'humidity'), loc='upper right')
        self.MplWidget.canvas.axes.set_title('temp - humidity')
        self.MplWidget.canvas.draw()

    def temp_windy(self):
        x = self.df['temp']
        y = self.df['windy']

        self.MplWidget.canvas.axes.clear()
        self.MplWidget.canvas.axes.plot(x, y)
        self.MplWidget.canvas.axes.legend(('temp', 'windy'), loc='upper right')
        self.MplWidget.canvas.axes.set_title('temp - windy')
        self.MplWidget.canvas.draw()

    def MyPlot(self):
         button = QPushButton("Click Me", self)
         button.move(100,450)

         button2 = QPushButton("Click Me Two", self)
         button2.move(100, 450)



if __name__=="__main__":
    app= QtWidgets.QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())