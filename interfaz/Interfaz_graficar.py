import sys
from PyQt5 import uic, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QApplication, QTableWidget, QTableWidgetItem
from PyQt5 import QtGui
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
import pandas as pd

class programa(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("programa.ui", self)
        #botones

        self.graficar.clicked.connect(self.grafica)
        self.aceptar.clicked.connect(self.cantidad)


    def cantidad(self):
        canti = self.can.toPlainText()
        canti=int(canti)
        self.tableWidget.setRowCount(canti)
        self.tableWidget.setColumnCount(2)
        self.tableWidget.setHorizontalHeaderLabels(('Datos x', 'Datos y'))
        return canti
    def datos(self):
        print(self.cantidad())
        lista=[]
        pro=[]

        for x in range(self.cantidad()):

            print(self.tableWidget.item(x, 0).text())
            elementox=self.tableWidget.item(x, 0).text()
            elementox=int(elementox)
            pro.append(elementox)


            print(self.tableWidget.item(x, 1).text())
            elementoy=self.tableWidget.item(x, 1).text()
            elementoy=int(elementoy)
            pro.append(elementoy)


            lista.append(pro)
            pro=[]

        return lista

    def grafica(self):

        data = np.array(self.datos())

        variable1 = np.array(data[:, 0])
        variable2 = np.array(data[:, 1])

        graph_title = "grafica 1"

        graph_x = "x"

        graph_y = "y"

        plt.plot(variable1, variable2, 'b.')
        plt.suptitle(graph_title)
        plt.xlabel(variable1)
        plt.ylabel(variable2)
        plt.grid(b=True)
        plt.legend()
        pyplot.axhline(0, color="black")
        pyplot.axvline(0, color="black")
        plt.show()
        plt.close()

if __name__ == "__main__":

    prog = QApplication(sys.argv)
    GUI = programa()
    GUI.show()
    sys.exit(prog.exec_())

