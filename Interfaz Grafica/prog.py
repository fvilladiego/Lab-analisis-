import sys
from PyQt5 import uic
from PyQt5.QtWidgets import QMainWindow,QApplication

class programa(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("programa.ui",self)

if  __name__== "__main__":
    prog = QApplication(sys.argv)
    GUI=programa()
    GUI.show()
    sys.exit(prog.exec_())

