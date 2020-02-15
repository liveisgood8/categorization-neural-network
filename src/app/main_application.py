import sys
from PyQt5 import QtWidgets
from ..gui.main_window import MainWindow


def start():
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
