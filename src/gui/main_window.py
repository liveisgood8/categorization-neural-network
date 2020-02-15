from PyQt5 import QtWidgets
from .main_widget import MainWidget


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)

        widget = MainWidget(self)
        self.setCentralWidget(widget)
