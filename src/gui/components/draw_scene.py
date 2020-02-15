import src.core.config as config
from PyQt5 import QtWidgets
from PyQt5.QtCore import QObject, QPoint, Qt, pyqtSignal
from PyQt5.QtGui import QPen, QBrush


class Communication(QObject):
    doneDrawing = pyqtSignal()


class DrawScene(QtWidgets.QGraphicsScene):
    previousPoint: QPoint = None
    LINE_RADIUS_PX = 6

    def __init__(self, parent=None):
        super().__init__(parent)
        self.communication = Communication()
        self.setSceneRect(0, 0, config.IMG_WIDTH * 10, config.IMG_HEIGHT * 10)

    def mousePressEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent):
        if event.button() == Qt.MidButton:
            self.clear()
        else:
            self.addEllipse(event.scenePos().x() - self.LINE_RADIUS_PX, event.scenePos().y() - self.LINE_RADIUS_PX,
                            self.LINE_RADIUS_PX * 2, self.LINE_RADIUS_PX * 2,
                            QPen(Qt.NoPen),
                            QBrush(Qt.black))
            self.previousPoint = event.scenePos()
            event.ignore()

    def mouseMoveEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent):
        self.addEllipse(event.scenePos().x() - self.LINE_RADIUS_PX, event.scenePos().y() - self.LINE_RADIUS_PX,
                        self.LINE_RADIUS_PX * 2, self.LINE_RADIUS_PX * 2,
                        QPen(Qt.black, self.LINE_RADIUS_PX * 2, Qt.SolidLine))
        self.previousPoint = event.scenePos()
        event.ignore()

    def mouseReleaseEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent):
        if event.button() != Qt.MidButton:
            self.communication.doneDrawing.emit()
