import numpy as np
import src.core.config as config
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QRect
from PyQt5.QtGui import QPixmap, QImage, qGray
from .components.draw_scene import DrawScene


class MainWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        content_layout = QtWidgets.QVBoxLayout()
        self.setLayout(content_layout)

        draw_scene = DrawScene(self)
        draw_scene.communication.doneDrawing.connect(self.on_drawing_done)
        self.graphics_view = QtWidgets.QGraphicsView()
        self.graphics_view.setDragMode(QtWidgets.QGraphicsView.NoDrag)
        self.graphics_view.setScene(draw_scene)

        self.letter_label = QtWidgets.QLabel()
        self.set_letter('не определена')

        content_layout.addWidget(self.graphics_view)
        content_layout.addWidget(self.letter_label, 0, Qt.AlignHCenter)

    def set_letter(self, letter: str):
        self.letter_label.setText('Нарисованная буква: ' + letter)

    def pixmap_into_gray_matrix(self, pixmap: QPixmap) -> np.ndarray:
        matrix = np.empty(shape=(config.IMG_WIDTH, config.IMG_HEIGHT))
        image: QImage = pixmap.toImage()
        for i in range(pixmap.width()):
            for j in range(pixmap.height()):
                matrix[i][j] = 255 - qGray(image.pixel(i, j))
        return matrix

    def on_drawing_done(self):
        grab_rect: QRect = self.graphics_view.sceneRect().toRect()
        grab_rect.setX(1)
        grab_rect.setY(1)
        grab_rect.setWidth(grab_rect.width() - 1)
        grab_rect.setHeight(grab_rect.height() - 1)
        pixmap: QPixmap = self.graphics_view.grab(grab_rect)
        pixmap = pixmap.scaled(config.IMG_WIDTH, config.IMG_HEIGHT, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        print(self.pixmap_into_gray_matrix(pixmap))
