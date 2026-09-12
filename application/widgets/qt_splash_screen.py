from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QSplashScreen

from application.settings import ApplicationSettings


class SplashScreen(QSplashScreen):
    def __init__(self, width=360) -> None:
        pm = QPixmap(ApplicationSettings.RESOURCES_PATH + "/logo.png").scaled(
            width,
            width,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        super().__init__(pm)
        self.show()
