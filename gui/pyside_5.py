import os
import sys
import platform

from PySide6.QtGui import QGuiApplication, QIcon
from PySide6.QtCore import QSettings, QUrl
from PySide6.QtQml import QQmlApplicationEngine, QQmlContext
from PySide6.QtQuickControls2 import QQuickStyle

import rc_gallery

if __name__ == "__main__":
    QGuiApplication.setApplicationName("Gallery")
    QGuiApplication.setOrganizationName("QtProject")

    app = QGuiApplication()
    QIcon.setThemeName("gallery")

    settings = QSettings()
    if not os.environ.get("QT_QUICK_CONTROLS_STYLE"):
        QQuickStyle.setStyle(settings.value("style", str))

    engine = QQmlApplicationEngine()

    built_in_styles = ["Basic", "Fusion", "Imagine", "Material", "Universal"]
    if platform.system() == "Darwin":
        built_in_styles.append("macOS")
    elif platform.system() == "Windows":
        built_in_styles.append("Windows")
    engine.rootContext().setContextProperty("builtInStyles", built_in_styles)

    engine.load(":/gallery.qml")
    rootObjects = engine.rootObjects()
    if not rootObjects:
        sys.exit(-1)

    window = rootObjects[0]
    window.setIcon(QIcon(':/qt-project.org/logos/pysidelogo.png'))

    sys.exit(app.exec_())