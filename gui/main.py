import sys
from os.path import abspath, dirname, join

from PySide6.QtCore import QObject, Slot
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine
from style_rc import *
from PySide6.QtQuickControls2 import QQuickStyle

# https://doc.qt.io/qt-5/qtquicklayouts-index.html  layout
# https://doc.qt.io/qtforpython/examples/example_declarative_extending_chapter5-listproperties.html 自定义wigets
# https://doc.qt.io/qt-5/qml-qtquick-column.html 可考虑用于菜单
if __name__ == '__main__':
    app = QGuiApplication(sys.argv)
    QQuickStyle.setStyle("Material")
    engine = QQmlApplicationEngine()


    qmlFile = join(dirname(__file__), 'view.qml')
    engine.load(abspath(qmlFile))

    if not engine.rootObjects():
        sys.exit(-1)

    sys.exit(app.exec_())
