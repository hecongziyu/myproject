import sys
from os.path import abspath, dirname, join

from PySide6.QtCore import QObject, Slot
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine

from style_rc import *


# https://stackoverflow.com/questions/37429626/pyside-and-qt-properties-connecting-signals-from-python-to-qml 
# https://stackoverflow.com/questions/57619227/connect-qml-signal-to-pyside2-slot
# https://wiki.qt.io/index.php?title=Qt_for_Python_-_Connecting_QML_Signals&redirect=no#Connecting_signals_from_Python_to_QML
# https://doc.qt.io/qtforpython/tutorials/qmlsqlintegration/qmlsqlintegration.html
class Bridge(QObject):

    @Slot(str, result=str)
    def getColor(self, color_name):
        if color_name.lower() == "red":
            return "#ef9a9a"
        elif color_name.lower() == "green":
            return "#a5d6a7"
        elif color_name.lower() == "blue":
            return "#90caf9"
        else:
            return "white"

    @Slot(float, result=int)
    def getSize(self, s):
        size = int(s * 42) # Maximum font size
        if size <= 0:
            return 1
        else:
            return size

    @Slot(str, result=bool)
    def getItalic(self, s):
        if s.lower() == "italic":
            return True
        else:
            return False

    @Slot(str, result=bool)
    def getBold(self, s):
        if s.lower() == "bold":
            return True
        else:
            return False

    @Slot(str, result=bool)
    def getUnderline(self, s):
        if s.lower() == "underline":
            return True
        else:
            return False


#  pyside6-rcc style.qrc > style_rc.py
# python pyside_2.py --style material
if __name__ == '__main__':
    app = QGuiApplication(sys.argv)
    engine = QQmlApplicationEngine()

    # Instance of the Python object
    bridge = Bridge()

    # Expose the Python object to QML
    context = engine.rootContext()
    context.setContextProperty("con", bridge)

    # Get the path of the current directory, and then add the name
    # of the QML file, to load it.
    qmlFile = join(dirname(__file__), 'pyside_2.qml')
    engine.load(abspath(qmlFile))

    if not engine.rootObjects():
        sys.exit(-1)

    sys.exit(app.exec_())
