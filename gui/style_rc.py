# Resource object code (Python 3)
# Created by: object code
# Created by: The Resource Compiler for Qt version 6.0.1
# WARNING! All changes made in this file will be lost!

from PySide6 import QtCore

qt_resource_data = b"\
\x00\x00\x00i\
[\
Controls]\x0d\x0aStyle\
=Material\x0d\x0a\x0d\x0a[Un\
iversal]\x0d\x0aTheme=\
System\x0d\x0aAccent=R\
ed\x0d\x0a\x0d\x0a[Material]\
\x0d\x0aTheme=Dark\x0d\x0aAc\
cent=Red\
"

qt_resource_name = b"\
\x00\x08\
\x05\x94\xab\xc6\
\x00e\
\x00n\x00v\x00.\x00c\x00o\x00n\x00f\
"

qt_resource_struct = b"\
\x00\x00\x00\x00\x00\x02\x00\x00\x00\x01\x00\x00\x00\x01\
\x00\x00\x00\x00\x00\x00\x00\x00\
\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\
\x00\x00\x01x\x01Ht\x11\
"

def qInitResources():
    QtCore.qRegisterResourceData(0x03, qt_resource_struct, qt_resource_name, qt_resource_data)

def qCleanupResources():
    QtCore.qUnregisterResourceData(0x03, qt_resource_struct, qt_resource_name, qt_resource_data)

qInitResources()
