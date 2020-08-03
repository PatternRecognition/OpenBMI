# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'data/gui/main.ui'
#
# Created: Wed Dec  8 15:48:02 2010
#      by: PyQt4 UI code generator 4.7.3
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

class Ui_Main(object):
    def setupUi(self, Main):
        Main.setObjectName("Main")
        Main.resize(800, 600)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/main/icons/ida_logo.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        Main.setWindowIcon(icon)
        self.centralwidget = QtGui.QWidget(Main)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtGui.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label = QtGui.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(16)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        self.alarm_control = AlarmControl(self.centralwidget)
        self.alarm_control.setFrameShape(QtGui.QFrame.StyledPanel)
        self.alarm_control.setFrameShadow(QtGui.QFrame.Raised)
        self.alarm_control.setObjectName("alarm_control")
        self.verticalLayout.addWidget(self.alarm_control)
        self.widget = QtGui.QWidget(self.centralwidget)
        self.widget.setObjectName("widget")
        self.verticalLayout.addWidget(self.widget)
        Main.setCentralWidget(self.centralwidget)

        self.retranslateUi(Main)
        QtCore.QMetaObject.connectSlotsByName(Main)

    def retranslateUi(self, Main):
        Main.setWindowTitle(QtGui.QApplication.translate("Main", "AlarmControl", None, QtGui.QApplication.UnicodeUTF8))
        self.label.setText(QtGui.QApplication.translate("Main", "TextLabel", None, QtGui.QApplication.UnicodeUTF8))

from alarm import AlarmControl
import alarm_rc
