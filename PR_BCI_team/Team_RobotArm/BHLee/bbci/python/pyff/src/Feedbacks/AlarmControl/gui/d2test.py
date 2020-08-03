# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'data/gui/d2test.ui'
#
# Created: Wed Feb 17 15:18:03 2010
#      by: PyQt4 UI code generator 4.7
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

class Ui_d2test(object):
    def setupUi(self, d2test):
        d2test.setObjectName("d2test")
        d2test.resize(238, 286)
        d2test.setWindowTitle("Ersatzhandlung")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/main/icons/ida_logo.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        d2test.setWindowIcon(icon)
        self.verticalLayout = QtGui.QVBoxLayout(d2test)
        self.verticalLayout.setContentsMargins(-1, 0, -1, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.top_bar = QtGui.QLabel(d2test)
        font = QtGui.QFont()
        font.setPointSize(30)
        self.top_bar.setFont(font)
        self.top_bar.setAlignment(QtCore.Qt.AlignCenter)
        self.top_bar.setObjectName("top_bar")
        self.verticalLayout.addWidget(self.top_bar)
        self.letter = QtGui.QLabel(d2test)
        font = QtGui.QFont()
        font.setFamily("Helvetica")
        font.setPointSize(30)
        font.setWeight(75)
        font.setBold(True)
        self.letter.setFont(font)
        self.letter.setText("p")
        self.letter.setScaledContents(False)
        self.letter.setAlignment(QtCore.Qt.AlignCenter)
        self.letter.setTextInteractionFlags(QtCore.Qt.NoTextInteraction)
        self.letter.setObjectName("letter")
        self.verticalLayout.addWidget(self.letter)
        self.bottom_bar = QtGui.QLabel(d2test)
        font = QtGui.QFont()
        font.setPointSize(30)
        self.bottom_bar.setFont(font)
        self.bottom_bar.setAlignment(QtCore.Qt.AlignCenter)
        self.bottom_bar.setObjectName("bottom_bar")
        self.verticalLayout.addWidget(self.bottom_bar)
        self.progress = QtGui.QProgressBar(d2test)
        self.progress.setProperty("value", 0)
        self.progress.setAlignment(QtCore.Qt.AlignCenter)
        self.progress.setObjectName("progress")
        self.verticalLayout.addWidget(self.progress)

        self.retranslateUi(d2test)
        QtCore.QMetaObject.connectSlotsByName(d2test)

    def retranslateUi(self, d2test):
        self.top_bar.setText(QtGui.QApplication.translate("d2test", "|||", None, QtGui.QApplication.UnicodeUTF8))
        self.bottom_bar.setText(QtGui.QApplication.translate("d2test", "|||", None, QtGui.QApplication.UnicodeUTF8))

import alarm_rc
