# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'data/gui/copyspeller.ui'
#
# Created: Wed Dec  8 15:48:03 2010
#      by: PyQt4 UI code generator 4.7.3
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

class Ui_CopySpeller(object):
    def setupUi(self, CopySpeller):
        CopySpeller.setObjectName("CopySpeller")
        CopySpeller.resize(718, 376)
        self.verticalLayout = QtGui.QVBoxLayout(CopySpeller)
        self.verticalLayout.setObjectName("verticalLayout")
        self.textBrowser = QtGui.QTextBrowser(CopySpeller)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.textBrowser.setFont(font)
        self.textBrowser.setObjectName("textBrowser")
        self.verticalLayout.addWidget(self.textBrowser)
        self.textEdit = QtGui.QTextEdit(CopySpeller)
        font = QtGui.QFont()
        font.setPointSize(14)
        self.textEdit.setFont(font)
        self.textEdit.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.textEdit.setObjectName("textEdit")
        self.verticalLayout.addWidget(self.textEdit)
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.pushButton_start = QtGui.QPushButton(CopySpeller)
        self.pushButton_start.setObjectName("pushButton_start")
        self.horizontalLayout.addWidget(self.pushButton_start)
        spacerItem = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.verticalLayout.addLayout(self.horizontalLayout)

        self.retranslateUi(CopySpeller)
        QtCore.QMetaObject.connectSlotsByName(CopySpeller)

    def retranslateUi(self, CopySpeller):
        CopySpeller.setWindowTitle(QtGui.QApplication.translate("CopySpeller", "Form", None, QtGui.QApplication.UnicodeUTF8))
        self.pushButton_start.setText(QtGui.QApplication.translate("CopySpeller", "Start", None, QtGui.QApplication.UnicodeUTF8))

