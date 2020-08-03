# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'data/gui/trial.ui'
#
# Created: Wed Dec  8 15:48:03 2010
#      by: PyQt4 UI code generator 4.7.3
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

class Ui_trial(object):
    def setupUi(self, trial):
        trial.setObjectName("trial")
        trial.resize(238, 286)
        trial.setWindowTitle("Ersatzhandlung")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/main/icons/ida_logo.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        trial.setWindowIcon(icon)
        self.verticalLayout = QtGui.QVBoxLayout(trial)
        self.verticalLayout.setContentsMargins(-1, 0, -1, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.status = QtGui.QLabel(trial)
        font = QtGui.QFont()
        font.setPointSize(16)
        self.status.setFont(font)
        self.status.setText("None")
        self.status.setScaledContents(False)
        self.status.setAlignment(QtCore.Qt.AlignCenter)
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(QtCore.Qt.NoTextInteraction)
        self.status.setObjectName("status")
        self.verticalLayout.addWidget(self.status)

        self.retranslateUi(trial)
        QtCore.QMetaObject.connectSlotsByName(trial)

    def retranslateUi(self, trial):
        pass

import alarm_rc
