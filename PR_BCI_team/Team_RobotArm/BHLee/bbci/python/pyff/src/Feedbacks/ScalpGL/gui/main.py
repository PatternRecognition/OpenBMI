# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'data/gui/scalp.ui'
#
# Created: Mon May 25 18:57:32 2009
#      by: PyQt4 UI code generator 4.5-snapshot-20090518
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

class Ui_scalp_main_window(object):
    def setupUi(self, scalp_main_window):
        scalp_main_window.setObjectName("scalp_main_window")
        scalp_main_window.resize(680, 668)
        self.scalp_widget = QtGui.QWidget(scalp_main_window)
        self.scalp_widget.setObjectName("scalp_widget")
        self.verticalLayout = QtGui.QVBoxLayout(self.scalp_widget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.verticalLayout_2 = QtGui.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.control_panel = ControlPanel(self.scalp_widget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.control_panel.sizePolicy().hasHeightForWidth())
        self.control_panel.setSizePolicy(sizePolicy)
        self.control_panel.setObjectName("control_panel")
        self.verticalLayout_2.addWidget(self.control_panel)
        self.main_view = QtGui.QWidget(self.scalp_widget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.main_view.sizePolicy().hasHeightForWidth())
        self.main_view.setSizePolicy(sizePolicy)
        self.main_view.setObjectName("main_view")
        self.horizontalLayout = QtGui.QHBoxLayout(self.main_view)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.clients = Clients(self.main_view)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(2)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.clients.sizePolicy().hasHeightForWidth())
        self.clients.setSizePolicy(sizePolicy)
        self.clients.setObjectName("clients")
        self.horizontalLayout.addWidget(self.clients)
        self.client_selector = ClientSelector(self.main_view)
        self.client_selector.setObjectName("client_selector")
        self.horizontalLayout.addWidget(self.client_selector)
        self.verticalLayout_2.addWidget(self.main_view)
        self.verticalLayout.addLayout(self.verticalLayout_2)
        scalp_main_window.setCentralWidget(self.scalp_widget)
        self.menubar = QtGui.QMenuBar(scalp_main_window)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 680, 25))
        self.menubar.setObjectName("menubar")
        self.menuScalp = QtGui.QMenu(self.menubar)
        self.menuScalp.setObjectName("menuScalp")
        scalp_main_window.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(scalp_main_window)
        self.statusbar.setObjectName("statusbar")
        scalp_main_window.setStatusBar(self.statusbar)
        self.actionQuit = QtGui.QAction(scalp_main_window)
        self.actionQuit.setObjectName("actionQuit")
        self.menuScalp.addAction(self.actionQuit)
        self.menubar.addAction(self.menuScalp.menuAction())

        self.retranslateUi(scalp_main_window)
        QtCore.QObject.connect(self.actionQuit, QtCore.SIGNAL("activated()"), scalp_main_window.close)
        QtCore.QMetaObject.connectSlotsByName(scalp_main_window)

    def retranslateUi(self, scalp_main_window):
        scalp_main_window.setWindowTitle(QtGui.QApplication.translate("scalp_main_window", "ScalpGL", None, QtGui.QApplication.UnicodeUTF8))
        self.menuScalp.setTitle(QtGui.QApplication.translate("scalp_main_window", "Scalp", None, QtGui.QApplication.UnicodeUTF8))
        self.actionQuit.setText(QtGui.QApplication.translate("scalp_main_window", "Quit", None, QtGui.QApplication.UnicodeUTF8))

from client_selector import ClientSelector
from control_panel import ControlPanel
from clients import Clients
