# -*- coding: utf-8 -*-

""" Copyright (c) 2008, 2009 Torsten Schmits

This file is part of the pyff framework. pyff is free software;
you can redistribute it and/or modify it under the terms of the GNU General
Public License version 2, as published by the Free Software Foundation.

pyff is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
details.

You should have received a copy of the GNU General Public License along with
this program; if not, write to the Free Software Foundation, Inc., 59 Temple
Place, Suite 330, Boston, MA  02111-1307  USA

"""

from PyQt4.QtGui import QFrame, QIcon, QLabel, QToolBar, QStatusBar, \
                        QVBoxLayout, QTableView, QAction, QAbstractItemView
from PyQt4.QtCore import QDate, QTime, SIGNAL, QModelIndex

from AlarmControl.util.metadata import datadir
from AlarmControl.util.clock import Clock

class AlarmControl(QFrame):
    icon_path = ':/main/icons/'

    def __init__(self, parent):
        QFrame.__init__(self, parent)
        self.__setup_actions()
        self.__setup_toolbar()
        self.__setup_table()
        self.__setup_statusbar()
        self.__setup_time()
        self.__setup_layout()
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)

    def __setup_actions(self):
        self.actions = {}
        a = lambda icon, text: QAction(QIcon(self.icon_path + icon), text, self)
        self.actions['show_active_messages'] = a('active_messages', 
                                         'Aktive Meldungen anzeigen')
        self.actions['show_shortterm_archive'] = a('shortterm_archive', 
                                         'Kurzzeitarchiv anzeigen')
        self.actions['show_longterm_archive'] = a('longterm_archive', 
                                         'Langzeitarchiv anzeigen')
        self.actions['ack_dysfunction'] = a('checkmark', 'Erste Warnung quittieren')
        self.actions['ack_selected'] = a('checkmark', 
                                         u'Ausgewählte Nachrichten' + \
                                         ' quittieren'.encode('utf-8'))
        self.actions['ack_all'] = a('checkmark', 'Alle Nachrichten quittieren')
        self.actions['ack_selected'].setEnabled(False)
        self.actions['ack_all'].setEnabled(False)

    def __setup_toolbar(self):
        self.toolbar = QToolBar()
        self.toolbar.addAction(self.actions['show_active_messages'])
        self.toolbar.addAction(self.actions['show_shortterm_archive'])
        self.toolbar.addAction(self.actions['show_longterm_archive'])
        self.toolbar.addAction(self.actions['ack_dysfunction'])
        self.toolbar.addAction(self.actions['ack_selected'])
        self.toolbar.addAction(self.actions['ack_all'])

    def __setup_table(self):
        self.messages = QTableView()
        self.messages.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.messages.horizontalHeader().setStretchLastSection(True)

    def __setup_time(self):
        self.clock = Clock()
        self.update_time()

    def __setup_statusbar(self):
        def label(text=''):
            l = QLabel(text, None)
            l.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
            return l
        self.statusbar = QStatusBar()
        self.date = label()
        self.time = label()
        self.message_count = label('Liste: 0')
        self.ack_message_count = label('Quit: 0')
        self.statusbar.addPermanentWidget(self.ack_message_count)
        self.statusbar.addPermanentWidget(self.message_count)
        self.statusbar.addPermanentWidget(self.time)
        self.statusbar.addPermanentWidget(self.date)
        self.statusbar.setSizeGripEnabled(False)

    def __setup_layout(self):
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.toolbar)
        main_layout.addWidget(self.messages)
        main_layout.addWidget(self.statusbar)

    def set_model(self, model):
        self.messages.setModel(model)
        for s in ['Inserted', 'Removed']:
            self.connect(model, 
                         SIGNAL('rows' + s + '(const QModelIndex &, int, int)'),
                         self.model_changed)
        self.model_changed()

    def model_changed(self):
        row_count = self.messages.model().rowCount()
        col_count = self.messages.model().columnCount()
        map(self.messages.resizeColumnToContents, xrange(col_count - 1))
        self.message_count.setText('Liste: ' + str(row_count))

    def update_time(self):
        self.date.setText(self.clock.date)
        self.time.setText(self.clock.time)

    def update_ack_count(self, count):
        self.ack_message_count.setText('Quit: ' + str(count))
