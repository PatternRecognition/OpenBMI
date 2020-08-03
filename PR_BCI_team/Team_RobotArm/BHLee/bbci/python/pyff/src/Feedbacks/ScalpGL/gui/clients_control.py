__copyright__ = """ Copyright (c) 2008, 2009 Torsten Schmits

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

from copy import copy

from PyQt4.QtCore import QObject, SIGNAL, Qt
from PyQt4.QtGui import QWidget, QGridLayout, QHBoxLayout, QLabel, QSpinBox, \
                        QCheckBox, QPushButton, QGroupBox

from .control_widget import ControlWidget

class ClientsControl(ControlWidget):
    def __init__(self, config, parent=None):
        ControlWidget.__init__(self, config, parent)
        self._setup_layout()
        self.update_layout = SIGNAL('update_layout()')
        self.test = SIGNAL('test()')

    def _setup_layout(self):
        lay = QHBoxLayout(self)
        gb = QGroupBox('Settings')
        lay.addWidget(gb)
        layout = QGridLayout(gb)
        measures = QHBoxLayout()
        measures.addWidget(QLabel('Clients:'))
        self.rows = QSpinBox()
        self.cols = QSpinBox()
        self.rows.setRange(1, 4)
        self.cols.setRange(1, 4)
        measures.addWidget(self.rows)
        measures.addWidget(QLabel('x'))
        measures.addWidget(self.cols)
        ok = QPushButton('OK')
        QObject.connect(ok, SIGNAL('released()'), self._update_measures)
        measures.addWidget(ok)
        test = QPushButton('Test')
        QObject.connect(test, SIGNAL('released()'), self._test)
        measures.addWidget(test)
        layout.addLayout(measures, 0, 0)

    def _update_measures(self):
        self.config['rows'] = self.rows.value()
        self.config['cols'] = self.cols.value()
        self.emit(self.update_layout)

    def _test(self):
        self.emit(self.test)

