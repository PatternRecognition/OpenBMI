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

from PyQt4.QtCore import QObject, SIGNAL, Qt
from PyQt4.QtGui import QWidget, QGridLayout, QHBoxLayout, QLabel, QComboBox, \
                        QCheckBox

class SignalControl(QWidget):
    def __init__(self, electrodes, config):
        QWidget.__init__(self, None)
        self.electrodes = electrodes
        self.config = config
        self.layout = QHBoxLayout(self)
        self.layout.addWidget(QLabel('Electrode:'))
        names = QComboBox()
        names.addItems(self.electrodes.names)
        self.layout.addWidget(names)
        QObject.connect(names, SIGNAL('currentIndexChanged(const QString&)'),
                        lambda name: self.confset('name', str(name)))
        fixed = QCheckBox('Fixed Limits', None)
        fixed.setCheckState(2)
        self.layout.addWidget(fixed)
        QObject.connect(fixed, SIGNAL('stateChanged(int)'),
                        lambda state: self.confset('fixed', bool(state)))

    def confset(self, key, value):
        self.config[key] = value
