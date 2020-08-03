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

from PyQt4.QtGui import QWidget, QGridLayout, QVBoxLayout, QLabel

from .clients_control import ClientsControl

class ControlPanel(QWidget):
    def __init__(self, parent):
        QWidget.__init__(self, parent)
        self._setup_layout()

    def _setup_layout(self):
        self.common_control_layout = QVBoxLayout()
        self.common_control_layout.addWidget(QLabel('Control Panel'))
        self.layout = QVBoxLayout(self)
        self.layout.addLayout(self.common_control_layout)
        self.current_client = None

    def set_clients_control(self, widget):
        self.common_control_layout.addWidget(widget)

    def set_client_control(self, client):
        if self.current_client:
            self.layout.removeWidget(self.current_client)
        self.current_client = client
        self.layout.addWidget(self.current_client)
