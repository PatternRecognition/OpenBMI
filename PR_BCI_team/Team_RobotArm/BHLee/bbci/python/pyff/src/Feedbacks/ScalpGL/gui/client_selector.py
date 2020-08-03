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

from PyQt4.QtGui import QVBoxLayout, QFrame, QLabel, QPixmap, QDrag

from PyQt4.QtCore import QMimeData, QPoint

class ClientInfo(QLabel):
    icon_dir = 'data/gui/icons/'
    def __init__(self, client_type, tooltip):
        QLabel.__init__(self)
        self.type = client_type
        self.setPixmap(QPixmap(self.icon_dir + client_type).scaled(64, 64))
        self.setToolTip(tooltip)

class ClientSelector(QFrame):
    def __init__(self, parent):
        QFrame.__init__(self, parent)
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
        layout = QVBoxLayout(self)
        layout.addWidget(ClientInfo('3dscalp', '3D Scalp'))
        layout.addWidget(ClientInfo('2dscalp', '2D Scalp'))
        layout.addWidget(ClientInfo('signal', 'Signal'))
        layout.addStretch(1)

    def mousePressEvent(self, event):
        """ Create a Drag object that carries the identifier string
        from the clicked ClientInfo.

        """
        child = self.childAt(event.pos())
        mime_data = QMimeData()
        mime_data.setText(child.type)

        drag = QDrag(self)
        drag.setPixmap(child.pixmap())
        drag.setHotSpot(QPoint(drag.pixmap().width() / 2,
                               drag.pixmap().height() / 2))
        drag.setMimeData(mime_data)
        drag.exec_()
