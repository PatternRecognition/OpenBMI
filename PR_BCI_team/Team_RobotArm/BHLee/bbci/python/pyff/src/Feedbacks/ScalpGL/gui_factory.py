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

from PyQt4.QtCore import QObject

from .gui import GLWidget, DummyClient, ScalpPlot2D, SignalPlot, GUI, ClientsControl
from .gui.client_control import SignalControl
from .gl import HeadScene, HeadScene61

class GuiFactory(object):
    def __init__(self, data):
        self.data = data
        self.data['gl_share'] = None
        self.gl_active = False

    def main(self, drop_handler, mouse_press_handler):
        gui = GUI(self.data)
        gui.clients.dropEvent = drop_handler
        gui.clients.mousePressEvent = mouse_press_handler
        cc = ClientsControl(gui.clients.config)
        QObject.connect(cc, cc.update_layout, gui.clients.update_layout)
        QObject.connect(cc, cc.test, gui.clients.test)
        gui.control_panel.set_clients_control(cc)
        return gui

    def client(self, view_type):
        method = getattr(self, 'client_' + view_type, self.client_dummy)
        return method()

    def client_dummy(self):
        return DummyClient()

    def client_3dscalp(self):
        self.gl_active = True
        #scene = HeadScene(self.data['head_list'], self.data['cap_list'])
        scene = HeadScene61(self.data['head_list'], self.data['cap_61_list'])
        widget = GLWidget(scene, shared=self.data['gl_share'])
        if self.data['gl_share'] is None:
            self.data['gl_share'] = widget
        return widget

    def client_2dscalp(self):
        return ScalpPlot2D(self.data['electrodes_61'])

    def client_signal(self):
        return SignalPlot(self.data['electrodes_61'])

    def control(self, client):
        method = getattr(self, 'control_' + client.type, self.client_dummy)
        return method(client.config)

    def control_signal(self, config):
        return SignalControl(self.data['electrodes_61'], config)

