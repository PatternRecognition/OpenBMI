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

from PyQt4.QtCore import QObject, SIGNAL
from matplotlib.cm import jet
from numpy import random, asarray, zeros

from .mesh_factory import MeshFactory
from .gui_factory import GuiFactory
from .gl import HeadScene, DisplayList
from .model.electrode_mesh import ElectrodeMesh61, ElectrodeMeshESGrid

class Control(QObject):
    def init(self):
        self.data = {}
        self.mesh_factory = MeshFactory(self.data)
        self.gui_factory = GuiFactory(self.data)
        self._setup_head()

    def _setup_head(self):
        self._create_meshes()
        self._create_lists()
        self._create_electrodes()

    def _create_meshes(self):
        self.data['cap'] = MeshFactory.hemisphere(complexity=100)
        self.data['cap_61'] = MeshFactory.cap_61()
        self.data['head'] = MeshFactory.mesh_from_file('head2_norm')

    def _create_lists(self):
        self.data['head_list'] = DisplayList(self.data['head'])
        self.data['cap_list'] = DisplayList(self.data['cap'])
        self.data['cap_61_list'] = DisplayList(self.data['cap_61'])

    def _create_electrodes(self):
        self.data['electrodes_eq'] = ElectrodeMeshESGrid()
        self.data['electrodes_61'] = ElectrodeMesh61()

    def launch_gui(self):
        self._create_gui()
        self._create_signals()
        self._init_electrode_test_voltage()
        #self._create_default_clients()
        self.gui.show()
        #self.startTimer(2000)

    def _create_signals(self):
        self.update_data = SIGNAL('update_data()')
        self.update_layout = SIGNAL('update_layout()')

    def _create_gui(self):
        self.gui = self.gui_factory.main(self.clients_drop_event,
                                         self.clients_mouse_press_event)

    def _init_electrode_test_voltage(self):
        self.data['electrodes_eq'].set_random_voltage()
        self.data['electrodes_61'].set_random_voltage()
        self._random_voltage_diff()

    def _create_default_clients(self):
        self._insert_client('3dscalp')
        #self._insert_client('signal', 1)
        self._insert_client('2dscalp', 1)

    def close_gui(self):
        #self.timer.stop()
        self.gui.close()

    def clients_drop_event(self, event):
        event.accept()
        client_type = str(event.mimeData().text()) # is QString
        old_client, index = self.gui.clients.client_at(event.pos())
        self._insert_client(client_type, index)

    def clients_mouse_press_event(self, event):
        client = self.gui.clients.childAt(event.pos())
        if hasattr(client, 'config'):
            control = self.gui_factory.control(client)
            self.gui.control_panel.set_client_control(control)

    def _insert_client(self, client_type, index=0):
        client = self.gui_factory.client(client_type)
        QObject.connect(self, self.update_data, client.update_view)
        self.gui.clients.insert_client_qt(client, index)
        self.emit(self.update_data)

    def _random_voltage_diff(self):
        #elec = self.data['electrodes_eq']
        #cap = self.data['cap'] 
        #elec.add_noise()
        #voltage = elec.interpolate(asarray(cap.vertices).shape)
        #colors = jet(voltage)
        #cap.set_colors(colors)
        elec = self.data['electrodes_61']
        cap = self.data['cap_61'] 
        elec.add_noise()
        elec.step()
        colors = jet(elec.electrodes.copy())
        cap.set_colors(colors)

    def update_voltage(self):
        if self.gui_factory.gl_active:
            self.data['cap_61_list'].generate()
        self.emit(self.update_data)

    def timerEvent(self, e):
        self._random_voltage_diff()
        self.update_voltage()
