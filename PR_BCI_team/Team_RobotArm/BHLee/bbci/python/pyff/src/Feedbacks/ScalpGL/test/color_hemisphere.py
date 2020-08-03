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

from unittest import TestCase

from numpy import asarray
from matplotlib.cm import Accent

from ..mesh_factory import MeshFactory
from ..gl.display_list import DisplayList
from ..model.electrode_mesh import ElectrodeMesh

class ColorHemisphereTest(TestCase):
    def test_it(self):
        hemi = MeshFactory.hemisphere(8)
        elec = ElectrodeMesh((4,4))
        elec.set_random_voltage()
        voltage = elec.interpolate(asarray(hemi.vertices).shape)
        colors = Accent(voltage)
        hemi.set_colors(colors)
