__copyright__ = """ Copyright (c) 2010-2011 Torsten Schmits

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

from P300Matrix.control import P300Matrix

class GUITest(TestCase):
    def test_gui(self):
        c = P300Matrix()
        c.on_interaction_event(None)
        c.on_init()
        c.on_play()
