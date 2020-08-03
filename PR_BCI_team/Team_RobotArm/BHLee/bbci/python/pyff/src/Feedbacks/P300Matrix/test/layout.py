__copyright__ = """ Copyright (c) 2010 Torsten Schmits

This program is free software; you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by the
Free Software Foundation; either version 3 of the License, or (at your
option) any later version.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, see <http://www.gnu.org/licenses/>.
"""

from unittest import TestCase

from VisionEgg.Core import Screen
from P300Matrix.factory.layout import LayoutFactory

class LayoutTest(TestCase):
    def test_layout(self):
        screen = Screen()
        fac = LayoutFactory(screen)
        el = fac.symbols(range(10))
        lay = fac.matrix(el, 3)
        print lay.group(6)
