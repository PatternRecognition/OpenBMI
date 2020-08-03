__copyright__ = """ Copyright (c) 2012 Torsten Schmits

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

import unittest, threading, time

from MapNavi import MapNavi

class InputGenerator(threading.Thread):
    def __init__(self, map):
        self._map = map
        threading.Thread.__init__(self)

    def run(self):
        while not self._map.running:
            time.sleep(0.2)
        keys = ('cl_output', 'stepsize')
        input = ((30, 1), (0, 1), (-90, 2), (90, 1))
        for values in input:
            data = dict(zip(keys, values))
            self._map.on_control_event(data)

class MapNaviTest(unittest.TestCase):
    def test_map(self):
        mn = MapNavi()
        input = InputGenerator(mn)
        input.start()
        mn.on_init()
        mn.on_play()

if __name__ == '__main__':
    unittest.main()
