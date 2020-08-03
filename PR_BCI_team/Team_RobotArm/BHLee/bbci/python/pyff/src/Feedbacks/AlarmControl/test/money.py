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

from time import sleep

from unittest import TestCase

from AlarmControl.money import MoneyControl

class MoneyTest(TestCase):
    def test_money(self):
        m = MoneyControl()
        m.init()
        m.start_trial()
        sleep(4)
        m.solve_correct()
        self.assertEqual(int(m.total_money), 3)
