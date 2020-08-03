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

from sys import argv
from time import sleep
from unittest import TestCase
from threading import Thread

from PyQt4 import QtGui
from PyQt4.QtGui import QApplication

from Feedbacks.ScalpGL import ScalpGL
from Feedbacks.ScalpGL.control import Control

class ScalpTest(TestCase):
    def test_scalp(self):
        scalp = ScalpGL()
        scalp.on_init()
        scalp.on_play()

    def no_test_gui(self):
        app = QApplication(argv)
        g = Control()
        g.init()
        g.launch_gui()
        app.exec_()
