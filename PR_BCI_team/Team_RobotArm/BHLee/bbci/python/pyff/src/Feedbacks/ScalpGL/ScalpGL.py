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
from threading import Thread

from PyQt4.QtGui import QApplication

from FeedbackBase.MainloopFeedback import MainloopFeedback

from .control import Control

class MainLoop(Thread):
    def __init__(self, scalp):
        Thread.__init__(self)
        self.scalp = scalp

    def run(self):
        self.scalp._mainloop()

class ScalpGL(MainloopFeedback, Control):
    def __init__(self, foo=None, boo=None):
        Control.__init__(self)

    def init(self):
        Control.init(self)
        self.mainloop = MainLoop(self)

    def on_play(self):
        self.app = QApplication(argv)
        self.pre_mainloop()
        #self.mainloop.start()
        self.app.exec_()
        self.post_mainloop()

    def pre_mainloop(self):
        self.launch_gui()

    def post_mainloop(self):
        self.close_gui()

    def on_interaction_event(self, data):
        pass

    def tick(self):
        pass
