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
from time import sleep, time
import logging

from PyQt4.QtGui import QApplication
from PyQt4 import QtCore

from FeedbackBase.MainloopFeedback import MainloopFeedback

from control.main import Control

class MainLoop(Thread):
    def __init__(self, alarm):
        Thread.__init__(self)
        self.alarm = alarm

    def run(self):
        try:
            self.alarm._mainloop()
        except:
            pass

class AlarmControl(MainloopFeedback, Control):
    def __init__(self, port_num=None):
        self._debug = False
        MainloopFeedback.__init__(self, port_num)
        Control.__init__(self, self.send_parallel)
        wrapped_funcs = ['on_play', 'on_stop', 'stop_game', 'on_quit',
                         'on_interaction_event', 'check_time']
        for f in wrapped_funcs:
            self._set_wrapper(f)
        # does not really work when changed in the GUI, leaving it here to be
        # able to use the old experiment if needed.
        self.do_d2 = False
        # Can the control speller be stopped by a control signal?
        self.can_stop_on_control_signal = False

    def _set_wrapper(self, name):
        self.logger.debug("Wrapping method: %s" % name)
        wrapped_name = '_wrap_' + name
        wrapped = getattr(self, wrapped_name)
        wrapper = lambda *args: self._try_wrapper(name, wrapped, *args)
        setattr(self, name, wrapper)

    def init(self):
        self._app = QApplication(argv)
        self.tick_frequency = 2
        Control.init(self)
        self._main_loop = MainLoop(self)

    def _try_wrapper(self, name, func, *args):
        try:
            func(*args)
        except Exception, e:
            self.logger.debug('%s: %s' % (name, str(e)))
            if self._debug:
                raise

    def _wrap_on_play(self):
        self.pre_mainloop()
        self._main_loop.start()
        self._app.exec_()
        self.post_mainloop()

    def _wrap_on_stop(self):
        MainloopFeedback.on_stop(self)
        self.stop_game()
        self.reset()

    def _wrap_stop_game(self):
        MainloopFeedback.on_pause(self)
        Control.stop_game(self)

    def _wrap_on_quit(self):
        MainloopFeedback.on_quit(self)
        self.stop_game()
        self.reset()

    # TODO: document me!
    control_signal = QtCore.pyqtSignal()

    def pre_mainloop(self):
        self._start_time = time()
        self.launch_gui()
        if not self.do_d2:
            self.control_signal.connect(self._trial._gui.mystop)


    def post_mainloop(self):
        self.close_gui()
        self.control_signal.disconnect()

    def _wrap_on_interaction_event(self, data):
        self.update_parameters()

    def play_tick(self):
        self.check_time()
        sleep(1)

    def _wrap_check_time(self):
        if self.stop_time and self.time_running > self.stop_time:
            self.on_stop()

    @property
    def time_running(self):
        return time() - self._start_time

    # special stuff for the copyspeller task:
    # we will receive a control event when a message was not followed by an erp
    # and thus not recognized by the participant. in this case the copyspeller
    # task should be stopped
    def on_control_event(self, data):
        if not self.do_d2 and self.can_stop_on_control_signal:
            v = data['cl_output']
            if v > 0:
                # we need to use the qt event queue bc we're in a different
                # thread and calling a method of the copy speller widget would
                # cause a crash
                self.control_signal.emit()
            else:
                # BCI system reports that ERP was recognized
                pass

