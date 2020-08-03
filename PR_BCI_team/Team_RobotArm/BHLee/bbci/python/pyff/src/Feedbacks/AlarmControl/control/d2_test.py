# -*- coding: utf-8 -*-

""" Copyright (c) 2008, 2009 Torsten Schmits

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

from __future__ import with_statement

from time import time
from os import makedirs, path
from time import sleep
from copy import copy
from random import Random
from threading import Timer
import logging

from AlarmControl.model.d2_test import D2Test
from AlarmControl.util.triggers import *
from AlarmControl.gui.gui import D2TestWidget
from AlarmControl.control.trial import TrialControl
from AlarmControl.util import constants

class D2TestControl(TrialControl):
    def __init__(self, *args, **kargs):
        TrialControl.__init__(self, *args, **kargs)
        self.logger = logging.getLogger("D2TestControl")
        self._timer = Timer(1000, lambda: True)
        self._d2_gui_initialized = False

    def _setup_trial(self):
        self._trial = D2Test(self.target_quota, self.count, self.letters)

    def _setup_gui(self):
        self._gui = D2TestWidget(self, self._trial_gui, self.key_yes, self.key_no)
        self._d2_gui_initialized = True

    def _connect_signals(self):
        pass

    def update_parameters(self):
        TrialControl.update_parameters(self)
        self._setup_trial()
        if self._d2_gui_initialized:
            self._gui.set_keys(self.key_yes, self.key_no)

    def reset(self):
        self._trial.recreate()

    def start_trial(self):
        self._running = True
        self._timer = Timer(self.duration, self.stop_trial)
        self._timer.start()
        TrialControl.start_trial(self)

    def stop_trial(self):
        if self._running:
            self._timer.cancel()
            self._running = False
            self._trial.stop()
            TrialControl.stop_trial(self)

    def stop_game(self):
        self._timer.cancel()
        TrialControl.stop_game(self)

    def quit(self):
        self._timer.cancel()
        TrialControl.quit(self)

    def solve(self, solution):
        # trigger first, as solve() skips to the next atom
        self._trigger(self._trial.trigger_code)
        result = self._trial.solve(solution)
        if self._trial.finished:
            self.stop_trial()
        self._bookie.solve_d2(result)
        if result:
            self._trigger(TRIG_RICHTIG)
        else:
            self._trigger(TRIG_FALSCH)
        if self._running:
            self._gui.next_atom()

    @property
    def progress_string(self):
        return constants.d2_progress_string % self._bookie.total_money
