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

from copy import copy
from os import makedirs, path

from AlarmControl.money import Bookie

class TrialControl(object):
    _log = []
    log_file = None
    _bookie = Bookie()

    def __init__(self, control, trial_gui):
        self._control = control
        self._trial_gui = trial_gui
        self.__init_attributes()

    def __init_attributes(self):
        self._trial_running = False
        self._finished = False

    def _trigger(self, trigger):
        self._control.trigger(trigger)

    def init(self):
        self._setup_trial()
        self._setup_gui()
        self._connect_signals()

    def start_trial(self):
        self._trial_running = True
        self._trial_gui.start_trial(self._gui, self._trial)
        self._bookie.start_trial()

    def stop_trial(self):
        self._trial_running = False
        self._log_trial()
        self._trial.recreate()
        self._trial_gui.stop_trial()
        self._stop_trial()
        self._control.stop_trial()

    def _stop_trial(self):
        pass

    def update_parameters(self):
        pass

    def reset(self):
        self._bookie.reset()

    def quit(self):
        self.stop_trial()
        self._finished = True

    def stop_game(self):
        self._finished = True
        self.final_money_dialog()

    def _log_trial(self):
        self._log.append(copy(self._trial))

    @classmethod
    def write_log(self):
        strings = [str(p) + '\n' for p in self._log]
        dir = path.dirname(self.log_file)
        if not path.exists(dir):
            makedirs(dir)
        with open(self.log_file, mode='w') as f:
            f.writelines(strings)

    def final_money_dialog(self):
        info = u'Sie haben einen Gesamtbetrag\nvon %.2f Euro verdient.'
        self._trial_gui.set_status(info % self._bookie.total_money)
