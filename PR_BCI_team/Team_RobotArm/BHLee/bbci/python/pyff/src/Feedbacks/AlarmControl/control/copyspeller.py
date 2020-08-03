# Copyright 2010  Bastian Venthur <bastia.venthur@tu-berlin.de>

import logging

from AlarmControl.control.trial import TrialControl
from AlarmControl.gui.gui import CopySpellerWidget

class CopySpellerControl(TrialControl):

    def __init__(self, *args, **kwargs):
        TrialControl.__init__(self, *args, **kwargs)
        self.logger = logging.getLogger('CopySpellerControl')
        self.progress_string = ''

    def _setup_trial(self):
        self._trial = CopySpellerDummy()


    def _setup_gui(self):
        self._gui = CopySpellerWidget(self, self._trial_gui)

    def _connect_signals(self):
        pass


class CopySpellerDummy(object):
    """Dummy replacement for D2Test class."""

    # Maybe only the recreate meth is needed (pass)
    def __init__(self):
        self.finished = False
        self.recreate()

    def recreate(self):
        self.finished = False

    def stop(self):
        self.finished = True
