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

from unittest import TestCase
from time import sleep

from AlarmControl import AlarmControl

class StressControl(AlarmControl):
    def init(self):
        AlarmControl.init(self)
        self.common_stop_time = 0
        self.common_time_scale = 10000000

    def random_message(self, polling_frequency):
        return
        AlarmControl.random_message(self, polling_frequency)
        self.ack_failure()
        self.ack_dysfunction()

    def _schedule_next_message(self):
        if not self._finished:
            kind = self._next_message_kind
            msg = self._message_factory.message(kind)
            if self._schedule_message(msg):
                self._emit_messages()
            if self._stall_message_generation:
                sleep(0.5)
                self.ack_dysfunction(True)
                self.resolve_failure()
                self.ack_dysfunction(True)
            else:
                self.ack_dysfunction(True)
                self._time_next_message()

class StressTest(TestCase):
    def test_stress(self):
        alarm = StressControl()
        alarm.on_init()
        alarm.on_play()
