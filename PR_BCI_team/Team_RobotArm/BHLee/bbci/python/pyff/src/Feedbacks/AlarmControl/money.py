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

from time import time
import logging

class Bookie(object):
    def __init__(self):
        self.logger = logging.getLogger("Bookie")
        self.reset()

    def reset(self):
        self.logger.debug("reset.")
        self.total_money = 0.

    def start_trial(self):
        self.logger.debug("start_trial.")
        self._in_trial = True
        self._trial_start = time()

    def solve_d2(self, success):
        self.calc_reward_d2(success)

    def calc_reward_d2(self, success):
        self.logger.debug("calc_reward_d2.")
        if success:
            self.total_money += self.d2_reward
        else:
            self.total_money -= self.d2_penalty
        self.logger.debug('Total: ' + str(self.total_money))
