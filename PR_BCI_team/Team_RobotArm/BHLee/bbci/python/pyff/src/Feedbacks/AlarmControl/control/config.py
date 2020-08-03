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

import logging
from os import path

from AlarmControl.util.clock import Clock

class Config(object):
    def __init__(self):
        if not hasattr(self, "logger"):
            self.logger = logging.getLogger("Config")
        self.__init_attributes()
        self.__set_attributes()

    def __init_attributes(self):
        self.logger.debug("Config: Initializing Attributes.")
        self._dict_names = ['money', 'd2', 'common']
        self.__init_money_config()
        self.__init_d2_config()
        self.__init_common_config()

    def __init_money_config(self):
        """ Config values for the reward system.
        d2_{reward,penalty}: Amount of Euro to add/subtract for success
        resp. failure

        """
        self.logger.debug("Config: Initializing Money Config.")
        self._money = dict()
        self._money['d2_reward'] = .01
        self._money['d2_penalty'] = .01

    def __init_d2_config(self):
        """ Config values for the test d2.
        duration: Time available for a single d2 trial
        target_quota: Portion of actual d2s of the symbols
        count: Number of symbols for each d2 trial
        letters: Letters to be used
        key_{yes,no}: Input keys for target/no target

        """
        self.logger.debug("Config: Initializing D2 Config.")
        self._d2 = dict()
        self._d2['duration'] = 10.
        self._d2['target_quota'] = .5
        self._d2['count'] = 10
        self._d2['letters'] = ['p', 'd']
        self._d2['key_yes'] = 'j'
        self._d2['key_no'] = 'k'

    def __init_common_config(self):
        """ General options for the feedback.
        time_scale: Speedup factor
        {main,fail}_gui_{size,pos}: Initial size and position of the
                                    two windows
        trial_log_file: Filename to write the log to
        stop_time: Seconds the experiment should be running maximally
        blink_speed: Message color blinking interval in seconds
        key_{yes,no}: Input keys for answering the main gui questions

        """
        self.logger.debug("Config: Initalizing Common Config.")
        clock = Clock()
        self._common = dict()
        self._common['time_scale'] = 1.
        self._common['main_gui_size'] = (600, 400)
        self._common['fail_gui_size'] = (400, 300)
        self._common['main_gui_pos'] = (0, 100)
        self._common['fail_gui_pos'] = (800, 100)
        self._common['trial_log_file'] = path.join('data', 'log', 'trials_%s_%s'
                                                   % (clock.date, clock.time))
        self._common['stop_time'] = 60*45
        self._common['blink_speed'] = 0.7
        self._common['key_yes'] = 'j'
        self._common['key_no'] = 'k'

    def __set_attributes(self):
        """ Turn all the keys of the dictionaries into attributes
        prefixed by the dictionary name.

        """
        self.logger.debug("Config: Setting Attributes.")
        for c in self._dict_names:
            try:
                config = getattr(self, '_' + c)
                for k, v in config.iteritems():
                    setattr(self, c + '_' + k, v)
            except AttributeError:
                self.logger.debug('Config: Invalid name in Config._dict_names: %s' % c)

    def apply_config(self, money, d2, common):
        """ Transfer all the attributes set in __set_attributes to the
        corresponding arguments, determined by the dictionary name.

        """
        self.logger.debug("Config: Applying Config.")
        for c in self._dict_names:
            try:
                control = eval(c)
                config = getattr(self, '_' + c)
                for k in config:
                    try:
                        attr = c + '_' + k
                        setattr(control, k, getattr(self, attr))
                    except AttributeError:
                        self.logger.debug('Config: Invalid attribute in config dict %s: %s' % (c, k))
            except (NameError, AttributeError):
                self.logger.debug('Config: Invalid name in Config._dict_names: %s' % c)
