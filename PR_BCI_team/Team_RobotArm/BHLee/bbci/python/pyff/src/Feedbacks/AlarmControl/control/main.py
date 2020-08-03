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
import logging
from time import sleep
from copy import copy
from random import Random
from itertools import izip
from threading import Timer

from PyQt4.QtCore import QObject, SIGNAL, SLOT, qDebug, QSignalMapper, \
                         QString, Qt, QTimer
from PyQt4.QtGui import QPushButton, QApplication, QMessageBox

from AlarmControl.message_factory import MessageFactory
from AlarmControl.model.messages import ShortTermArchive, LongTermArchive, \
                                        ActiveMessages, FailureMessages, \
                                        Messages
from AlarmControl.model.d2_test import D2Test
from AlarmControl.util.metadata import datadir
from AlarmControl.util.triggers import *
from AlarmControl.gui.gui import MainGUI, TrialGUI
from AlarmControl.util.clock import Clock
from AlarmControl.util.string import camelcaseify
from AlarmControl.control.trial import TrialControl
from AlarmControl.control.d2_test import D2TestControl
from AlarmControl.control.copyspeller import CopySpellerControl
from AlarmControl.control.config import Config
from AlarmControl.util.list import index_if
from AlarmControl.util import constants

class Control(QObject, Config):
    message_dir = path.join(datadir, 'messages')
    _message_schedule_path = path.join(message_dir,
                            'message_queue.txt')

    def __init__(self, trigger):
        # Multiple inheritance is great, yeah...
        if not hasattr(self, "logger"):
            self.logger = logging.getLogger("Control")
        self.logger.debug("foo")
        QObject.__init__(self)
        Config.__init__(self)
        self.trigger = trigger
        self._initialized = False
        # schedule time for copyspeller task [s]
        self.t_copyspeller = 20 
        # duration of the copyspeller task [s]
        self.cs_duration = 30
        # are we waiting for the subject to turn to the left monitor?
        self.finalize_cs = False

    def init(self):
        self.logger.debug("Initializing.")
        self.__init_attributes()
        self._setup_messages()
        self._initialized = True

    def __init_attributes(self):
        self.logger.debug("Initializing Attributes.")
        self.__init_internals()

    def __init_configurables(self):
        self.logger.debug("Initializing Configurables.")
        self.trials = []
        self.trials += ['d2_test']
        #self.time_scale = 4.
        #self.main_gui_size = (600, 400)
        #self.fail_gui_size = (400, 300)
        #self.main_gui_pos = (0, 100)
        #self.fail_gui_pos = (800, 100)
        #self.trials = []
        #self.trials += ['d2_test']
        #clock = Clock()
        #self.trial_log_file = 'data/log/trials_%s_%s' % (clock.date, clock.time)
        pass

    def __init_internals(self):
        self.logger.debug("Initializing Internals.")
        self._trial_log = []
        self._waiting_for_resolution = False
        self._stall_message_generation = False
        self._gui_running = False
        self._ack_count = 0
        self._finished = False
        self._random = Random()
        self._message_timer = Timer(10000., lambda: True)
        self._setup_desktop()
        self._setup_windows()
        self._main_gui.label.setText('')
        if self.do_d2:
            self._trial = D2TestControl(self, self._trial_gui)
        else:
            self.common_fail_gui_size = (700, 400)
            self._trial = CopySpellerControl(self, self._trial_gui)

    def _setup_messages(self):
        self.logger.debug("Setting up Messages.")
        self._messages = {}
        self._message_factory = MessageFactory()
        self._message_schedule = self._message_factory.schedule_from_file(
            self._message_schedule_path)
        self._messages['active'] = ActiveMessages()
        self._messages['failure'] = FailureMessages()
        self._messages['shortterm'] = ShortTermArchive()
        self._messages['longterm'] = LongTermArchive()
        self._messages['scheduled'] = Messages()
        self._next_message_index = 0

    def launch_gui(self):
        self.logger.debug("Launching GUI.")
        self.update_parameters()
        self._trial.init()
        self._connect_signals()
        self.set_message_archive('active')
        self._gui_running = True
        self._trial_gui.show()
        self._main_gui.show()
        self.update_parameters()
        self.trigger(TRIG_START)
        self._schedule_next_message()
        if not self.do_d2:
            self.schedule_copyspeller()

    def _setup_desktop(self):
        self.logger.debug("Setting up Desktop.")
        self._desktop = QApplication.desktop()
        self._num_screens = self._desktop.numScreens()
        main_screen_num = self._desktop.primaryScreen()
        self._main_screen_res = self._desktop.screenGeometry(main_screen_num)

    def _setup_windows(self):
        self.logger.debug("Setting up Windows.")
        self._main_gui = MainGUI(self)
        self._trial_gui = TrialGUI(self)
        self._main_gui.setFocusPolicy(Qt.StrongFocus)
        self._trial_gui.setFocusPolicy(Qt.StrongFocus)
        self._main_gui.gain_focus()

    def _connect_signals(self):
        self.logger.debug("Connection Signals.")
        self.alarm = self._main_gui.alarm_control
        actions = self.alarm.actions
        for action in actions.keys():
            if hasattr(self, action):
                QObject.connect(actions[action], SIGNAL('triggered()'),
                                getattr(self, action))
        QObject.connect(self, SIGNAL("_schedule_signal()"),
                        self._schedule_next_message)


    def close_gui(self):
        self.logger.debug("Closing GUI.")
        self._finished = True
        self._stop_timers()
        self._main_gui.close()
        self._trial_gui.close()
        self._gui_running = False

    def _schedule_next_message_callback(self):
        self.emit(SIGNAL("_schedule_signal()"))

    def _schedule_next_message(self):
        self.logger.debug("Scheduling Next Message.")
        if not self._finished:
            kind = self._next_message_kind
            msg = self._message_factory.message(kind)
            if self._schedule_message(msg):
                self._emit_messages()
            if not self._stall_message_generation:
                self._time_next_message()

    def schedule_copyspeller(self):
        self.logger.debug("Scheduling copyspeller")
        if self.t_copyspeller > 0:
            self.timer = QTimer.singleShot(1000 * self.t_copyspeller,
                                           self.start_copyspeller)

    def start_copyspeller(self):
        self.logger.debug("Starting copyspeller.")
        # TODO: is the trigger set?
        self.trigger(TRIG_START_NEBENAUFGABE)
        self._main_gui.label.setText('Bitte zum rechten Monitor drehen.')
        msg = self._message_factory.message(1)
        msg.question = 'Bitte auf den rechten Monitor wechseln!'
        msg._init_table_data()
        self._messages['active'].clear()
        if self._schedule_message(msg):
            self._emit_messages()
        self._start_trial()

    def _time_next_message(self):
        self.logger.debug("Timing Next Message.")
        self._next_message_index += 1
        if self._next_message_index >= len(self._message_schedule):
            self.on_stop()
        else:
            next = self._next_interval
            self.logger.debug(next)
            self.logger.debug(self._next_message_index)
            self._message_timer = Timer(max(next, 0),
                                        self._schedule_next_message_callback)
            self._message_timer.start()

    @property
    def _current_message(self):
        return self._message_schedule[self._next_message_index]

    @property
    def _next_interval(self):
        return float(self._current_message[0]) / self.time_scale

    @property
    def _next_message_kind(self):
        return self._current_message[1]

    def _schedule_message(self, message):
        """ Schedule a chosen message for emission, return whether
        emission should be performed (message is failure or warning).

        """
        self.logger.debug("Scheduling Message.")
        if not self._finished:
            self._messages['scheduled'].push(message)
            if message.emit:
                return True
        return False

    def _emit_messages(self):
        """ Move all messages to the Qt model and show them in the main
        view, trigger with the newest message.

        """
        self.logger.debug("Emitting Messages.")
        if self._messages['scheduled']:
            for m in self._messages['scheduled']:
                self._push_message(m)
            self._trigger_emission(self._messages['scheduled'][-1])
            self._messages['scheduled'].clear()

    def _trigger_emission(self, message):
        self.logger.debug("triggering emission.")
        self.trigger(color_trig(message.kind))

    # Is this method dead? Looks like it's not called anywhere within the code
    # of Alarm Control.
    def _trigger_click(self, index):
        self.logger.debug("Triggering Click.")
        if len(self._current_message_list) > index:
            msg = self._current_message_list[index]
            if msg and msg.emit:
                self.trigger(color_trig(msg.kind, 'klick'))

    def _trigger_ack(self, message, correct=None):
        self.logger.debug("Triggering ACK.")
        if message:
            self.trigger(color_trig(message.kind, 'quit', correct))

    def _push_message(self, message):
        """ Add one message to the Qt model, which will update the view.
        If the message requires a trial, stop scheduling messages and
        add the current message to the failure list.

        """
        self.logger.debug("Pushing Message.")
        self._messages['active'].push(message)
        if self.do_d2:
            if message.failure:
                self.pause_message_generation()
                self._messages['failure'].push(message)

    def update(self):
        self.logger.debug("Updating.")
        self.alarm.update_time()
        self.alarm.update_ack_count(self._current_message_list.ackables_count)

    def update_parameters(self):
        self.logger.debug("Updating Parameters.")
        if self._initialized:
            self.apply_config(self._trial._bookie,
                            self._trial, self)
            if self._gui_running:
                self._update_window_constraints()
            TrialControl.log_file = self.trial_log_file
            self._trial.update_parameters()
            self._messages['active'].blink_delay = self.blink_speed
            self._main_gui.set_keys(self.key_yes, self.key_no)

    def _update_window_constraints(self):
        self.logger.debug("Updating Window Constraints.")
        self._main_gui.resize(*self.main_gui_size)
        self._trial_gui.resize(*self.fail_gui_size)
        self._main_gui.move(*self.main_gui_pos)
        self._trial_gui.move(*self.fail_gui_pos)

    def reset(self):
        self.logger.debug("Resetting.")
        map(Messages.clear, self._messages.itervalues())
        self._trial.reset()

    def stop_game(self):
        self.logger.debug("Stopping Game.")
        if not self._finished:
            self.logger.debug('game over')
            self._finished = True
            while self._waiting_for_resolution:
                sleep(0.1)
            self.trigger(TRIG_END)
            self._stop_timers()
            self._trial.stop_game()
            TrialControl.write_log()

    on_stop = stop_game

    def set_message_archive(self, new):
        self.logger.debug("Setting Message Archive.")
        self._current_message_list = self._messages[new]
        self.alarm.set_model(self._current_message_list)
        self.update()

    def show_active_messages(self):
        self.logger.debug("Showing Active Messages.")
        self.set_message_archive('active')

    def show_shortterm_archive(self):
        self.logger.debug("Showing Shortterm Archive.")
        self.set_message_archive('shortterm')

    def show_longterm_archive(self):
        self.logger.debug("Showing Longterm Archive.")
        self.set_message_archive('longterm')

    def ack(self, correct=None):
        self.logger.debug("ACK'ing.")
        if self._messages['active'].can_ack:
            messages = self._messages['active'].ack()
            if messages:
                self._ack_count += len(messages)
                for message in messages:
                    archive_msg = self._message_factory.archive_message(message)
                    self._messages['shortterm'].push(archive_msg)
                    self._messages['longterm'].push(archive_msg)
                self._trigger_ack(messages[-1], correct)

    def recognize_failure(self):
        self.logger.debug("Recognizing Failure.")
        am = self._messages['active']
        msg = am.first_ackable
        if msg and msg.failure and not msg.recognized:
            msg.recognize()
            self._trial_gui.recognize()
            am.suspend_blinking()
            am.recognize()
            self.trigger(TRIG_ROT_GESEHEN)
            if self.finalize_cs:
                self.finalize_cs = False
                self.trigger(TRIG_STOP_NEBENAUFGABE)

    def launch_trial(self):
        self.logger.debug("Launching Trial.")
        msg = self._messages['active'].first_ackable
        if msg and msg.failure and not self._waiting_for_resolution \
           and msg.recognized and not msg.resolved and self._need_new_trial:
            self.trigger(TRIG_START_NEBENAUFGABE)
            self._start_trial()

    def ack_failure(self):
        self.logger.debug("ACK'ing Failure.")
        am = self._messages['active']
        msg = am.first_ackable
        if not self._finished:
            if msg and msg.failure and not self._waiting_for_resolution \
            and msg.resolved:
                self.start_message_generation()
                self._trial_gui.ack()
                self.ack()
                self._time_next_message()

    def start_message_generation(self):
        self.logger.debug("Starting Message Generation.")
        self._stall_message_generation = False

    def pause_message_generation(self):
        self.logger.debug("Pausing Message Generation.")
        self._stall_message_generation = True

    def ack_dysfunction(self, answer):
        """ Solve a yellow or red message. In case of yellow, the
        message disappears, with a red one, the trial ist prepared.

        """
        self.logger.debug("ACK'ing Dysfunction.")
        msg = self._messages['active'].first_ackable
        if not self._finished:
            if msg:
                correct = msg.question.solve(answer)
                # FIXME1 we should rather send triggers like yellow-correct,
                # yellow incorrect, etc comment the below stuff out to get the
                # original behaviour
                # also comment out FIXME2 on bottom of this function.
                #self.trigger(TRIG_HAUPTFENSTER_RICHTIG if correct else
                #            TRIG_HAUPTFENSTER_FALSCH)
                if msg.failure:
                    self.recognize_failure()
                    if self.do_d2:
                        self.ack_failure()
                    else:
                        self.ack(correct)
                else:
                    self.ack(correct)
            # FIXME2
            #elif self._messages['active']:
            #    self.trigger(TRIG_BLAU_BEANTWORTET)

    def resolve_failure(self):
        self.logger.debug("Resolving Failure.")
        msg = self._messages['active'].first_ackable
        if msg and msg.failure and not self._waiting_for_resolution \
           and msg.recognized:
            msg.resolve()

    def _start_trial(self):
        self.logger.debug("Starting Trial.")
        self._waiting_for_resolution = True
        self._main_gui.lose_focus()
        self._trial.start_trial()
        self._messages['active'].start_trial()
        self._trial_gui.activateWindow()

    @property
    def _need_new_trial(self):
        fails = self._messages['failure']
        return len(fails) > 0 \
               and not self._waiting_for_resolution \
               and not self._finished

    def stop_trial(self):
        self.logger.debug("Stopping Trial.")
        self._main_gui.gain_focus()
        self._waiting_for_resolution = False
        self.resolve_failure()
        self._messages['active'].stop_trial()
        self._trial_gui.set_status(self._trial.progress_string +
                               '\n' + constants.direction_turn_left)
        if not self.do_d2:
            self.schedule_copyspeller()
            self._main_gui.label.setText('')
            # delete all messages in the main gui
            self._messages['active'].clear()
            # force the participant to press enter so we know for sure when he
            # was staring at the main gui again
            msg = self._message_factory.message(1)
            if self._schedule_message(msg):
                self.finalize_cs = True
                self._emit_messages()


    def _stop_timers(self):
        self.logger.debug("Stopping Timers.")
        self._message_timer.cancel()
        self._messages['active'].stop_timer()

