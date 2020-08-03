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

import logging
import random

from PyQt4.QtGui import QMainWindow, QDialog, QWidget
from PyQt4.QtCore import Qt
from PyQt4.QtCore import QObject, SIGNAL, QTimer

from .main import Ui_Main
from AlarmControl.gui.d2test import Ui_d2test
from AlarmControl.gui.copyspeller import Ui_CopySpeller
from .trial import Ui_trial
from AlarmControl.util import constants
from AlarmControl.util import triggers
from AlarmControl.gui.lorem import lorem

class ExclusiveWidget(object):
    """ Widget class with focus locking.

    """
    def __init__(self):
        self.logger = logging.getLogger("ExclusiveWidget")
        self.active = True

    def lose_focus(self):
        self.logger.debug("lose_focus.")
        self.active = False

    def gain_focus(self):
        self.logger.debug("gain_focus.")
        self.active = True
        self.activateWindow()
        self.setFocus()

    def focusOutEvent(self, event):
        """ Work around the focus issue, as selective modality requires
        hiding of a window.

        """
        self.logger.debug("focusOutEvent.")
        if self.active:
            self.gain_focus()

class MainGUI(ExclusiveWidget, QMainWindow, Ui_Main):
    def __init__(self, control, key_yes='f', key_no='l'):
        self.logger = logging.getLogger("MainGUI")
        QMainWindow.__init__(self)
        ExclusiveWidget.__init__(self)
        self.setupUi(self)
        self.control = control
        self.set_keys(key_yes, key_no)

    def set_keys(self, key_yes, key_no):
        self._key_yes = key_yes
        self._key_no = key_no

    def keyPressEvent(self, event):
        self.logger.debug("keyPressEvent")
        text = event.text()
        if event.key() == Qt.Key_Return:
            self.control.launch_trial()
        elif text in [self._key_yes, self._key_no]:
            self.control.ack_dysfunction(text == self._key_yes)

    def closeEvent(self, event):
        self.logger.debug("closeEvent")
        self.control.close_gui()

class TrialGUI(QWidget, Ui_trial):
    def __init__(self, control):
        self.logger = logging.getLogger("TrialGUI")
        QWidget.__init__(self)
        self.setupUi(self)
        self.set_status(u'Lösen Sie bitte folgende Aufgabe so schnell' +
                     u' aber auch richtig wie möglich und geben unten' +
                     ' Ihr Ergebnis ein!')
        self.trial = None
        self.control = control

    def set_trial(self, widget):
        """ Remove the current trial child widget and add the argument.

        """
        self.logger.debug("set_trial")
        if self.trial is not None:
            self.trial.hide()
            self.verticalLayout.removeWidget(self.trial)
        self.trial = widget
        self.verticalLayout.addWidget(self.trial)
        self.trial.show()

    def start_trial(self, widget, model):
        self.logger.debug("start_trial")
        self.set_trial(widget)
        self.trial.gain_focus()
        self.trial.start(model)

    def stop_trial(self):
        self.logger.debug("stop_trial")
        self.trial.stop()
        self.trial.lose_focus()

    def ack(self):
        self.logger.debug("ack")
        self.set_status('')

    def recognize(self):
        self.logger.debug("recoginze")
        self.set_status(constants.direction_press_enter)

    def set_status(self, string):
        self.logger.debug("set_status")
        self.status.setText(string)

    def closeEvent(self, event):
        self.logger.debug("closeEvent")
        self.control.close_gui()

class TrialWidget(ExclusiveWidget, QWidget):
    def __init__(self, control, gui):
        self.logger = logging.getLogger("TrialWidget")
        QWidget.__init__(self)
        self._control = control
        self._gui = gui

    def _status(self, string):
        self.logger.debug("_status")
        self._gui.set_status(string)

class D2TestWidget(TrialWidget, Ui_d2test):
    def __init__(self, control, gui, key_yes, key_no):
        self.logger = logging.getLogger("D2TestWidget")
        TrialWidget.__init__(self, control, gui)
        self.setupUi(self)
        self.set_keys(key_yes, key_no)

    def set_keys(self, key_yes, key_no):
        self._key_yes = key_yes
        self._key_no = key_no

    def keyPressEvent(self, event):
        self.logger.debug("keyPressEvent")
        if not event.isAutoRepeat():
            text = str(event.text())
            if text == self._key_yes:
                self.solve(True)
            elif text == self._key_no:
                self.solve(False)

    def solve(self, solution):
        self.logger.debug("solve")
        self._control.solve(solution)
        self.progress.setValue(self._d2_test.progress)
        self.logger.debug(str(self._d2_test.progress))

    def start(self, d2test):
        self.logger.debug("start")
        self._status(u'Drücken Sie "%s", wenn ein "d" mit zwei Strichen'
                     u' sichtbar ist, sonst "%s"' % (self._key_yes, self._key_no))
        self._d2_test = d2test
        self.progress.setRange(0, self._d2_test.count)
        self.progress.setValue(0)
        self.next_atom()

    def stop(self):
        self.logger.debug("stop")
        self.set_atom_text('', '', '')

    def set_text(self, status, ignore=None):
        self.logger.debug("set_text")
        self._status(status)

    def next_atom(self):
        self.logger.debug("next_atom")
        atom = self._d2_test.current
        self.set_atom_text(atom.letter, atom.top_bars, atom.bottom_bars)

    def set_atom_text(self, letter, top, bottom):
        self.logger.debug("set_atom_text")
        self.letter.setText(letter)
        self.top_bar.setText(top)
        self.bottom_bar.setText(bottom)


class CopySpellerWidget(TrialWidget, Ui_CopySpeller):
    def __init__(self, control, gui):
        self.logger = logging.getLogger("D2TestWidget")
        self.logger.debug('Initializing Copyspeller Widget')
        TrialWidget.__init__(self, control, gui)
        self.setupUi(self)
        self.wait_for_key = False
        # we rather not rely on auto connection, somehow the signal for clicked
        # buttons is triggered twice
        QObject.connect(self.pushButton_start, SIGNAL('clicked()'),
                        self.pb_start_clicked)
        # is the cs task running?
        self.running = False

    def trigger(self, value):
        """Please don't ask!"""
        self._control._control.trigger(value)

    def mystop(self):
        """Stop the copyspeller task (this method can be used within this class
        in contrast to stop().
        """
        # TODO: check if the cs is running (this method can also be triggered
        # from the outside even when cs is not running.
        if not self.running:
            return
        self.running = False
        self._control.stop_trial()
        self.trigger(triggers.TRIG_STOP_CS)

    def pb_start_clicked(self):
        self.logger.debug("pushButton clicked")
        if self.wait_for_key:
            self.wait_for_key = False
            self.start2()

    def start(self, d2test):
        self.logger.debug("start")
        self._status(u'Tippen sie den oberen Text in das untere Feld und \
                clicken sie anschließend auf "Weiter"')
        self.setFocus()
        self.wait_for_key = True
        self.pushButton_start.setEnabled(True)
        self.textEdit.setEnabled(False)
        self.textBrowser.setEnabled(False)
        self.textBrowser.setText(random.choice(lorem))
        self._d2_test = d2test

    def start2(self):
        self.running = True
        self.trigger(triggers.TRIG_START_CS)
        self.textEdit.setEnabled(True)
        self.textBrowser.setEnabled(True)
        self.textEdit.setFocus()
        self.pushButton_start.setEnabled(False)
        QTimer.singleShot(1000 * self._control._control.cs_duration,
                self.mystop)

    # this method is apparently called from outside as a result of
    # _self.control.stop_trial -- don't call it directly.
    def stop(self):
        self.logger.debug("stop")
        self.textEdit.clear()
        self.textBrowser.clear()

    def set_text(self, status, ignore=None):
        self.logger.debug("set_text")
        self._status(status)

    def focusOutEvent(self, event):
        """Hack to disable focus problem when clicking start."""
        return

