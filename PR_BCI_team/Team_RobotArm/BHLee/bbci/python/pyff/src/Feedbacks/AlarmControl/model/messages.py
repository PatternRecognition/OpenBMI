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
from threading import Timer

from PyQt4.QtCore import QAbstractTableModel, QModelIndex, Qt, QVariant, SIGNAL
from PyQt4.QtGui import QBrush, QFont

from AlarmControl.util.list import index_if
from AlarmControl.util import constants

class Message(object):
    """ An emitted message, to be displayed as active and not
    acknowledged. Stores information about the date and time it had
    been emitted and the info text and subsystem.

    """
    def __init__(self, **attrs):
        map(lambda i: setattr(self, *i), attrs.iteritems())
        self.subsystem = 'FOO'
        self._init_table_data()
        self.failure = False

    def _init_table_data(self):
        self.table_data = [self.date, self.time, self.subsystem, str(self.question)]

    def __getitem__(self, index):
        return self.table_data[index]

    @property
    def is_ackable(self):
        return self.ackable

class FailureMessage(Message):
    def __init__(self, **attrs):
        Message.__init__(self, **attrs)
        self.failure = True
        self.resolved = False
        self.recognized = False

    def resolve(self):
        self.resolved = True

    def recognize(self):
        self.recognized = True

class ArchiveMessage(Message):
    """ Message variant to be stored in one of the archives.
    Additionally contains info about the point of time it left the
    active message list.

    """
    def _init_table_data(self):
        self.table_data = [self.date, self.time, self.date_quit, self.time_quit,
                           self.subsystem, str(self.question)]

class Messages(QAbstractTableModel):
    headers = ['Datum', 'Uhrzeit', 'SPS', 'Meldungstext']

    def __init__(self, parent=None):
        self.logger = logging.getLogger("Messages")
        QAbstractTableModel.__init__(self, parent)
        self.__init_attributes()

    def __init_attributes(self):
        self.messages = []
        self.ackables_count = 0
        self._font = QFont()
        self._font.setPointSize(12)
        self._font.setBold(True)
        self._data_role_handlers = { Qt.DisplayRole: self.text,
                                     Qt.BackgroundRole: self.color,
                                     Qt.ToolTipRole: self.actions,
                                     Qt.FontRole: self.font }

    def push(self, message):
        """ Inform the Qt model about the insertion and add the message
        to the archive.

        """
        r = self.rowCount()
        self.beginInsertRows(QModelIndex(), r, r)
        self.messages.append(message)
        self.endInsertRows()
        self.ackables_count += message.ackable

    def ack(self, index):
        return

    def _remove_range(self, lower, upper):
        removals = self.messages[lower:upper + 1]
        self.beginRemoveRows(QModelIndex(), lower, upper)
        del self.messages[lower:upper + 1]
        self.endRemoveRows()
        self.ackables_count -= len(filter(lambda m: m.ackable, removals))
        return removals

    def clear(self):
        """ Remove all messages.

        """
        self.beginRemoveRows(QModelIndex(), 0, self.rowCount())
        self.messages = []
        self.endRemoveRows()

    def data(self, index, role=Qt.DisplayRole):
        handler = self._data_role_handlers.get(role, self._data_unhandled)
        if not index.isValid():
            handler = self._data_unhandled
        return handler(index.row(), index.column())

    def _data_unhandled(self, r=None, c=None):
        return QVariant()

    def text(self, row, col):
        try:
            c = self[row][col]
        except IndexError, e:
            self.logger.debug('Messages.text: ' + str(e))
            return QVariant('null')
        if isinstance(c, str):
            c = c.decode('utf-8')
        return QVariant(c)

    def font(self, row, col):
        return QVariant(self._font)

    def color(self, row, col):
        try:
            return QVariant(QBrush(self[row].color))
        except IndexError, e:
            self.logger.debug('Messages.color: ' + str(e))
            return QVariant(QBrush(Qt.gray))

    def comment(self, row, col):
        c = self[row].comment
        if isinstance(c, str):
            c = c.decode('utf-8')
        return QVariant(c)

    def actions(self, row, col):
        a = ', '.join(map(str, self[row].actions))
        if isinstance(a, str):
            a = a.decode('utf-8')
        return QVariant(a)

    def __getitem__(self, index):
        return self.messages[index]

    def __len__(self):
        return len(self.messages)

    def set(self, messages):
        self.messages = messages

    def rowCount(self, parent=QModelIndex()):
        return len(self.messages)

    def columnCount(self, parent=QModelIndex()):
        return len(self.headers)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return QVariant(self.headers[section])
        else:
            return QAbstractTableModel.headerData(self, section, orientation,
                                                  role)

    @property
    def first_ackable(self):
        """ Return the message in the list with the lowest index, that
        is ackable.

        """
        index = self.first_ackable_index
        return self.messages[index] if index is not None else None

    @property
    def first_ackable_index(self):
        """ Return the lowest index of an ackable messsage.

        """
        return index_if(lambda m: m.ackable, self.messages)

    @property
    def can_ack(self):
        return False

class ActiveMessages(Messages):
    """ Extends Messages with an acknowledgment function to remove
    messages and a tracking list of non ackable messages.

    """
    wait_for_enter = 0
    wait_for_recognize = 1
    wait_for_trial = 2
    wait_for_ack = 3

    def __init__(self, parent=None):
        super(ActiveMessages, self).__init__(parent)
        self.__init_attributes()
        self._blink_color_active = True
        self.resume_blinking()

    def __init_attributes(self):
        self.blink_delay = .5
        self._direction_turn_right = constants.direction_turn_right
        self._direction_ack = constants.direction_ack
        self._status = self.wait_for_recognize

    def _start_timer(self):
        self._blink_timer = Timer(self.blink_delay, self.blink)
        self._blink_timer.start()

    def stop_timer(self):
        self._blink_timer.cancel()
        self.suspend_blinking()

    def push(self, message):
        Messages.push(self, message)
        self._blink_color_active = True

    def recognize(self):
        self._status = self.wait_for_enter

    def start_trial(self):
        self._status = self.wait_for_trial

    def stop_trial(self):
        self._status = self.wait_for_ack

    def ack(self):
        """ Remove all messages older than the oldest ackable,
        inclusive.

        """
        if self._no_blink:
            self.resume_blinking()
            self._status = self.wait_for_recognize
        if self.can_ack:
            return self._remove_range(0, self.first_ackable_index)

    @property
    def can_ack(self):
        return self.first_ackable_index is not None

    def blink(self):
        if not self._no_blink:
            self._blink_color_active = not self._blink_color_active
            self._update_blinking_row()
            self._start_timer()

    def suspend_blinking(self):
        self._no_blink = True
        self._blink_color_active = True
        self._blink_timer.cancel()
        self._update_blinking_row()

    def resume_blinking(self):
        self._no_blink = False
        self._start_timer()

    def _update_blinking_row(self):
        if self.can_ack:
            fa = self.first_ackable_index
            self.emit(SIGNAL('dataChanged(const QModelIndex &,' +
                             ' const QModelIndex &)'),
                      self.index(fa, 0),
                      self.index(fa, self.columnCount() - 1))

    def text(self, row, col):
        """ Override text query to supply directional messages
        when neccessary.

        """
        if row == self.first_ackable_index and col == 3:
            if self._status == self.wait_for_enter:
                return self._direction_turn_right
            elif self._status == self.wait_for_ack:
                return self._direction_ack
        return Messages.text(self, row, col)

    def color(self, row, col):
        """ Override color query to alternatingly return white for the
        active message.

        """
        if row == self.first_ackable_index:
            if not self._blink_color_active:
                return QVariant(QBrush(Qt.white))
        return super(ActiveMessages, self).color(row, col)

class FailureMessages(Messages):
    def ack(self):
        """ Remove messages at the given indices.

        """
        messages = self._remove_range(0, 0)
        map(FailureMessage.resolve, messages)
        return messages

class MessageArchive(Messages):
    headers = ['Datum', 'Uhrzeit', 'Datum Quit', 'Uhrzeit Quit', 'SPS', 'Meldungstext']

    #def columnCount(self, parent=QModelIndex()):
        #return 6

class ShortTermArchive(MessageArchive):
    """ Extends MessageArchive by storage limits. These are checked when a
    message is added to the archive.

    """
    def __init__(self, limit=20, *args):
        Messages.__init__(self, *args)
        self.limit = limit

    def push(self, message):
        """ Add a message to the archive and check limits.

        """
        Messages.push(self, message)
        self.purge()

    def purge(self):
        """ Check if storage limits have been exceeded, delete oldest 
        messages first.

        """
        excess = self.rowCount() - self.limit
        if excess > 0:
            self._remove_range(0, excess - 1)

class LongTermArchive(MessageArchive):
    def __init__(self, *args):
        Messages.__init__(self, *args)

