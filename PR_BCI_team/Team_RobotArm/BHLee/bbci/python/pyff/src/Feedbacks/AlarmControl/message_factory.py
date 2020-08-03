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

from PyQt4.QtCore import Qt, qDebug

from util.clock import Clock
from model.messages import Message, ArchiveMessage, FailureMessage
from model.character_question import CharacterQuestion

timeframes = { 'Woche': 604800., 'Tag': 86400, 'Stunde': 3600. }

def parse_frequency(string):
    try:
        string = string.split('Std')[0]
        num, denom = string.split('/')
        return float(num) / (float(denom) * 3600.)
    except Exception, e:
        #print 'Frequency parse error: ' + string
        #print e
        return 0.

class MessageFactory(object):
    colors = { 1: Qt.red,
               2: Qt.yellow,
               3: Qt.cyan,
               4: Qt.gray }
    ackables = [1]
    clock = Clock()

    def message(self, kind):
        kind = int(kind)
        emit = kind in [1, 2, 3]
        ackable = kind in [1, 2]
        color = self.colors[kind]
        statement = kind in [3, 4]
        question = CharacterQuestion(statement)
        message_type = (FailureMessage if kind is 1 else Message)
        return message_type(date=self.clock.date, time=self.clock.time,
                            color=color, ackable=ackable, kind=kind,
                            emit=emit, question=question)

    def archive_message(self, message):
        attrs = message.__dict__
        return ArchiveMessage(date_quit=self.clock.date,
                              time_quit=self.clock.time, **attrs)

    def schedule_from_file(self, filename):
        with open(filename) as f:
            return map(self.csv_data, f.readlines())

    def csv_data(self, line):
        line = line.rstrip()
        data = line.split(' ', 1)
        if not len(data) == 2:
            raise Exception('Bad csv format: %s' % line)
        return data
