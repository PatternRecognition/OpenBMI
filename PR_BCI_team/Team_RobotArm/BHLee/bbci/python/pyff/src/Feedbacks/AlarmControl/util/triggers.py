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

TRIG_START, TRIG_END = 252, 253
TRIG_BLAU, TRIG_GELB, TRIG_ROT  = 20, 21, 22
TRIG_BLAU_KLICK, TRIG_GELB_KLICK, TRIG_ROT_KLICK = 30, 31, 32
TRIG_BLAU_QUIT, TRIG_GELB_QUIT, TRIG_ROT_QUIT = 40, 41, 42
TRIG_START_NEBENAUFGABE = 50
TRIG_EINGABE = 51
TRIG_RICHTIG, TRIG_FALSCH = 65, 66
TRIG_MONEY = 99
TRIG_ROT_GESEHEN = 70
TRIG_D2_D_BASE = 100
TRIG_D2_P_BASE = 200
TRIG_HAUPTFENSTER_RICHTIG = 71
TRIG_HAUPTFENSTER_FALSCH = 72
TRIG_BLAU_BEANTWORTET = 73
# only used in copyspeller
# start-cs -- stop-cs is the interval where the subject is guaranteed to look
# at the right monitor
# stop-nebenaufgabe  -- start-nebenaufgabe is the interval where the sujbect is
# guaranteed to look at the left monitor
# in the intervals start-na..start-cs and stop-cs..stop-na the behaviour is
# undefined
TRIG_START_CS = 80
TRIG_STOP_CS = 180
TRIG_STOP_NEBENAUFGABE = 150

OFFSET_CORRECT = 100
OFSETT_FALSE = 200

triggers = { 1: 'ROT',
             2: 'GELB',
             3: 'BLAU' }

def color_trig(color, suffix=None, correct=None):
    name = triggers[color]
    if suffix:
        name += '_' + suffix.upper()
    trig = eval('TRIG_' + name)
    if correct == True:
        trig += OFFSET_CORRECT
    elif correct == False:
        trig += OFSETT_FALSE
    return trig
