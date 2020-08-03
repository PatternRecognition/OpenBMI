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

class Electrode(object):
    def __init__(self, name, coords):
        self.name = name
        self.coords = coords
        self.__init_attributes()

    def __init_attributes(self):
        self.voltage = 0.
        self.unit = 'mV'

    def __str__(self):
        return '%s: %s, %f %s' % (self.name, str(self.coords), self.voltage,
                                 self.unit)

    def set_voltage(self, v):
        self.voltage = v

