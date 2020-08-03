__copyright__ = """ Copyright (c) 2010-2011 Torsten Schmits

This program is free software; you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by the
Free Software Foundation; either version 3 of the License, or (at your
option) any later version.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, see <http://www.gnu.org/licenses/>.
"""

from VisionEgg.Text import Text

class Symbol(Text):
    def __init__(self, symbol, size=72, mag_factor=1.2, **kw):
        self._symbol = symbol
        Text.__init__(self, text=str(symbol), font_size=size, anchor='center',
                      **kw)
        self._size = self.parameters.size
        self._mag_factor = mag_factor
        self.set(ignore_size_parameter=False)

    def enhance(self, nr):
	if nr is 1:
        	self.set(size=(int(self._size[0] * self._mag_factor),
                       int(self._size[1] * self._mag_factor)))
	if nr is 2:	
		self.set(color=(1.0,0.0,0.0))
	if nr is 3:
		self.set(angle=315)

    def reduce(self):
        self.set(size=self._size)
	self.set(color=(1.0,1.0,1.0))
	self.set(angle=0.0)

    def __repr__(self):
        return '%s(%s)' % (self.__class__.__name__, self._symbol)

    def __eq__(self, sym):
        return self._symbol == sym

class HighlightGroup(list):
    def __init__(self, elements, is_column, index):
        list.__init__(self, filter(None, elements))
        self.is_column = is_column
        self.index = index

    def select(self, nr):
        for element in self:
            element.enhance(nr)

    def deselect(self):
        for element in self:
            element.reduce()
