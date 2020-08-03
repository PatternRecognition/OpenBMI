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

from itertools import izip, product
from math import ceil

from P300Matrix.model.layout import MatrixLayout
from P300Matrix.model.symbol import Symbol

class LayoutFactory(object):
    def __init__(self, screen, symbol_size=72, mag_factor=1.2):
        self._screen = screen
        self._symbol_size = symbol_size
        self._mag_factor = mag_factor

    def symbol(self, letter):
        return Symbol(letter, self._symbol_size, mag_factor=self._mag_factor)

    def symbols(self, letters):
        return map(self.symbol, letters)

    def matrix(self, elements, col_count, size):
        row_count = int(ceil(float(len(elements)) / col_count))
        def steps(size, count):
            dist = size / count
            return (dist * (i + 0.5) for i in xrange(count))
        row_x, col_y = map(steps, size, [col_count, row_count])
        for (y, x), symbol in izip(product(col_y, row_x), elements):
            symbol.set(position=(x, size[1] - y))
        return MatrixLayout(self._screen, elements, col_count, size=size)
