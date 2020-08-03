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

from VisionEgg.Core import Viewport

from P300Matrix.model.symbol import HighlightGroup

def transpose(input):
    return zip(*input)

class MatrixLayout(Viewport):
    def __init__(self, screen, elements, col_count, size):
        Viewport.__init__(self, screen=screen, size=size)
        self._elements = elements
        self.set(stimuli=elements)
        self._rows = map(None, *([iter(elements)] * col_count))
        self._cols = transpose(self._rows)
        self._row_count = len(self._rows)
        self.group_count = len(self._rows) + len(self._cols)

    def group(self, index):
        is_column = index >= self._row_count
        if is_column:
            index -= self._row_count
        g = (self._cols[index] if is_column else self._rows[index])
        return HighlightGroup(g, is_column, index)

    def highlight(self, group):
        self.group(group).select()
