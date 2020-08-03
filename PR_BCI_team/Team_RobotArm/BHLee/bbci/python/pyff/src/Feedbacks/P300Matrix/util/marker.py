__copyright__ = """ Copyright (c) 2011 Torsten Schmits

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

TRIG_ROW_BASE = 11
TRIG_COL_BASE = 21
TRIG_TARGET_OFFSET = 20

def matrix_trigger(group, target):
    print "Current target symbol: %s" % target
    base = TRIG_COL_BASE if group.is_column else TRIG_ROW_BASE
    if target in group:
        base += TRIG_TARGET_OFFSET
    return base + group.index
