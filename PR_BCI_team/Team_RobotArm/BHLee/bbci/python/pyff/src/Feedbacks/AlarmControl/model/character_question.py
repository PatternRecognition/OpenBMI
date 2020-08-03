__copyright__ = """ Copyright (c) 2009 Torsten Schmits

This program is free software; you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation; either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this
program; if not, see <http://www.gnu.org/licenses/>.

"""

from random import Random

class CharacterQuestion(object):
    _random = Random('BBCI')
    _letters = ('D', 'E', 'G', 'K', 'P', 'R')
    _numbers = range(4, 10)
    _names = ['eine Zahl', 'ein Buchstabe']

    def __init__(self, statement=False):
        self._true = statement or self._random.randint(0, 1)
        self._use_letter = self._random.randint(0, 1)
        if self._use_letter:
            index = self._true
            sampler = self._letters
        else:
            index = 1 - self._true
            sampler = self._numbers
        char = self._random.sample(sampler, 1)[0]
        name = self._names[index]
        if statement:
            self.text = '%s ist %s.' % (char, name)
        else:
            self.text = 'Ist %s %s?' % (char, name)

    def __str__(self):
        return self.text

    def solve(self, answer):
        return answer == self._true
