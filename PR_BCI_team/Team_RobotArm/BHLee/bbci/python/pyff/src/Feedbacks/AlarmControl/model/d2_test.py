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

from random import Random
from copy import copy
from itertools import repeat, imap
import logging

from AlarmControl.util.triggers import *

class D2Atom(object):
    _target_bar_counts = set(((0, 2), (1, 1), (2, 0)))
    _other_bar_counts = set(((0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)))
    _random = Random()

    def __init__(self, target, letters):
        self.is_target = target
        self._letters = letters
        self.setup()

    def setup(self):
        self._choose_letter()
        self._choose_bars()

    def _choose_letter(self):
        if self.is_target:
            self.letter = 'd'
        else:
            index = self._random.randint(0, len(self._letters) - 1)
            self.letter = self._letters[index]

    def _choose_bars(self):
        """ First, choose the set of bar count combinations, depending
        on whether the atom should be a target. If it shouldn't, but a
        'd' had been chosen by random, the combinations for targets must
        be removed. Then, generate the strings of bars according to a
        sampled combination.

        """
        if self.is_target:
            counts = self._target_bar_counts
        else:
            counts = copy(self._other_bar_counts)
            if self.letter == 'd':
                counts -= self._target_bar_counts
        self.bar_counts = self._random.sample(counts, 1)[0]
        self.top_bars = '|' * self.bar_counts[0]
        self.bottom_bars = '|' * self.bar_counts[1]

    def solve(self, solution):
        return solution == self.is_target

    def __str__(self):
        return self.top_bars + self.letter + self.bottom_bars

    def __eq__(self, other):
        return self.bar_counts == other.bar_counts and self.letter == \
                other.letter

    def __repr__(self):
        return 'D2Atom(%s)' % str(self)

class D2Test(object):
    def __init__(self, target_quota, count, letters):
        self.logger = logging.getLogger("D2Test")
        self._target_quota = target_quota
        self.count = count or 1
        self._letters = letters
        self._random = Random()
        self.current = None
        self._total = 0
        self._total_correct = 0
        self.recreate()

    def recreate(self):
        self.finished = False
        self._solutions = []
        targets = (self._random.uniform(0, 1) < self._target_quota
                   for i in xrange(self.count))
        self._atoms = [D2Atom(targets.next(), self._letters)]
        for t in targets:
            a = D2Atom(t, self._letters)
            while a == self._atoms[-1]:
                a.setup()
            self._atoms.append(a)
        self._atoms_iter = iter(self._atoms)
        self.next()

    def solve(self, solution):
        self._solutions.append([self.current.is_target, solution])
        sol = self.current.solve(solution)
        self.next()
        return sol

    def next(self):
        try:
            self.current = self._atoms_iter.next()
        except StopIteration:
            self.finished = True

    def stop(self):
        self.finished = True
        self._total += self.progress
        self._total_correct += len(filter(lambda (a, b): a == b,
                                          self._solutions))

    @property
    def progress(self):
        return len(self._solutions)

    @property
    def global_progress(self):
        return self._total_correct, self._total

    @property
    def trigger_code(self):
        letter = self.current.letter.upper()
        try:
            base = eval('TRIG_D2_%s_BASE' % letter)
        except NameError:
            base = -100
            self.logger.debug('No matching trigger for letter %s' % letter)
        return base + 10 * int(self.current.bar_counts[0]) + \
                int(self.current.bar_counts[1])

