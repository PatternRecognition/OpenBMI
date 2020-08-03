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

import random

from FeedbackBase.VisionEggFeedback import VisionEggFeedback
from lib.speller import Speller

from P300Matrix.config import Config
from P300Matrix.view import View
from P300Matrix.util.marker import matrix_trigger

class P300Matrix(Speller, VisionEggFeedback, Config):
    def __init__(self, *a, **kw):
        VisionEggFeedback.__init__(self, view_type=View, *a, **kw)
        Speller.__init__(self)
        Config.__init__(self)

    @Speller.stimulus
    def flash(self, sequence):
        """ Initiate a series of row/column flashes randomly. """
        times = [self.highlight_time, self.between_highlights_time]
        def gen():
            for i in sequence:
                group = self._view.matrix.group(i)
                if self.trial_type==2:
                    self._trigger(matrix_trigger(group,
                                                 self.current_target))
                else:
                    self._trigger(matrix_trigger(group,
                                                 self._trial._save_target))
		"""unsere Modifikation """
                group.select(self.stimulus_type)
                yield
                group.deselect()
                yield
        self.stimulus_sequence(gen(), times).run()

    @Speller.sequences
    def random_sequences(self):
        last = []
        for i in range(self.nr_sequences):
            pop = range(self._view.matrix.group_count)
            def seq():
                for i in list(pop):
                    dupe = True
                    while dupe:
                        sample = random.choice(pop)
                        offset = max(0, self.min_dist - i)
                        dupe = offset and sample in last[-offset:]
                    pop.remove(sample)
                    yield sample
            last = list(seq())
            yield last
