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

from lib import marker
from lib.vision_egg import VisionEggView
from lib.vision_egg.model.target_word import TargetWord

from P300Matrix.factory.layout import LayoutFactory

class View(VisionEggView):
    def init(self):
        sz = self.screen.size
        target_height = self._word_vpos * 2
        self._target_viewport_height = sz[1] - target_height - self._word_margin
        self._target_word = TargetWord(position=(sz[0] / 2, self._word_vpos),
                                       center_at_target=True,
                                       symbol_size=self._word_font_size,
                                       target_size=self._word_target_font_size)
        self._target_word_viewport = Viewport(screen=self.screen,
                                              stimuli=self._target_word,
                                              size=(sz[0], target_height))
        self._target_word_viewport.set(position=(0,
                                                 self._target_viewport_height))
        self.add_viewport(self._target_word_viewport)
        self._setup_flash_layout()

    def _setup_flash_layout(self):
        sz = self.screen.size
        factory = LayoutFactory(self.screen, symbol_size=self._matrix_font_size,
                                mag_factor=self._magnification_factor)
        elements = factory.symbols(self._symbols)
        self.matrix = factory.matrix(elements, self._matrix_columns,
                                     (sz[0], self._target_viewport_height))
        self.add_viewport(self.matrix)

    def word(self, word):
        self._target_word.set_word(word)
        self.update()

    def next_target(self):
        self._target_word.next_target()
        self.update()

    def previous_target(self):
        self._target_word.previous_target()
        self.update()

    def eeg_letter(self, text, symbol, update_word=True):
        self._trigger(marker.FEEDBACK_START, wait=True)
        self.present_center_word(symbol, self._feedback_time)
        self._trigger(marker.FEEDBACK_END, wait=True)
        if update_word:
            self._target_word.set(text=text, target=len(text)-1)
