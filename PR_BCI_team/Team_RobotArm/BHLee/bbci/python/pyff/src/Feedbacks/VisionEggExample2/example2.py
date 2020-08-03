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
from VisionEgg.Gratings import SinGrating2D 

from FeedbackBase.VisionEggFeedback import VisionEggFeedback

class VEExample2(VisionEggFeedback):
    """ In this example, the prepare() function is passed to the
    framework. The function is called before each stimulus transition,
    presentation is interrupted as soon as False is returned (or, in
    this case, None).
    """
    def init(self):
        # Setup our resources
        self.words = ['BBCI', 'Vision', 'Egg']

    def run(self):
        # Reset status tracking variables
        self.current_word = 0
        # Our stimuli. This can be anything compatible with VisionEgg
        self.word = Text(font_size=72, position=(320, 200), anchor='center')
        self.grating = SinGrating2D(anchor='lowerleft', position=(220, 240),
                                    size=(200, 100))
        self.set_stimuli(self.word, self.grating)
        # Pass the transition function and the per-stimulus display duration
        # to the stimulus sequence factory
        # the first two arguments are intervals for a random value
        s = self.stimulus_sequence(self.prepare, [[1., 3.], [3., 5.], 1.])
        # Start the stimulus sequence
        s.run()

    def prepare(self):
        if self.current_word < len(self.words):
            w = self.words[self.current_word]
            self.word.set(text=w)
            self.current_word += 1
            return True

if __name__ == '__main__':
    e = VEExample2()
    e.on_init()
    e.on_play()
