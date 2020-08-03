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

from FeedbackBase.VisionEggFeedback import VisionEggFeedback

class VEExample1(VisionEggFeedback):
    """ This example works the following way:
        The prepare generator function is passed to the framework,
        which starts the stimulus presentation every time a yield
        statement in the generator is encountered. When the
        presentation time is over, the next word in prepare() is set.
        As soon as the loop in prepare() is exhausted, the run()
        function of the presentation handler returns.
    """
    def run(self):
        # Add a text object in about the center
        self.word = self.add_text_stimulus(font_size=72, position=(300, 200))
        # Add a picture above
        self.image = self.add_image_stimulus(position=(300, 300))
        # This feedback uses a generator function for controlling the stimulus
        # transition. Note that the function has to be called before passing
        generator = self.prepare()
        # Pass the transition function and the per-stimulus display durations
        # to the stimulus sequence factory. As there are three stimuli, the
        # last one is displayed 5 seconds again.
        s = self.stimulus_sequence(generator, [5., 1.])
        # Start the stimulus sequence
        s.run()

    def prepare(self):
        """ This is a generator function. It is the same as a loop, but
        execution is always suspended at a yield statement. The argument
        of yield acts as the next iterator element. In this case, none
        is needed, as we only want to prepare the next stimulus and use
        yield to signal that we are finished.
        """
        self.image.set_file('ida.png')
        for w in ['BBCI', 'Vision', 'Egg']:
            # Change the text to display the next stimulus
            self.word.set(text=w)
            # and signal that we are done with the next stimulus and
            # that the waiting period can begin
            yield

if __name__ == '__main__':
    e = VEExample1()
    e.on_init()
    e.on_play()
