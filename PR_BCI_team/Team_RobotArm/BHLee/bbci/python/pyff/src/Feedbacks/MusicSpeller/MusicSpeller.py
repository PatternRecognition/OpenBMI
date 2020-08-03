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

import os, copy, shutil, time, logging

import pygame
from pygame import mixer
from scikits.audiolab import wavread, wavwrite
import numpy

from VisionEgg.Core import Viewport
import VisionEgg.GL as gl

from lib import marker
from lib.speller import Speller
from lib.vision_egg import VisionEggView
from lib.vision_egg.model.target_word import TargetWord
from lib.vision_egg.model.stimulus import TextureStimulus
from FeedbackBase.VisionEggFeedback import VisionEggFeedback

_trigger_base = 20

class Instrument(object):
    def __init__(self, name, original, variant, image):
        self.name = name
        self.original = original
        self.variant = variant
        self.image = image

class TargetPictureWord(TargetWord):
    def add(self, path, height):
        stim = TextureStimulus(anchor='bottom', internal_format=gl.GL_RGBA)
        stim.set_file(path)
        stim.set_height(height)
        self.append(stim)
        self._rearrange()

    def _set_target(self, index):
        self._target_index = index
        self.set(target=index)

class MusicSpellerView(VisionEggView):
    def set_instruments(self, instruments):
        self._instruments_ordered = instruments
        self._instruments = dict(((i.name, i) for i in instruments))

    def init(self):
        self._setup_word()
        self._setup_feedback()

    def _setup_word(self):
        sz = self.screen.size
        target_height = self._word_vpos * 2
        viewport_height = sz[1] - target_height - self._word_margin
        self._target_word = TargetPictureWord(position=(sz[0] / 2,
                                                        self._word_vpos),
                                              center_at_target=True,
                                              symbol_size=self._pic_height,
                                              target_size=self._target_height,
                                              spacing=self._word_spacing)
        self._target_word_viewport = Viewport(screen=self.screen,
                                              stimuli=self._target_word,
                                              size=(sz[0], target_height),
                                              position=(0, viewport_height))
        self.add_viewport(self._target_word_viewport)

    def _setup_feedback(self):
        sz = self.screen.size
        self._center_pic = self.add_image_stimulus(position=(sz[0] / 2.,
                                                             sz[1] / 2.),
                                                   internal_format=gl.GL_RGBA)
        self._center_pic.set()
        self._center_pic.hide()

    def word(self, word):
        imgs = [self._instruments[name].image for name in word]
        self._target_word.set_word(imgs)
        self.update()

    def next_target(self):
        self._target_word.next_target()
        self.update()

    def previous_target(self):
        self._target_word.previous_target()
        self.update()

    def eeg_letter(self, text, symbol, update_word=True):
        self._trigger(marker.FEEDBACK_START, wait=True)
        self._present_feedback(symbol, self._feedback_time)
        self._trigger(marker.FEEDBACK_END, wait=True)
        if update_word:
            self._target_word.set(text=text, target=len(text)-1)

    def _present_feedback(self, num, seconds, color=None):
        num = int(num)
        if num:
            self._center_pic.set_file(self._instruments_ordered[num-1].image)
            self._center_pic.set_height(self._feedback_height)
            self._center_pic.show()
            self.present(seconds)
            self._center_pic.hide()
        else:
            self.present_center_word('X', seconds)

class MusicSpellerInternal(Speller, VisionEggFeedback):
    """ Auditory speller feedback.
    Stimuli are mixes of music loops, where each sample has a
    corresponding variant version that the test subject is supposed to
    identify.
    The feedback generates mixtures of all samples, one for each variant
    stimulus plus one with all the original samples.
    Mixes are then played back in random order, with one of them being
    the target stimulus.
    The test subject is presented with a picture of the target
    instrument, the classifier should then identify the subject's
    recognition of the loop with the corresponding variant stimulus.
    """
    def __init__(self, *a, **kw):
        Speller.__init__(self)
        VisionEggFeedback.__init__(self, view_type=MusicSpellerView, *a, **kw)

    def init(self):
        self._setup_mixer()
        super(MusicSpellerInternal, self).init()

    def update_parameters(self):
        self._clean_temp_dir()
        self._load_samples()
        self._create_sample_mixes()
        self._create_sounds()
        super(MusicSpellerInternal, self).update_parameters()

    def _setup_mixer(self):
        """ Prepare pygame.mixer.
        _channel represents the mixer channel in use by the feedback.
        _sound_end_event is a number identifying the event sent by
        pygame once playback ends on the channel.
        """
        mixer.init()
        self._channel = mixer.Channel(0)
        self._sound_end_event = pygame.locals.USEREVENT
        self._channel.set_endevent(self._sound_end_event)
        self._view.add_event_handlers([(self._sound_end_event, self.sound_end)])

    def _clean_temp_dir(self):
        """ Delete temporary files of the feedback's last run.
        """
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        os.mkdir(self.temp_dir)

    def _load_samples(self):
        """ Read all files in the trial directory and separate the
        variant stimuli identified by a '_var' suffix.
        Use scikits.audiolab.wavread to create numpy arrays of the
        samples, store these in _orig_samples and _var_samples.
        """
        from os.path import join, basename, splitext
        def list_subdir(d):
            sub = join(self.sample_dir, self.trial_name, d)
            return [join(sub, f) for f in sorted(os.listdir(sub))]
        orig, var, img = map(list_subdir, ('original', 'variant', 'image'))
        names = [basename(splitext(n)[0]) for n in img]
        ig, self._fs, self._enc = wavread(orig[0])
        wav = lambda fname: wavread(fname)[0]
        self._instruments = map(Instrument, names, map(wav, orig),
                                map(wav, var), img)
        self._sample_count = len(self._instruments)
        self._mix_count = self._sample_count + 1
        self._view.set_instruments(self._instruments)
        self.symbols = map(str, range(self._mix_count))

    def _create_sample_mixes(self):
        """ Create mixtures of the samples.
        First mix consists of all non-variant samples, then vary each
        sample and mix with the remaining non-variants.
        If lengths are inequal, fill with zeros.
        """
        def _zero_pad(arrays):
            """ Fill all arrays with zeros to reach the length of the
            longest sample.
            """
            maxlen = max(map(len, arrays))
            for a in arrays:
                yield numpy.hstack((a, numpy.zeros(maxlen - len(a))))
        def _mix(samples):
            """ Sum up all the samples and normalize the result to
            attain a mixture.
            """
            norm_factor = 1. / len(samples)
            samples = _zero_pad(samples)
            return norm_factor * numpy.vstack(samples).sum(0)
        original = [i.original for i in self._instruments]
        variant = [i.variant for i in self._instruments]
        self._sample_mixes = [_mix(original)]
        for index in xrange(self._sample_count):
            samples = copy.copy(original)
            samples[index] = variant[index]
            self._sample_mixes.append(_mix(samples))

    def _create_sounds(self):
        """ Write the mixtures to files and load them as pygame Sound
        objects.
        """
        self._sounds = []
        for index, sample in enumerate(self._sample_mixes):
            path = os.path.join(self.temp_dir, 'mix_{0}.wav'.format(index))
            wavwrite(sample, path, self._fs, self._enc)
            self._sounds.append(mixer.Sound(path))

    def sound_end(self, event):
        self._sound_playing = False

class MusicSpeller(MusicSpellerInternal):
    def init_parameters(self):
        """ The feedback's directory is used as basedir for temporary
        and data directories. The parameter 'trial_name' is used as the
        subdirectory of 'data' from where the samples are read.
        """
        module_dir = os.path.dirname(__file__)
        self.temp_dir = os.path.join(module_dir, 'tmp')
        self.sample_dir = os.path.join(module_dir, 'data')
        self.trial_name = 'std'
        self.trial_count = 1
        self.blocks_per_trial = 1
        self.phrases = [['drum', 'vocal', 'synth', 'drum']]
        self.phrase_countdown = False
        self.feedback_time = 2.
        self.pic_height = 60
        self.target_height = 90
        self.feedback_height = 150
        self.word_spacing = 10
        self.word_margin = 10
        self.word_vpos = 50
        self.bg_color = 'olivedrab'
        self._view_parameters += ['feedback_time', 'pic_height',
                                  'target_height', 'feedback_height',
                                  'word_spacing', 'word_margin', 'word_vpos']

    @Speller.sequences
    def _permutations(self):
        """ Create index permutations for the blocks. """
        return [numpy.random.permutation(self._mix_count) for i in
                xrange(self.blocks_per_trial)]

    @Speller.stimulus
    def _play_block_samples(self, samples):
        """ Iterate the block index permutation, send the corresponding
        marker, play the sound and wait for pygame's end event to
        arrive.
        """
        def gen():
            for classno in samples:
                self._sound_playing = True
                self._trigger(_trigger_base+classno)
                self._channel.play(self._sounds[classno])
                while self._sound_playing:
                    yield
        self.stimulus_sequence(gen(), 0.01).run()

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    speller = MusicSpeller()
    speller.on_init()
    speller.on_play()
