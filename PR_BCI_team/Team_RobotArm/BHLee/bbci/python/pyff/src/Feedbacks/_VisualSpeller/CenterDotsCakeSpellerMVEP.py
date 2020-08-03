# mvep_speller.py - mVEP Speller feedback
# Copyright (C) 2010  Mirko Dietrich
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

from math import sin, cos, pi

from VisualSpellerVE import VisualSpellerVE
import pygame
from lib.P300Aux.P300Functions import random_flash_sequence
from VEShapes import HexagonOpening, DotField
from CakeSpellerVE import CakeSpellerVE

class CenterDotsCakeSpellerMVEP(CakeSpellerVE):
    """
    Visual speller as a cake with six pieces with a moving dot field
    in the center.

    """

    def init(self):
        CakeSpellerVE.init(self)

        self.countdown_color = (0.8, 0.8, 0.8)
        self.shape_color = (0.5, 0.5, 0.5)
        self.edge_color = (0.8, 0.8, 0.8)
        self.intertrial_duration = 0.5 # sec
        self.interstimulus_duration = 0.1 # sec

        ## motion onset stimulus parameters
        self.stim_radius    = self.speller_radius * 0.2
        self.dot_speed      = 0.03
        self.dot_size       = 10.0
        self.dot_distance   = 18.0     # distance between dots

        # feedback mode
        #  0 = motion and color
        #  1 = motion-only
        #  2 = color-only
        self.feedback_mode = 0

        self.stimulus_duration = 0.3

        # SOA:stimulus onset asynchrony (in seconds)
        # how fine the animation should be resolved temporally
        self.soa_steps = 10

    def init_screen_elements(self):
        CakeSpellerVE.init_screen_elements(self)
        # Hexagon with opening
        self._ve_hexagon = \
            HexagonOpening(color=self.shape_color, edge_color=self.edge_color,
                           position=(self._centerPos[0], self._centerPos[1]),
                           radius=self.speller_radius,
                           opening_radius=self.stim_radius)

        # Dot field stimulus
        self._ve_dots = \
            DotField(on=False,
                     center=(self._centerPos[0], self._centerPos[1]),
                     width=self.stim_radius * 2., height=self.stim_radius * 2.,
                     dot_size=self.dot_size, dot_distance=self.dot_distance,
                     anti_aliasing=True)
        a = self.speller_radius * self.dot_speed
        self._dots_end_pos = [(
                self._centerPos[0] + a * cos((i + .5) / 6. * 2. * pi),
                self._centerPos[1] + a * sin((i + .5) / 6. * 2. * pi)
            ) for i in xrange(6)]

        # remove other hexagon
        for e in self._ve_edges:
            self._ve_elements.remove(e)
        for e in self._ve_shapes:
            self._ve_elements.remove(e)

        # add hexagon with opening
        self._ve_elements.insert(0, self._ve_hexagon)
        # put stimulus behind hexagon
        self._ve_elements.insert(0, self._ve_dots)

        # change color of countdown
        self._ve_countdown.set(color=self.countdown_color)

    def set_standard_screen(self, std=True):
        """
        set screen elements to standard state.

        """
        is_level1 = self._current_level == 1
        for i in xrange(self._nr_elements):
            self._ve_letters[self._nr_letters + i].set(
                on=not is_level1, color=self.stimuli_colors[i % len(self.stimuli_colors)])
        for i, color in enumerate(self.stimuli_colors):
            for j in xrange(5):
                self._ve_letters[i * 5 + j].set(on=is_level1, color=color)

    def stimulus(self, i_element, on=True):
        """
        turn on/off the stimulus elements and turn off/on the normal elements.

        """
        pass

    def play_tick(self):
        """
        called every frame, if in play mode.
        """
        if self._state_trial:
            self.__trial()
        else:
            CakeSpellerVE.play_tick(self)

    def __trial(self):
        # generate random sequences:
        if self._current_sequence==0 and self._current_stimulus==0:
            self.flash_sequence = []
            for _ in range(self.nr_sequences):
                random_flash_sequence(self,
                                      set=range(self._nr_elements),
                                      min_dist=self.min_dist,
                                      seq_len=self._nr_elements)

        currentStimulus = self.flash_sequence[self._current_sequence*self._nr_elements + self._current_stimulus]
        # set stimulus:
        self.stimulus(currentStimulus, True)
        self._ve_oscillator.set(on=True)

        # send trigger:
        if ((self.offline and self._current_level==1 and currentStimulus==self._classified_element) or
            (self.offline and self._current_level==2 and currentStimulus==self._classified_letter)):
            self.send_parallel(self.STIMULUS[self._current_level-1][currentStimulus] + self.TARGET_ADD)
            self.logger.info("[TRIGGER] %d" % (self.STIMULUS[self._current_level-1][currentStimulus] + self.TARGET_ADD))
        else:
            self.send_parallel(self.STIMULUS[self._current_level-1][currentStimulus])
            self.logger.info("[TRIGGER] %d" % self.STIMULUS[self._current_level-1][currentStimulus])

        ##################################
        # Present stimulus:

        # transform stimulus index to hexagon way of counting
        if currentStimulus == 0:
            idx = 1
        elif currentStimulus == 1:
            idx = 0
        elif currentStimulus == 2:
            idx = 5
        elif currentStimulus == 3:
            idx = 4
        elif currentStimulus == 4:
            idx = 3
        elif currentStimulus == 5:
            idx = 2
        #self._ve_oscillator.set(on=bool((self._current_stimulus + 1) % 2))
        s = self.soa_steps
        self._ve_dots.set(on=True)
        if self.feedback_mode != 1:
            self._ve_dots.set(color=self.stimuli_colors[currentStimulus])
        else:
            self._ve_dots.set(color=self.shape_color)
        self._presentation.set(
            go_duration=(self.stimulus_duration / s, 'seconds'))
        if self.feedback_mode != 2:
            self._ve_dots.set(
                orientation= (idx - 1) * 60)
        end_pos_x = self._dots_end_pos[idx][0]
        end_pos_y = self._dots_end_pos[idx][1]
        for k in xrange(s):
            progr = float(k) / (float(s) - 1)
            pos_x = (1. - progr) * self._centerPos[0] + progr * end_pos_x
            pos_y = (1. - progr) * self._centerPos[1] + progr * end_pos_y
            if self.feedback_mode != 2:
                self._ve_dots.set(center=(pos_x, pos_y))
            self._presentation.go()
            if (self.interstimulus_duration == 0.0):
                self._ve_oscillator.set(on=False)

        self._ve_dots.set(on=False)
        # stimulus presentation finished
        ##################################

        # reset to normal:
        self._ve_oscillator.set(on=False)
        self.stimulus(currentStimulus, False)

        # present interstimulus:
        self._presentation.set(go_duration=(self.interstimulus_duration, 'seconds'))
        self._presentation.go()

        if self.debug:
            self.on_control_event({'cl_output':(self.random.random(), currentStimulus+1)})


        # increase
        self._current_stimulus = (self._current_stimulus+1) % self._nr_elements
        if self._current_stimulus == 0:
            self._current_sequence = (self._current_sequence+1) % self.nr_sequences

        # check for end of trial:
        if self._current_sequence == 0 and self._current_stimulus == 0:

            # send trigger:
            if self._current_level==1:
                self.send_parallel(self.END_LEVEL1)
                self.logger.info("[TRIGGER] %d" % self.END_LEVEL1)
            else:
                self.send_parallel(self.END_LEVEL2)
                self.logger.info("[TRIGGER] %d" % self.END_LEVEL2)

            # decide how to continue:
            self._state_trial = False
            self._state_classify = True

if __name__ == '__main__':
    fb = CenterDotsCakeSpellerMVEP()
    fb.on_init()
    fb.on_play()
