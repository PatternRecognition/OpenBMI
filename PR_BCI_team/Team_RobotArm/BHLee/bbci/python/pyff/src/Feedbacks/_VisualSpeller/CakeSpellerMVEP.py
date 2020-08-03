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
from VisionEgg.MoreStimuli import Target2D
from lib.P300Aux.P300Functions import random_flash_sequence

#from VisualSpellerVE import VisualSpellerVE
from CakeSpellerVE import CakeSpellerVE
import pygame

class CakeSpellerMVEP(CakeSpellerVE):
    """
    Visual speller as a cake with six pieces using mVEP.

    """

    def init(self):
        CakeSpellerVE.init(self)

        self.countdown_color = (0.8, 0.8, 0.8)
        self.intertrial_duration = 0.5 # sec
        self.interstimulus_duration = 0.0 # sec

        ## motion onset stimulus parameters
        # start/end relative to speller radius
        self.onset_bar_start = 0.15
        self.onset_bar_end = 0.3
        self.onset_bar_width = 5.0
        self.onset_color = (0.4, 0.4, 0.4)
        # stimulus duration (in sec)
        self.stimulus_duration = 0.1
        # how fine the animation should be resolved temporally
        self.soa_steps = 6
        # enlarge motion-onset bar while moving
        self.enlarge_bar = True

        ################################
        # Alternative layout settings
        ################################
        # if true letters are positioned along the hexagon outlines
        self.alt_letter_layout = False 
        self.alt_speller_radius = 320
        self.alt_font_size_level2 = 70

        # if set to true each triangle will have one fixation point
        self.alt_fix_points = False
        # distance from middle (proportional to radius)
        self.alt_fix_points_dist = 0.5
        # disable fix points at inter-trial period
        self.alt_fix_points_disable_inter = False
        self.synchronized_countdown=False

    def init_screen_elements(self):
        # adjust sizes for alt layout
        if self.alt_letter_layout:
            self.speller_radius = self.alt_speller_radius
            self.font_size_level2 = self.alt_font_size_level2


        CakeSpellerVE.init_screen_elements(self)

        # create motion-onset bar
        a = self.speller_radius * self.onset_bar_start
        self._onset_bar_start_pos = [(
                self._centerPos[0] + a * cos((i + .5) / 6. * 2. * pi),
                self._centerPos[1] + a * sin((i + .5) / 6. * 2. * pi)
            ) for i in xrange(6)]
        a = self.speller_radius * self.onset_bar_end
        self._onset_bar_end_pos = [(
                self._centerPos[0] + a * cos((i + .5) / 6. * 2. * pi),
                self._centerPos[1] + a * sin((i + .5) / 6. * 2. * pi)
            ) for i in xrange(6)]
        self._ve_onset_bar = \
            Target2D(size=(self.speller_radius * 0.1, self.onset_bar_width),
                     color=self.onset_color,
                     on=False)
        self._ve_elements.append(self._ve_onset_bar)



        # replace letters by alternative layout if requested
        if self.alt_letter_layout:
            # level 1
            a = self.speller_radius
            k = 0
            for i in xrange(self._nr_elements - 1, -1, -1):
                pos = (
                    self._centerPos[0] + a * cos((i + 2.5) / 6. * 2. * pi),
                    self._centerPos[1] + a * sin((i + 2.5) / 6. * 2. * pi))
                self._shape_positions[i] = pos
                for j in xrange(len(self.letter_set[i])):
                    if i in (2, 5):
                        self._letter_positions[k] = (
                            pos[0] + j * 60 - 115, pos[1])
                    elif i in (1, 4):
                        self._letter_positions[k] = (
                            pos[0] - j * 30 + 60, pos[1] + j * 55 - 110)
                    else:
                        self._letter_positions[k] = (
                            pos[0] + j * 30 - 60, pos[1] + j * 55 - 110)
                    self._ve_letters[k].set(position=self._letter_positions[k])
                    k += 1
            # level 2
            for i in xrange(self._nr_elements - 1, -1, -1):
                pos = (
                    self._centerPos[0] + a * cos((i + 2.5) / 6. * 2. * pi),
                    self._centerPos[1] + a * sin((i + 2.5) / 6. * 2. * pi))
                self._ve_letters[k].set(position=pos)
                k += 1
            self._shape_positions.reverse()

        # add alternative fixation points
        if self.alt_fix_points:
            self._ve_elements.remove(self._ve_fixationpoint)
            a = self.speller_radius * self.alt_fix_points_dist
            self._ve_alt_fix_points = []
            for i in xrange(self._nr_elements):
                pos = (
                    self._centerPos[0] + a * cos((i + .5) / 6. * 2. * pi),
                    self._centerPos[1] + a * sin((i + .5) / 6. * 2. * pi))
                self._ve_alt_fix_points.append(Target2D(
                        size=(self.fixationpoint_size, self.fixationpoint_size),
                        position=pos,
                        color=self.fixationpoint_color))
                self._ve_elements.append(self._ve_alt_fix_points[-1])


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

    def fixation(self, state=True):
        """
        turn on/off the fixation elements.

        """
        if self.alt_fix_points and self.alt_fix_points_disable_inter:
            for p in self._ve_alt_fix_points:
                p.set(on=state)
        else:
            self._ve_fixationpoint.set(on=state)

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
       
        # check if current stimulus is target and then send trigger:
        target_add = 0
        if self._current_level==1:
            if self._desired_letters[:1] in self.letter_set[currentStimulus]:
                # current stimulus is target group:
                target_add = self.TARGET_ADD
        else:
            if currentStimulus==self._idx_backdoor:
                # current stimulus is backdoor:
                if not self._desired_letters[:1] in self.letter_set[self._classified_element]:
                    # we are in the wrong group. backdoor is target:
                    target_add = self.TARGET_ADD                
            else:
                # current stimulus is no backdoor:
                if self._desired_letters[:1]==self.letter_set[self._classified_element][currentStimulus]:
                    # current stimulus is target symbol:
                    target_add = self.TARGET_ADD

        self.send_parallel(self.STIMULUS[self._current_level-1][currentStimulus] + target_add)
        self.logger.info("[TRIGGER] %d" % (self.STIMULUS[self._current_level-1][currentStimulus] + target_add))
        
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
        s = self.soa_steps
        self._ve_onset_bar.set(on=True, color=self.stimuli_colors[currentStimulus])
        self._presentation.set(
            go_duration=(self.stimulus_duration / s, 'seconds'))
        self._ve_onset_bar.set(
            orientation=(idx - 1) * 60)
        start_pos_x = self._onset_bar_start_pos[idx][0]
        start_pos_y = self._onset_bar_start_pos[idx][1]
        end_pos_x = self._onset_bar_end_pos[idx][0]
        end_pos_y = self._onset_bar_end_pos[idx][1]
        for k in xrange(s):
            progr = float(k) / (float(s) - 1)
            pos_x = (1. - progr) * start_pos_x + progr * end_pos_x
            pos_y = (1. - progr) * start_pos_y + progr * end_pos_y
            if self.enlarge_bar:
                self._ve_onset_bar.set(
                    size=((1. - progr) *
                          self.speller_radius * self.onset_bar_start
                          + progr *
                          self.speller_radius * self.onset_bar_end,
                          self.onset_bar_width))
            self._ve_onset_bar.set(position=(pos_x, pos_y))
            self._presentation.go()
            if (self.interstimulus_duration == 0.0):
                self._ve_oscillator.set(on=False)

        self._ve_onset_bar.set(on=False)
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

    def post__countdown(self):
        pass
        #pygame.time.wait(100) # wait to prevent VisionEgg Texture error(?)

if __name__ == '__main__':
    fb = CakeSpellerMVEP()
    fb.on_init()
    fb.on_play()
