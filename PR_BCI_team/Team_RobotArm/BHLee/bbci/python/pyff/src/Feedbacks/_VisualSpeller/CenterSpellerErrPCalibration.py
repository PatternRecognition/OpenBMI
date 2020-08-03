
'''Feedbacks.VisualSpeller.CenterSpellerErrPCalibration
# Copyright (C) 2010  "Nico Schmidt"
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

Created on Jul 21, 2010

@author: "Nico Schmidt"
'''
from CenterSpellerVE import CenterSpellerVE
from VisionEgg.MoreStimuli import Arrow
#from VisionEgg.FlowControl import FunctionController

import numpy as NP
import random, pygame

class CenterSpellerErrPCalibration(CenterSpellerVE):
    '''
    classdocs
    '''
    ARROW_ONSET = 5
    USER_RESPONSE = 6
    MACHINE_ERROR = 90
    USER_ERROR = 91
    
    def init(self):
        '''
        initialize parameters
        '''
        CenterSpellerVE.init(self)
        self.wait_between_triggers = 50
        self.error_rate = 0.15
        self.error_mindist = 1
        self.len_error_sequence = 100
        self.preceding_nonerrors = 4
        self.arrow_color = (0.6,0.6,0.6)
        self.arrow_size = (30.,10.)
        self.arrow_round_time = 1.5
        self.arrow_precision = 10 # in degrees
        self.arrow_start_orientation = -90
        self.stop_after_response = True
        self.stop_after_rounds = 2
        self.arrow_random_start = True
        self.arrow_reset_after_trial = False
        self.one_level_only = False
        self.one_level_letters = ['A','E','I','D','N','<']
        self.choose_second_at_error = False
        
        self.offline = False
        self.debug = True
        self.nCountdown = 2
        self.animation_time = 0.5
        self.wait_before_classify = 1.0
        self.feedback_show_shape = True
        self.feedback_show_shape_at_center = True
        self.desired_phrase = "THE_MARCH_HARE_AND_THE_HATTER_WERE_HAVING_TEA."
        self.use_ErrP_detection = False
    
    def prepare_mainloop(self):
        # one-level condition:
        if self.one_level_only:
            self._current_level = 2
            self._classified_element = 0
            self._idx_backdoor = -1
            self.letter_set[0] = self.one_level_letters
            self._nr_letters += 1
        # superclass-method:
        CenterSpellerVE.prepare_mainloop(self)
        
        assert(360 % self.arrow_precision == 0) # precision should be a factor of 360
        self._arrow_time_steps = self.arrow_precision * self.arrow_round_time / 360.
        self._current_arrow_orientation = self.arrow_start_orientation
        self._response_orientation = None
        self.nr_sequences = 1
        self._start_trial = True
        if not self.stop_after_response:
            self.arrow_reset_after_trial = True
        self._iTrial = 0
        self._iMachineError = 0
        self._iUserError = 0
        
        ## build error sequence:
        n = int(round(self.error_rate * self.len_error_sequence/2 * self.error_mindist))
        error_sequence_l1 = [i%self.error_mindist==0 and 1 or 0 for i in xrange(n)]
        error_sequence_l2 = error_sequence_l1[:]
        for _ in xrange(self.len_error_sequence/2 - n):
            error_sequence_l1.insert(random.randint(0,len(error_sequence_l1)-1),0)
            error_sequence_l2.insert(random.randint(0,len(error_sequence_l2)-1),0)
        self._error_sequence = []
        for i in xrange(self.len_error_sequence/2):
            self._error_sequence.extend([error_sequence_l1[i], error_sequence_l2[i]])
    
    def init_screen_elements(self):
        self._ve_arrow = Arrow(color=self.arrow_color,
                               position=self._centerPos,
                               size=self.arrow_size,
                               orientation=self.arrow_start_orientation,
                               on=False)
        self._ve_elements.append(self._ve_arrow)
        CenterSpellerVE.init_screen_elements(self)
        
    
    def set_standard_screen(self):
        self._countdown_screen = False
        self._ve_arrow.set(on=True)
        
        is_level1 = self._current_level==1
        
        ## turn on visible elements:
        for i in xrange(self._nr_elements): # shapes
            self._ve_shapes[i].set(on=(is_level1 or self.level_2_symbols), position=self._countdown_shape_positions[i])
        for i in xrange(self._nr_letters): # level 1 letters
            self._ve_letters[i].set(on=is_level1, position=self._countdown_letter_positions[i])
        for i in xrange(len(self.letter_set[self._classified_element])): # level 2 letters
            self._ve_letters[self._nr_letters + i].set(on=not is_level1,
                                                       position=self._countdown_shape_positions[i],
                                                       text=(is_level1 and " " or self.letter_set[self._classified_element][i]))
    
    def play_tick(self):
        """
        called every frame, if in play mode.
        """
        if self._state_trial:
            self.__trial()
        else:            
            CenterSpellerVE.play_tick(self)
                
    def post__countdown(self):
        self._tic_fb = 0
        self._ve_fixationpoint.set(on=True)
        
    def __trial(self):
        '''
        called from FeedbackController
        '''
        if self._start_trial:
            self._start_trial = False
            self._response_orientation = None # prevent user from pressing key before trial starts
            self.send_parallel(self.ARROW_ONSET)
            self.logger.info("[TRIGGER] %d" % self.ARROW_ONSET)
            
        self._current_arrow_orientation += self.arrow_precision
        self._ve_arrow.set(orientation=self._current_arrow_orientation)
        self._presentation.set(go_duration=(self._arrow_time_steps, 'seconds'))
        self._presentation.go()

        # check if we already got a user response:
        if self._response_orientation == None:
            return # continue one more rotation
        
        # check for end of trial:
        if self.stop_after_response or self._current_arrow_orientation >= self.arrow_start_orientation + self.stop_after_rounds*360:
        
            ## turn off all elements and letters:
            for i in xrange(self._nr_elements):
                self._ve_shapes[i].set(on=False)
            for i in xrange(len(self._ve_letters)):
                self._ve_letters[i].set(on=False)
            self._ve_arrow.set(on=False)
            self._presentation.set(go_duration=(0.001, 'seconds'))
            self._presentation.go()
            
            if self.arrow_random_start:
                self._current_arrow_orientation = random.randint(0,360)                
            elif self.arrow_reset_after_trial:
                self._current_arrow_orientation = self.arrow_start_orientation
            
            self._state_trial = False
            self._state_classify = True
        
    def pre__classify(self):
        pygame.time.wait(self.wait_between_triggers) # wait to prevent trigger overlap (USER_RESPONSE and *ERROR)
        
        # use ditance from arrow to elements as classifier output:
        orientations = NP.array([270, 330, 30, 90, 150, 210])
        dist = abs(orientations - self._response_orientation%360)
        dist[dist>180] = 360 - dist[dist>180]
        for i in xrange(self._nr_elements):
            for _ in xrange(self.nr_sequences):
                self.on_control_event({'cl_output':(dist[i],i+1)})
        
        ## USER ERROR?
        user_error = False
        if len(self._desired_letters) > 0:
            selected = NP.argmin(dist)
            if self._current_level == 1:
                if not self._desired_letters[:1] in self.letter_set[selected]:
                    # wrong group selected:
                    user_error = True
            else:
                if selected == self._idx_backdoor:
                    ## backdoor selected:
                    if self._desired_letters[:1] in self.letter_set[self._classified_element]:
                        # backdoor selection wrong:
                        user_error = True
                else:
                    ## no backdoor selected:
                    if self._desired_letters[:1] != self.letter_set[self._classified_element][selected]:
                        # wrong letter selected:
                        user_error = True
                        
        if user_error: # send user_error trigger:
            self._iUserError += 1 
            print "*** User Errors: ", self._iUserError
            self.send_parallel(self.USER_ERROR)
            self.logger.info("[TRIGGER] %d" % self.USER_ERROR)
        
        ## MACHINE ERROR?
        if self._error_sequence[(self._iTrial-self.preceding_nonerrors)%len(self._error_sequence)] and self._iTrial>=self.preceding_nonerrors:
            if self.choose_second_at_error:
                # choose element with 2nd-minimum distancee from arrow:
                idx = NP.argmin(dist)
                val = NP.max(dist)
            else:
                # choose one of the 3 elements with max distance from arrow:
                val = NP.min(dist)-1
                d = []
                d.append(NP.argmax(dist))
                dist[d[-1]] = -1
                d.append(NP.argmax(dist))
                dist[d[-1]] = -1
                d.append(NP.argmax(dist))
                idx = d[random.randint(0,2)]
            # reset the classifier_output for the best one and fill it with high values:
            self._classifier_output[idx] = list() # reset classifier output
            for _ in xrange(self.nr_sequences): # fill with maxvalues:
                self.on_control_event({'cl_output':(val,idx+1)})
            if user_error:
                pygame.time.wait(self.wait_between_triggers)
            self.send_parallel(self.MACHINE_ERROR)
            self.logger.info("[TRIGGER] %d" % self.MACHINE_ERROR)
            self._iMachineError += 1
            print "*** Machine Errors: ", self._iMachineError
        
        self._iTrial += 1
        # reset user response:
        self._response_orientation = None
        
        # turn off spelled and desired phrases:
        self._ve_spelled_phrase.set(on=False)
        self._ve_current_letter.set(on=False)
        self._ve_desired_letters.set(on=False)
        self._ve_letterbox.set(on=False)
        
    def post__feedback(self):
        if self.one_level_only:
            self._current_level = 2
        self._state_countdown = False
        self._state_trial = True
        self.set_standard_screen()
        self._ve_fixationpoint.set(on=True)     
        

    def keyboard_input(self, event):
        if event.key == pygame.K_ESCAPE:
            self.on_stop()
        elif not self.offline and event.key == pygame.K_SPACE:
            self.on_control_event({'stop_arrow':None})
            
            
    def on_control_event(self, data):
        if data.has_key(u'stop_arrow'):
            if self._response_orientation == None:
                self.send_parallel(self.USER_RESPONSE)
                self.logger.info("[TRIGGER] %d" % self.USER_RESPONSE)
                self._response_orientation = self._current_arrow_orientation
        CenterSpellerVE.on_control_event(self, data)
            

if __name__ == '__main__':
    fb = CenterSpellerErrPCalibration()
    fb.on_init()
    fb.on_play()
