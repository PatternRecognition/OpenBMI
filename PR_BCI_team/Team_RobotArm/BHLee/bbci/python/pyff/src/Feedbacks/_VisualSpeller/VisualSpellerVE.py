'''Feedbacks.VisualSpeller.VisualSpellerVE
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

Created on Mar 23, 2010

@author: "Nico Schmidt"


Serves as base class for spellers such as CakeSpeller and CenterSpeller.
'''



from time import time, clock
from FeedbackBase.MainloopFeedback import MainloopFeedback
from lib.P300Aux.P300Functions import random_flash_sequence

from VisionEgg.Core import Screen
from VisionEgg.Core import Viewport
from VisionEgg.FlowControl import Presentation
from VisionEgg.MoreStimuli import Target2D, FilledCircle
from _VisualSpeller.VEShapes import  FilledHexagon 
from VisionEgg.Text import Text
from VisionEgg import logger

import numpy as NP
import random, pygame, os, math, pygame.sndarray
import logging

from sys import platform, maxint

from lib import marker 

class VisualSpellerVE(MainloopFeedback):
    '''
    Visual Speller with six circles like the classical HexOSpell.
    '''

    # Triggers: look in Marker 
    END_LEVEL1, END_LEVEL2 = 244,245               # end of hex levels
    COPYSPELLING_FINISHED = 246
    STIMULUS = [ [11, 12, 13, 14, 15, 16] , [21, 22, 23, 24, 25, 26] ]
    RESPONSE = [ [51, 52, 53, 54, 55, 56] , [61, 62, 63, 64, 65, 66] ]
    TARGET_ADD = 20
    ERROR_ADD = 100
    COUNTDOWN_STIMULI = 239 
    ERROR_POTENTIAL = 96 # send if error potential is classified

    def init(self):
        '''
        initialize parameters
        '''
        self.log_filename = 'VisualSpellerVE.log'

        self.geometry=[0,0,1280,800]         ## size

        self.letterbox_size = (60,60)
        self.osc_size = 40
        self.font_size_phrase = 60       # the spelled phrase at the top
        self.font_size_current_letter = 80       # the spelled phrase at the top
        self.font_size_countdown = 150  # number during countdown
        self.desired_phrase = ""
        
        ## colors:
        self.bg_color = (0., 0., 0.)
        self.phrase_color = (0.2, 0.0, 1.0)
        self.current_letter_color = (1.0, 0.0, 0.0)
        self.countdown_color = (0.2, 0.0, 1.0)
        self.osc_color = (1,1,1)
        
        self.letter_set = [['A','B','C','D','E'], \
                           ['F','G','H','I','J'], \
                           ['K','L','M','N','O'], \
                           ['P','Q','R','S','T'], \
                           ['U','V','W','X','Y'], \
                           ['Z','_','.',',','<']]
        self.fullscreen = False
        self.use_oscillator = True
        self.offline = True
        self.copy_spelling = True  # in copy-spelling mode, selection of the target symbol is forced
        self.debug = False
        self.nCountdown = 5
        self.nr_sequences = 6
        self.randomize_sequence = True # set to False to present a fixed stimulus sequence
        self.min_dist = 2 # Min number of intermediate flashes bef. a flash is repeated twice

 
        self.stimulus_duration = 0.083   # 5 frames @60 Hz = 83ms flash
        self.interstimulus_duration = 0.1
        self.animation_time = 1 
        self.wait_before_classify = 1.
        self.feedback_duration = 1.
        self.feedback_ErrP_duration = 1.0


        # Countdown options
        self.do_animation=True
        self.synchronized_countdown=True
        if(self.synchronized_countdown):
            self.do_animation=False


        self.countdown_level1=True
        self.countdown_level2 =True

        self.countdown_shapes = {'circle':FilledCircle,
	                              'hexagon':FilledHexagon}
        self.countdown_shape_select = 'hexagon'
        self.countdown_shape_color=(0.7,0.7,0.7)
        self.countdown_shape_on=True
        self.countdown_blinking_nr=5   #number of pre-sequence stimuli(1 sec is approx. 5 frames at 60 Hz)


        self.wait_after_early_stopping=3 #sec
        self.abort_trial=False
        self.output_per_stimulus=True
        self.use_ErrP_detection = False


        if self.debug:
            msg = "!!! YOU\'RE IN DEBUG MODE! CLASSIFICATION WILL BE RANDOM OR KEYBOARD CONTROLLED !!!"
            self.logger.warning(msg)

    
    def pre_mainloop(self):

        ## logging
        assert(len(self.log_filename)!=0) # 'log_filename' must not be empty string!
        logger.setLevel(logging.ERROR)
        handler = logging.FileHandler(self.log_filename, 'w')
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s: %(message)s")
        handler.setFormatter(formatter)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(handler)

        self._nr_elements = 6
        self._idx_backdoor = 5
        self._init_classifier_output()
        self._classified_element = -1   
        self._classified_letter = -1
        for s in self.desired_phrase:
            assert s in [l for ls in self.letter_set for l in ls] # invalid letters in desired phrase!
        self._spelled_phrase = ""
        self._spelled_letters = ""
        self._desired_letters = self.desired_phrase
        self._copyspelling_finished = False
                
                    
        self._spellerHeight = self.geometry[3] - self.letterbox_size[1]
        self._centerPos = (self.geometry[2]/2., self._spellerHeight/2.)
        
        self._nr_letters = 0
        for i in xrange(len(self.letter_set)):
            self._nr_letters += len(self.letter_set[i])
            
        self._current_level = 1          # Index of current level
        self._current_sequence = 0       # Index of current sequence
        self._current_stimulus = 0       # Index of current stimlus
        self._current_countdown = self.nCountdown
        self.random = random.Random(clock())
        self._debug_classified = None
        
        ## init states:
        if not self.countdown_level1:
            self._state_countdown= False
            self._state_trial=True
        else:
            self._state_countdown = not self.offline
            self._state_trial = False
        
        self._state_classify = False
        self._state_feedback = False
        self._state_abort = False
        
        ## init containers for VE elements:
        self._ve_elements = []
        

        ## oscillator state:
        if not self.use_oscillator:
            self.osc_color = self.bg_color
            self.osc_size = 0
        
        ## call subclass-specific pre_mainloop:
        self.prepare_mainloop()

        ## build screen elements:
        self.__init_screen()

        if self.abort_trial:
            '''
            Start listener for abort_trial event eg. 
            '''

        ## send start trigger:
        self.send_parallel(marker.RUN_START)
        self.logger.info("[TRIGGER] %d" % marker.RUN_START)
        
        ## error potential classifier:
        self._ErrP_classifier = None

    def post_mainloop(self):
        """
        Sends end marker to parallel port.
        """

        if self.abort_trial: 
           '''
            Stop listener for abort_trial event
           '''
           pass
          
        pygame.time.wait(500)
        self.send_parallel(marker.RUN_END)
        self.logger.info("[TRIGGER] %d" % marker.RUN_END)
        pygame.time.wait(500)
        self._presentation.set(quit=True)
        self._screen.close()
    
    
    def __init_screen(self):
        ## create screen:
        if not self.fullscreen:
            os.environ['SDL_VIDEO_WINDOW_POS'] = '%d, %d' % (self.geometry[0], self.geometry[1])
        self._screen = Screen(size=(self.geometry[2], self.geometry[3]),
                              fullscreen=self.fullscreen,
                              bgcolor=self.bg_color,
                              sync_swap=True)
        
        ## create letter box on top:
        self._ve_letterbox = Target2D(position=(self._centerPos[0], self.geometry[3] * (1 - 0.01) - self.letterbox_size[1]/2.),
                                     size=(self.letterbox_size[0], self.letterbox_size[1]),
                                     color=self.phrase_color)
        self._ve_innerbox = Target2D(position=(self._centerPos[0], self.geometry[3] * (1 - 0.01) - self.letterbox_size[1]/2.),
                               size=(self.letterbox_size[0]-6, self.letterbox_size[1]-6),
                               color=self.bg_color)
        
        self._current_letter_position = (self._centerPos[0], self.geometry[3] * (1 - 0.015) - self.letterbox_size[1]/2.)
        self._ve_current_letter = Text(position=self._current_letter_position,
                             text=(len(self._desired_letters[:1])==0 and " " or self._desired_letters[:1]),
                             font_size=self.font_size_current_letter,
                             color=self.current_letter_color,
                             anchor='center')
             
        self._ve_desired_letters = Text(position=(self._centerPos[0] + 5 + self.letterbox_size[0]/2., self._current_letter_position[1]),
                                        text=(len(self._desired_letters[1:])==0 and " " or self._desired_letters[1:]),
                                        font_size=self.font_size_phrase,
                                        color=self.phrase_color,
                                        anchor='left')
        
        self._ve_spelled_phrase = Text(position=(self._centerPos[0] - 5 - self.letterbox_size[0]/2., self._current_letter_position[1]),
                                       text=(len(self._spelled_phrase)==0 and " " or self._spelled_phrase),
                                       font_size=self.font_size_phrase,
                                       color=self.phrase_color,
                                       anchor='right')
        
        ## add word box to elementlist:
        self._ve_elements.extend([self._ve_letterbox, self._ve_innerbox, self._ve_current_letter, self._ve_desired_letters, self._ve_spelled_phrase])


        ## create countdown: 
        self._ve_countdown = Text(position=self._centerPos,
                                  text=" ",
                                  font_size=self.font_size_countdown,
                                  color=self.countdown_color,
                                  anchor='center',
                                  on=False)


        ## create countdown shapes 
        self._ve_countdown_shape=self.countdown_shapes[self.countdown_shape_select](radius=90, 
                                       position=self._centerPos, 
                                       color=self.countdown_shape_color,
                                       on=False)
        
        ## create oscillator circle:
        self._ve_oscillator = FilledCircle(position=(self.osc_size/2+10,self.osc_size/2+10),
                                       radius=self.osc_size/2,
                                       color=self.osc_color,
                                       on=False)
        

        
        ## create shapes and letters:
        self.init_screen_elements()


        
        ## add remaining elements to element list:
        self._ve_elements.extend([self._ve_countdown_shape, self._ve_countdown, self._ve_oscillator])
        
        ## add elements to viewport:
        self._viewport = Viewport(screen=self._screen, stimuli=self._ve_elements)
        self._presentation = Presentation(viewports=[self._viewport],
                                          handle_event_callbacks=[(pygame.KEYDOWN, self.keyboard_input),
                                                                  (pygame.QUIT, self.__stop)])
        
  


    def play_tick(self):
        """
        called every loop, if in play mode.
        """

        self.pre_play_tick()

        if self._state_countdown: 
            self.pre__countdown()
            self.__countdown()
            self.post__countdown()
        elif self._state_trial:
            self.pre__trial()
            self.__trial()
            self.post__trial()
        elif self._state_classify: 
            self.pre__classify()
            self.__classify()
            self.post__classify()
        elif self._state_feedback:
            self.pre__feedback()
            self.__feedback()
            self.post__feedback()
        elif self._state_abort:
            self.__abort()
            self.post__abort()
        else:
            self.pre__idle()
            self.__idle()
            self.post__idle()
        self.post_play_tick()
    
    def __stop(self, *args):
        self.on_stop()
    
    def __idle(self):
        if self.offline and len(self._desired_letters) > 0:
            # add new letter:
            for e in xrange(len(self.letter_set)):
                for l in xrange(len(self.letter_set[e])):
                    if self._desired_letters[0] == self.letter_set[e][l]:
                        self._classified_element = e
                        self._classified_letter = l
            if self.countdown_level1:
                self._state_countdown = True
            else:
                self._state_trial=True
        else:
            ## otherwise just wait until a new letter is sent:
            self._presentation.set(go_duration=(0.1, 'seconds'))
            self._presentation.go()        


    
    def __countdown(self):  
   
        def blink():
            i=0            
            while(i<self.countdown_blinking_nr):
                self._ve_countdown_shape.set(on=True)
                self._ve_countdown.set(text="%d" % self._current_countdown,on=True)
                self.send_parallel(self.COUNTDOWN_STIMULI )
                self.logger.info("[TRIGGER] %d" % self.COUNTDOWN_STIMULI )
                self._presentation.set(go_duration=(self.stimulus_duration, 'seconds'))
                self._presentation.go() 
                self._ve_countdown_shape.set(on=False)
                self._ve_countdown.set(on=False)
                self._presentation.set(go_duration=(self.interstimulus_duration, 'seconds'))
                self._presentation.go()  
                i=i+1

        
        if self._current_countdown == self.nCountdown:
            self.send_parallel(marker.COUNTDOWN_START)
            self.logger.info("[TRIGGER] %d" % marker.COUNTDOWN_START)
            self.set_countdown_screen()
            self._ve_countdown.set(on=True)
            self._presentation.set(go_duration=(1, 'seconds'))


        self._ve_countdown.set(text="%d" % self._current_countdown)
        self._presentation.go()
        self._current_countdown = (self._current_countdown-1) % self.nCountdown


        if self.synchronized_countdown and self._current_countdown ==1 and self.countdown_shape_on:
           self.set_synchronized_countdown_screen()
           blink()
           self._current_countdown = self.nCountdown
           self.set_standard_screen()
           self._state_countdown = False
           self._state_trial = True
           self._ve_countdown.set(on=False)
           self._ve_countdown_shape.set(on=False)            
           self._ve_countdown.set(color=self.countdown_color)
       
        if self._current_countdown == 0:
            # Executed only if self.synchronized_countdown = False
            self._current_countdown = self.nCountdown
            self.set_standard_screen()
            pygame.time.wait(10)
            self._state_countdown = False
            self._state_trial = True
            self._ve_countdown_shape.set(on=False)
            self._ve_countdown.set(on=False)
            self._ve_countdown.set(color=self.countdown_color)

   
    def __trial(self):
        
        if self._current_sequence==0 and self._current_stimulus==0:
            
            #level 1 animation when there is no countdown
            if self._current_level==1 and not self.countdown_level1: 
                 if self.do_animation:
                      self.set_countdown_screen()
                 self.set_standard_screen()
            # generate random sequences:
            if self.randomize_sequence:
                self.flash_sequence = []
                for _ in range(self.nr_sequences):
                    random_flash_sequence(self,
                                      set=range(self._nr_elements),
                                      min_dist=self.min_dist,
                                      seq_len=self._nr_elements)
            # or else use fixed sequence: 
            else:
                self.flash_sequence = range(self._nr_elements)
                
        
        if self.randomize_sequence:
            currentStimulus = self.flash_sequence[self._current_sequence*self._nr_elements + self._current_stimulus]
        else:
            currentStimulus = self.flash_sequence[self._current_stimulus]
        # set stimulus:
        self.stimulus(currentStimulus, True)
        #self._ve_oscillator.set(on=True)


        if self.abort_trial and self.abort_trial_check():
             # restart trial on abort_trial event:
            self._state_trial = False
            self._state_abort=True
            return

        # check if current stimulus is target and then send trigger:
        target_add = 0
        if len(self._desired_letters) > 0:
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
            
        # present stimulus:
        self._presentation.set(go_duration=(self.stimulus_duration, 'seconds'))
        self._presentation.go()

        # reset to normal:
        self._ve_oscillator.set(on=False)                
        self.stimulus(currentStimulus, False)

        # present interstimulus:
        self._presentation.set(go_duration=(self.interstimulus_duration, 'seconds'))
        self._presentation.go()
        
        if self.debug:
            self.on_control_event({'cl_output':(self.random.random(), currentStimulus+1)})
                
        ## TODO: check here for classification !!!!
        if self.output_per_stimulus:  
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
        else:
            # increase  
            self._current_stimulus = (self._current_stimulus+1) % self._nr_elements
            if self._current_stimulus == 0:
                self._current_sequence = (self._current_sequence+1) % self.nr_sequences
                if self.check_classification(self._current_sequence+1):
                    self._state_trial = False
                    self._state_classify = True
                    pygame.time.wait(self.wait_after_early_stopping*1000)
                      
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
    
    def check_classification(self,nr):
        #print self._classifier_output
        means = [None]*self._nr_elements
        minimum = maxint
        classified = None
        for ii in range(self._nr_elements):
            means[ii] = sum(self._classifier_output[ii]) /nr
            if means[ii]<minimum:
               minimum = means[ii]
               classified = ii+1
         
        print "\n**** Class: %d (mean=%f)\n" % (classified,means[classified-1]) 
        return classified                 
    



    def __classify(self):
 
        ## wait until all classifier outputs are received:
        self._presentation.set(go_duration=(self.wait_before_classify, 'seconds'))
        self._presentation.go()
        if self.offline:
           if self._current_level==1:
            classified = self._classified_element
           else:
                classified = self._classified_letter
        elif not self._debug_classified == None:
            classified = self._debug_classified
            self._debug_classified = None
        else:
            if self.output_per_stimulus:  
                nClassified = sum([len(self._classifier_output[i]) for i in xrange(self._nr_elements)])
                if nClassified < self._nr_elements * self.nr_sequences:
                    pygame.time.wait(20)
                    print 'not enough classifier-outputs received! (something may be wrong)'
                    return
            
                ## classify and set output:
                means = [None]*self._nr_elements
                minimum = maxint
                classified = None
                for ii in range(self._nr_elements):
                    means[ii] = sum(self._classifier_output[ii]) / self.nr_sequences
                    if means[ii]<minimum:
                        minimum = means[ii]
                        classified = ii
                print "\n**** Class: %d (mean=%f)\n" % (classified+1,means[classified])
            else:
                means = [None]*self._nr_elements
                minimum = maxint
                classified = None
                for ii in range(self._nr_elements):
                    means[ii] = sum(self._classifier_output[ii]) /self.nr_sequences
                    if means[ii]<minimum:
                        minimum = means[ii]
                        classified = ii
                print "\n**** Class: %d (mean=%f)\n" % (classified+1,means[classified])
            
            ## Reset classifier output to empty lists
            self._init_classifier_output()
        
        error_add = 0
        ## evaluate classification:
        if self._current_level == 1:
            self._classified_element = classified
            if len(self._desired_letters) > 0 and not self._desired_letters[:1] in self.letter_set[classified]:
                # wrong group selected:
                error_add = self.ERROR_ADD
        else:
            self._classified_letter = classified
            if self._classified_letter == self._idx_backdoor:
                ## backdoor classified:
                if len(self._desired_letters) > 0 and self._desired_letters[:1] in self.letter_set[self._classified_element]:
                    # backdoor selection wrong:
                    error_add = self.ERROR_ADD
            else:
                ## no backdoor classified:
                spelled_letter = self.letter_set[self._classified_element][self._classified_letter]
                if len(self._desired_letters) > 0 and spelled_letter != self._desired_letters[:1]:
                    # wrong letter spelled:
                    error_add = self.ERROR_ADD
        
        ## send response trigger:
        self.send_parallel(self.RESPONSE[self._current_level-1][classified] + error_add)
        self.logger.info("[TRIGGER] %d" % (self.RESPONSE[self._current_level-1][classified] + error_add))
        

        self._state_classify = False
        self._state_feedback = True

    
    def __feedback(self):
        self._state_feedback = False
        
        ## call subclass method:
        self.feedback()
        
        ## check ErrP classification:
        if self.use_ErrP_detection:
            t=0
            while self._ErrP_classifier is None and t<1000:
                t += 50
                pygame.time.wait(50)
            if self._ErrP_classifier is None:
                print "no ErrP classifier received! "
            if self._ErrP_classifier:
                self.send_parallel(self.ERROR_POTENTIAL)
                self.logger.info("[TRIGGER] %d" % (self.ERROR_POTENTIAL))                    
                
        ## call subclass method:
        if not self.countdown_level2:
            self.switch_level()
        
        ## update phrases:
        if ((self._current_level == 2) and # only update, if we are at the end of level 2,
           (self._classified_letter != self._idx_backdoor or self.copy_spelling) and # if copyspelling off, we have no backdoor selected
           (self._ErrP_classifier is None or not self._ErrP_classifier)): # no ErrP was detected (or ErrP detection is off)
            if self.copy_spelling:
                ## in copy spelling we force the desired letter to be spelled
                if len(self._desired_letters) > 0:
                    spelled_letter = self._desired_letters[:1]
                else:
                    print "??? moved beyond desired phrase in copy spelling ???"
            else:
                spelled_letter = self.letter_set[self._classified_element][self._classified_letter]
                
            ## update desired phrase:
            if len(self._desired_letters) > 0:
                if spelled_letter == self._desired_letters[:1]:
                    # correct letter spelled:
                    self._desired_letters = self._desired_letters[1:] # remove first letter
                else:
                    # wrong letter spelled:
                    if spelled_letter == "<":
                        self._desired_letters = self._spelled_phrase[-1:] + self._desired_letters
                    else:
                        self._desired_letters = "<" + self._desired_letters
                if len(self._desired_letters) == 0:
                    self._copyspelling_finished = True
                        
            ## update spelled phrase:
            self._spelled_letters += spelled_letter
            if spelled_letter == "<":
                self._spelled_phrase = self._spelled_phrase[:-1]
            else:
                self._spelled_phrase += spelled_letter
            
            ## update screen phrases:
            self._ve_spelled_phrase.set(text=(len(self._spelled_phrase)==0 and " " or self._spelled_phrase))
            self._ve_current_letter.set(text=(len(self._desired_letters[:1])==0 and " " or self._desired_letters[:1]))
            self._ve_desired_letters.set(text=(len(self._desired_letters[1:])==0 and " " or self._desired_letters[1:]))
        
        if self.use_ErrP_detection and self._ErrP_classifier:
            self._state_trial = True

        else:
            if self._current_level == 1:
                # continue with level2 trial:
                if self.countdown_level2:
                    self._state_countdown = True
                else:
                    self._state_trial = True
                    
                
            elif not self.offline:
                # start countdown
                if self.countdown_level1:
                    self._state_countdown = True
                else: 
                    self._state_trial=True
            
            # set new level:
            self._current_level = 3 - self._current_level
        
        ## reset ErrP_classifier:
        self._ErrP_classifier = None
        
        # check copyspelling:
        if self._copyspelling_finished:
            self._copyspelling_finished = False
            self.on_control_event({'print':0}) # print desired phrase
            self.on_control_event({'print':1}) # print spelled phrase
            self.on_control_event({'print':2}) # print all spelled letters
            self.send_parallel(self.COPYSPELLING_FINISHED)
            self.logger.info("[TRIGGER] %d" % (self.COPYSPELLING_FINISHED))
            pygame.time.wait(50)
            


    def __abort(self):

        # play warning sound 

        def sine_array_onecycle(hz, peak,sample_rate):
            length = sample_rate / float(hz)
            omega = NP.pi * 2 / length
            xvalues = NP.arange(int(length)) * omega
            return (peak * NP.sin(xvalues))
    
        def sine_array(hz, peak, samples_rate ):
            return NP.resize(sine_array_onecycle(hz, peak,sample_rate), (sample_rate,))
        

        sample_rate=44100 
        pygame.mixer.init(sample_rate, -16, 2) # 44.1kHz, 16-bit signed, stereo
        f = sine_array(8000, 1,sample_rate)
        f = NP.array(zip (f , f))
        sound = pygame.sndarray.make_sound(f)
        channel = sound.play(-1)
        channel.set_volume(0.2,0.2)
        pygame.time.delay(1000)
        sound.stop()

  
        if self._current_level==1 and self.countdown_level1:
            self._state_countdown = True
        elif self._current_level==2 and self.countdown_level2:
            self._state_countdown= True
        else:
            self._state_trial= True
        
        self._init_classifier_output() 
      


    def _init_classifier_output(self):
        ## Empty lists
        self._classifier_output = [list() for _ in xrange(self._nr_elements)]

    def abort_trial_check(self): 
        '''
        Check if event is an abort trial event 
        '''
        return False
 
    

    def keyboard_input(self, event):
        if event.key == pygame.K_ESCAPE:
            self.on_stop()
        elif event.key == pygame.K_KP_ENTER:
            self.on_control_event({'print':0}) # print desired phrase
            self.on_control_event({'print':1}) # print spelled phrase
            self.on_control_event({'print':2}) # print all spelled letters
        elif self.debug:
            if ((event.key >= pygame.K_a and event.key <= pygame.K_z) or
                (event.key == pygame.K_LESS) or
                (event.key == pygame.K_PERIOD) or
                (event.key == pygame.K_COMMA)):
                self.on_control_event({'new_letter':chr(event.key).upper()})
            elif event.key == pygame.K_MINUS:
                self.on_control_event({'new_letter':chr(pygame.K_UNDERSCORE)})
            elif event.key == pygame.K_BACKSPACE:
                self.on_control_event({'new_letter':chr(pygame.K_LESS)})
            elif event.key == pygame.K_SPACE:
                self.on_control_event({'new_letter':chr(pygame.K_UNDERSCORE)})
            elif event.key == pygame.K_UP and self.use_ErrP_detection:
                self.on_control_event({'cl_output':(1,7)})
            elif event.key == pygame.K_DOWN and self.use_ErrP_detection:
                self.on_control_event({'cl_output':(0,7)})
            if not self.offline:
                if (event.key >= pygame.K_0 and event.key <= pygame.K_5):
                    self._debug_classified = int(chr(event.key))
                elif (event.key >= pygame.K_KP0 and event.key <= pygame.K_KP5):
                    self._debug_classified = int(chr(event.key-208))

    
    def on_control_event(self, data):
        self.logger.info("[CONTROL_EVENT] %s" % str(data))
        if data.has_key(u'cl_output'):
            # classification output was sent:
            score_data = data[u'cl_output']
            cl_out = score_data[0]
            iSubstim = int(score_data[1]) # evt auch "Subtrial"
            if iSubstim in range(1,7):
                self._classifier_output[iSubstim-1].append(cl_out)
            elif self.use_ErrP_detection:
                self._ErrP_classifier = cl_out
        elif data.has_key('new_letter'):
            # get new letter to spell:
            self._desired_letters += data['new_letter']
            self._ve_current_letter.set(text=(len(self._desired_letters[:1])==0 and " " or self._desired_letters[:1]))
            self._ve_desired_letters.set(text=(len(self._desired_letters[1:])==0 and " " or self._desired_letters[1:]))

        elif data.has_key(u'print'):
            if data[u'print']==0:
                self.logger.info("[DESIRED_PHRASE] %s" % self.desired_phrase)
            elif data[u'print']==1:
                self.logger.info("[SPELLED_PHRASE] %s" % self._spelled_phrase)
            elif data[u'print']==2:
                self.logger.info("[SPELLED_LETTERS] %s" % self._spelled_letters)
        
    
    '''
    ==========================
    == METHODS TO OVERLOAD: ==
    ==========================
    '''
    def init_screen_elements(self):
        '''
        overwrite this function in subclass.
        '''
        pass
    
    def prepare_mainloop(self):
        '''
        overwrite this function in subclass.
        '''
        pass
    
    def set_countdown_screen(self):
        '''
        set screen how it should look during countdown.
        overwrite this function in subclass.
        '''
        pass
    

    def set_standard_screen(self):
        '''
        set screen elements to standard state.
        overwrite this function in subclass.
        '''
        pass

    def set_synchronized_countdown_screen(self):
        '''
        set screen elements to for the synchronized countdown.
        overwrite this function in subclass.
        '''
        pass
    
    def stimulus(self, i_element, on=True):
        '''
        turn on/off the stimulus elements and turn off/on the normal elements.
        overwrite this function in subclass.
        '''    
        pass
    
    def feedback(self):
        '''
        set screen how it should look during feedback presentation.
        overwrite this function in subclass.
        '''
        pass
    
    def switch_level(self):
        '''
        overwrite this function in subclass.
        '''
        pass
        
    def pre_play_tick(self):
        pass
    def post_play_tick(self):
        pass
    
    def pre__countdown(self):
        pass
    def post__countdown(self):
        pass
    
    def pre__trial(self):
        pass
    def post__trial(self):
        pass
    
    def pre__classify(self):
        pass
    def post__classify(self):
        pass
    
    def pre__feedback(self):
        pass
    def post__feedback(self):
        pass
    
    def pre__idle(self):
        pass
    def post__idle(self):
        pass
     
    def post__abort(self):
        pass       

def animate(pos_start, pos_end, dt):
    return NP.add(pos_start, NP.multiply(NP.subtract(pos_end, pos_start), dt))

def animate_sigmoid(pos_start, pos_end, dt, accel=10.):
    return NP.add(pos_start, NP.divide(NP.subtract(pos_end, pos_start), (1. + NP.exp(-accel*dt + 0.5*accel))))

def animate_sigmoid2D(pos_start, pos_end, t, T, accel=10.):
    return (pos_start[0] + (pos_end[0]-pos_start[0])/(1. + NP.exp(-accel*(t/T) + 0.5*accel)),
            pos_start[1] + (pos_end[1]-pos_start[1])/(1. + NP.exp(-accel*(t/T) + 0.5*accel)))

if __name__ == '__main__':
    import CenterSpellerVE
    fb = CenterSpellerVE.CenterSpellerVE()
    fb.on_init()
    fb.on_play()
