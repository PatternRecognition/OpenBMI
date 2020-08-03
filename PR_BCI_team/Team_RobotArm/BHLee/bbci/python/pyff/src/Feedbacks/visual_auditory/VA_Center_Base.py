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

Created on 2011-12-5

@author: "Xingwei An"

'''
from time import clock
from FeedbackBase.MainloopFeedback import MainloopFeedback
from lib.P300Aux.P300Functions import random_flash_sequence
from lib.eyetracker import EyeTracker
from VEShapes import FilledTriangle, FilledHexagon,FilledHourglass,FilledCross
from VisionEgg.Core import Screen
from VisionEgg.Core import Viewport
from VisionEgg.FlowControl import Presentation
from VisionEgg.MoreStimuli import Target2D, FilledCircle, Arrow
from VisionEgg.Text import Text
from VisionEgg import logger
# from lib.P300Layout.CircularLayout import CircularLayout
#import numpy as NP
import numpy.core.umath as NP
import random, pygame, os, math
import logging
#from logging.handlers import FileHandler
from sys import platform, maxint
if platform == 'win32':
    import winsound


class VA_Center_Base(MainloopFeedback):
    '''
    Visual Speller with six circles like the classical HexOSpell.
    '''

    # Triggers:
    SHORTPAUSE_START, SHORTPAUSE_END = 249, 250
    RUN_START, RUN_END = 252, 253
    END_LEVEL1, END_LEVEL2 = 244,245               # end of hex levels
    COPYSPELLING_FINISHED = 246
    COUNTDOWN_START = 240
    STIMULUS = [ [11, 12, 13, 14, 15, 16, 17, 18] , [21, 22, 23, 24, 25, 26, 27, 28] ]
    RESPONSE = [ [51, 52, 53, 54, 55, 56, 57, 58] , [61, 62, 63, 64, 65, 66, 67, 68] ]
    TARGET_ADD = 20
    ERROR_ADD = 100
    INVALID_FIXATION = 99   # send if eyetracker detects fixation off the target location
    ERROR_POTENTIAL = 96 # send if error potential is classified

    def init(self):
        '''
        initialize parameters
        '''
        self.log_filename = 'VA_Center_Base.log'
        ## some parameter to be changed
        ## sizes:
#        self.screenPos = [0, 1200, 1280, 730]
        self.screenPos = [0, 0, 1280, 800]
#        self.screenPos = [1920, 0, 1920, 1080]  # the lab screen        
        
        self.auditory_stimulation = True
        self.visual_stimulation = True
        
        self.stimulus_duration = 0.130   # 5 frames @60 Hz = 83ms flash
        self.interstimulus_duration = 0.170
        self.nr_sequences = 5
        
        
        
        
        self.letterbox_size = (60,60)
        self._spellerHeight = self.screenPos[3] - self.letterbox_size[1]
        self._centerPos = (self.screenPos[2]/2., self._spellerHeight/2.)
        
#        ##  1 ########################initialize the screen parameters
        

        
        self.osc_size = 40
        self.font_size_phrase = 60       # the spelled phrase at the top
        self.font_size_current_letter = 80       # the spelled phrase at the top
        self.font_size_countdown = 150  # number during countdown
        self.desired_phrase = "BBCI_TU_BERLIN"
        self.auditory_cue_radius = 300   # the cirle for auditory countdown
        ## colors:
        self.bg_color = (0., 0., 0.)
        self.phrase_color = (0.2, 0.0, 1.0)
        self.current_letter_color = (1.0, 0.0, 0.0)
        self.countdown_color = (0.2, 0.9, 0.2)
        self.osc_color = (1,1,1)

        self.letter_set_all = [['A','B','C','D','E','^',' ',' '], \
                           ['F','G','H','I','J','^',' ',' '], \
                           ['K','L','M','N','O','^',' ',' '], \
                           ['P','Q','R','S','T','^',' ',' '], \
                           ['U','V','W','X','Y','^',' ',' '], \
                           ['Z','_','.',',','<','^',' ',' '], \
                           [' ',' ',' ',' ',' ',' ',' ',' '], \
                           [' ',' ',' ',' ',' ',' ',' ',' ']
                           ]
        self.fullscreen = False
        self.use_oscillator = True
        self.offline = True
        self.copy_spelling = True  # in copy-spelling mode, selection of the target symbol is forced
        self.debug = True
        
        self.randomize_sequence = True # set to False to present a fixed stimulus sequence
        self.min_dist = 2 # Min number of intermediate flashes bef. a flash is repeated twice
        self.animation_time = 1.
        
        self.wait_before_classify = 1.
        self.feedback_duration = 1.
        self.feedback_ErrP_duration = 1.0

        # Eyetracker settings
        self.use_eyetracker = False
        self.et_currentxy = (0, 0)      # Current fixation
        self.et_duration = 100
        self.et_range = 100        # Maximum acceptable distance between target and actual fixation
        self.et_range_time = 200    # maximum acceptable fixation off the designated point
        
        self.use_ErrP_detection = True
        
        ## 2 ########################## initialize the stimulation parameters
#        self._nr_elements = 4
#        self._idx_backdoor = 3
#        self._nr_elements_A = 4
#        self.nCountdown = 3            
        self.synchronized_countdown = True
        self.do_animation = True
        ## sizes:
        self.letter_radius = 40
        self.speller_radius = 220
        self.font_size_level1 = 45         # letters in level 1
        self.font_size_level2 = 130         # letters in level 2
        self.feedbackbox_size = 200.0
        self.fixationpoint_size = 4.0
        self.font_size_feedback_ErrP = 300
        
        ## stimulus types:
        self.stimulustype_color = True
        self.shape_on=True
        self.edge_color = (1.0, 1.0, 1.0)



        ## feedback type:
        self.feedback_show_shape = True
        self.feedback_show_shape_at_center =True
        
        ## level 2 appearance:
        self.level_2_symbols = True
        self.level_2_letter_colors = False
        self.level_2_animation = True
        self.backdoor_symbol = "^"
        #        self.backdoor_symbol = unichr(0x2191)
        
        ## colors:
        self.shape_color = (1.0, 1.0, 1.0)

        self.stimuli_colors_all = [[0.0,0.0,1.0],
                                  [0.0,0.53,0.006],
                                  [1.0,0.0,0.0],
                                  [1.0,1.0,0.0],
                                  [0.86,0.0,0.86],
                                  [0.,0.95,0.95],
                                  [1.0,0.5,0.5],
                                  [1.0,1.0,1.0]]
        self.letter_color = (.5, .5, .5)
        self.feedback_color = (0.9, 0.9, 0.9)
        self.fixationpoint_color = (1.0, 1.0, 1.0)
        self.feedback_ErrP_color = (0.7, 0.1, 0.1)
        
        ## register possible shapes:
        self.registered_shapes = {'circle':FilledCircle,
                                  'cross' : FilledCross,
                                  'hexagon':FilledHexagon,
                                  'hourglass':FilledHourglass,
                                  'triangle':FilledTriangle,
                                  'rectangle':Target2D,
                                  'arrow':Arrow}
                                  
        
        ## define shapes in the form ['name', {parameters}]:
        # where parameters can be VisionEgg parameters eg. 'radius', 'size', 'orientation' etc.

        self.shapes = [
            [ "triangle", {"innerColor":self.bg_color, "innerSize": 60, "size": 200}], 
            [ "hourglass", {"size": 100 }], 
            [ "cross", {"orientation": 45, "size": [30,180], "innerColor":self.bg_color}], 
            [ "triangle", {"innerColor":self.bg_color, "innerSize": 60, "orientation": 180, "size": 200}], 
            [ "hourglass", {"orientation": 90, "size": 100}], 
            [ "cross", {"size": [30,180], "innerColor":self.bg_color }],
            ["arrow",{"size":[200,30]}],
            [ "hourglass", {"orientation": 135, "size": 80}]
        ]
#
#        self._circle_layout = CircularLayout(nr_elements=self._nr_elements,
#                                           radius=self.speller_radius,
#                                           start=NP.pi/self._nr_elements*(self._nr_elements-1))  ##  
#        
#        self._circle_layout.positions.reverse()
#        
#        auditory_pos=CircularLayout(nr_elements=self._nr_elements,
#                                           radius=self.speller_radius+150,
#                                           start=NP.pi/self._nr_elements*(self._nr_elements-1))
#        auditory_pos.positions.reverse()
#        self.auditory_cue_pos=[]
#        for i in range(self._nr_elements_A):
#            self.auditory_cue_pos.append((self._centerPos[0]+auditory_pos.positions[i][0],self._centerPos[1]+auditory_pos.positions[i][1]))
        
        
        
        
        ## 6 shoold be replaced by nr_elements 5 should bereplaced by nr_elements-1
        
        '''
        For auditory stimulation
        '''
        self.init_pygame()
        self.sounds = ["sounds/set8-2/2.wav", "sounds/set8-2/3.wav", "sounds/set8-2/9.wav", "sounds/set8-2/8.wav", \
          "sounds/set8-2/7.wav", "sounds/set8-2/1.wav","sounds/set8-2/2.wav","sounds/set8-2/5.wav",]
        self.soundlist = []
        self.loadAuditoryStimuli(self.sounds)
        
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

        
        self._init_classifier_output()
        self._classified_element = -1   
        self._classified_letter = -1
        for s in self.desired_phrase:
            assert s in [l for ls in self.letter_set for l in ls] # invalid letters in desired phrase!
        self._spelled_phrase = ""
        self._spelled_letters = ""
        self._desired_letters = self.desired_phrase
        self._copyspelling_finished = False
                
                    
        
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
        self._state_countdown = not self.offline
        self._state_trial = False
        self._state_classify = False
        self._state_feedback = False
        
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

        ## current letter in offline mode:
#        self._offline_current_letter = None

        ## Eyetracker
        if self.use_eyetracker:
            self.et = EyeTracker()
            self.et.start()
            cwd = os.path.dirname(__file__)
            self.soundfile = os.path.join(cwd,'winSpaceCritStop.wav')
            #pygame.mixer.init()
            #pygame.time.wait(50)
            #self.sound_invalid = pygame.mixer.Sound(os.path.join(cwd,'winSpaceCritStop.wav'))
            #print ">>>>>>>>>>>>>>>>>>>>>>>>>> ",os.path.join(cwd,'winSpaceCritStop.wav')

        ## send start trigger:
        pygame.time.wait(1000)
        self.send_parallel(self.RUN_START)
        self.logger.info("[TRIGGER] %d" % self.RUN_START)
        
        ## error potential classifier:
        self._ErrP_classifier = None
        
    def post_mainloop(self):
        """
        Sends end marker to parallel port.
        """

        if self.use_eyetracker:
            self.et.stop()
            pygame.time.wait(10)
            self.et.close()
            
        pygame.time.wait(500)
        self.send_parallel(self.RUN_END)
        self.logger.info("[TRIGGER] %d" % self.RUN_END)
        pygame.time.wait(500)
        self._presentation.set(quit=True)
        self._screen.close()
    
    
    def __init_screen(self):
        ## create screen:
        if not self.fullscreen:
            os.environ['SDL_VIDEO_WINDOW_POS'] = '%d, %d' % (self.screenPos[0], self.screenPos[1])
        self._screen = Screen(size=(self.screenPos[2], self.screenPos[3]),
                              fullscreen=self.fullscreen,
                              bgcolor=self.bg_color,
                              sync_swap=True)
        
       
        
        ## 1 create letter box on top:
        
        self._ve_letterbox = Target2D(position=(self._centerPos[0], self.screenPos[3] * (1 - 0.01) - self.letterbox_size[1]/2.),
                                     size=(self.letterbox_size[0], self.letterbox_size[1]),
                                     color=self.phrase_color)
        ve_innerbox = Target2D(position=(self._centerPos[0], self.screenPos[3] * (1 - 0.01) - self.letterbox_size[1]/2.),
                               size=(self.letterbox_size[0]-6, self.letterbox_size[1]-6),
                               color=self.bg_color)
        
        self._current_letter_position = (self._centerPos[0], self.screenPos[3] * (1 - 0.015) - self.letterbox_size[1]/2.)
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
        
        ## create countdown:
        self._ve_countdown = Text(position=self._centerPos,
                                  text=" ",
                                  font_size=self.font_size_countdown,
                                  color=self.countdown_color,
                                  anchor='center',
                                  on=False)
        ##
        self._auditory_cue = FilledCircle(position=(self._centerPos[0]-self.auditory_cue_radius,self._centerPos[1]-self.auditory_cue_radius),
                                       radius=self.osc_size,
                                       color=self.osc_color,
                                       on=False)
        ## create oscillator circle:
        self._ve_oscillator = FilledCircle(position=(self.osc_size/2+10,self.osc_size/2+10),
                                       radius=self.osc_size/2,
                                       color=self.osc_color,
                                       on=False)
        
        ## add word box to elementlist:
        self._ve_elements.extend([self._ve_letterbox, ve_innerbox, self._ve_current_letter, self._ve_desired_letters, self._ve_spelled_phrase,self._auditory_cue])
        
        ## create shapes and letters:
        self.init_screen_elements()
        
        ## add remaining elements to element list:
        self._ve_elements.extend([self._ve_countdown, self._ve_oscillator])
        
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
        else:
            self.pre__idle()
            self.__idle()
            self.post__idle()
#        else:
#            self.pre__trial()
#            self.__trial()
#            self.post__trial()
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
            self._state_countdown = True
        else:
            ## otherwise just wait until a new letter is sent:
            self._presentation.set(go_duration=(0.1, 'seconds'))
            self._presentation.go()        
    
    def __countdown(self):
        for i in xrange(self._nr_elements*self._nr_elements):
            self._ve_letters[i].set(color = self.letter_color)
        if self._current_countdown == self.nCountdown:
            self.send_parallel(self.COUNTDOWN_START)
            self.logger.info("[TRIGGER] %d" % self.COUNTDOWN_START)
            self.set_countdown_screen()
            self._ve_countdown.set(on=True)
            self._presentation.set(go_duration=(1, 'seconds'))
        self._ve_countdown.set(text="%d" % self._current_countdown)
        #
 
        for i in range(self._nr_elements):
            self._ve_letters[i*(self._nr_elements)+self._nr_elements_A-self._current_countdown].set(color = self.countdown_color)
        
        self.start_auditory_cue((self._nr_elements_A-self._current_countdown) % self._nr_elements)
        self._auditory_cue.set(position = self.auditory_cue_pos[self._nr_elements_A-self._current_countdown],on = True)
        #
        self._presentation.go()
        #*
        #self._ve_letters[self._current_countdown-1].set(color = self.edge_color)
        for i in range(self._nr_elements):
            self._ve_letters[i*(self._nr_elements)+self._nr_elements_A-self._current_countdown].set(color = self.letter_color)
        self._auditory_cue.set(on = False)
        #*
        self._current_countdown = (self._current_countdown-1) % self.nCountdown
        if self._current_countdown == 0:
            if self.offline:
                for e in xrange(len(self.letter_set)):
                    for l in xrange(len(self.letter_set[e])):
                        if self._desired_letters[0] == self.letter_set[e][l]:
                            tm_e = e
                            tm_l = l
                self._ve_countdown.set(on=False, text=" ")
                for i in xrange(3):
                    if self._current_level==1:
                        self.start_auditory_cue(tm_e % self._nr_elements)
                        self._auditory_cue.set(position = self.auditory_cue_pos[tm_e],on = True)
                    else:
                        self.start_auditory_cue(tm_l % self._nr_elements)
                        self._auditory_cue.set(position = self.auditory_cue_pos[tm_l],on = True)
                    self._presentation.go() 
            self._auditory_cue.set(on = False)
                      
            self._current_countdown = self.nCountdown
            self._ve_countdown.set(on=False, text=" ")
            self.set_standard_screen()
            pygame.time.wait(20)
            self._state_countdown = False
            self._state_trial = True
            
            
    def __countdown_level2(self):
        if self._current_countdown == self.nCountdown:
            self.send_parallel(self.COUNTDOWN_START)
            self.logger.info("[TRIGGER] %d" % self.COUNTDOWN_START)
            self.set_countdown_screen()
            self._ve_countdown.set(on=True)
            self._presentation.set(go_duration=(1, 'seconds'))
        self._ve_countdown.set(text="%d" % self._current_countdown)
        #
        if self._current_countdown>1:
            for i in range(self._nr_elements):
                self._ve_letters[i*(self._nr_elements-1)+self._nr_elements-self._current_countdown].set(color = self.countdown_color)
        
        self.start_auditory_cue((self._nr_elements-self._current_countdown) % self._nr_elements)
        self._auditory_cue.set(position = self.auditory_cue_pos[self._nr_elements-self._current_countdown],on = True)
        #
        self._presentation.go()
        #*
        self._ve_letters[self._current_countdown-1].set(color = self.edge_color)
        self._auditory_cue.set(on = False)
        #*
        self._current_countdown = (self._current_countdown-1) % self.nCountdown
        if self._current_countdown == 0:
            self._current_countdown = self.nCountdown
            self._ve_countdown.set(on=False, text=" ")
            self.set_standard_screen()
            pygame.time.wait(20)
            self._state_countdown = False
            self._state_trial = True
    
    
    def __trial(self):
        if self._current_sequence==0 and self._current_stimulus==0:
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
        if self.visual_stimulation:
            self.stimulus(currentStimulus, True)
        else:
            self.stimulus_A_only(True)
        
        self._ve_oscillator.set(on=True)

        if self.use_eyetracker and not self.eyetracker_input():
            # restart trial on eye movements:
            self._state_trial = False
            self._state_countdown = True
            self._init_classifier_output()
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
        
       
        
        self.start_auditory_cue((currentStimulus) % self._nr_elements)
        
        self.send_parallel(self.STIMULUS[self._current_level-1][currentStimulus] + target_add)
        self.logger.info("[TRIGGER] %d" % (self.STIMULUS[self._current_level-1][currentStimulus] + target_add))
            
        # present stimulus:
        self._presentation.set(go_duration=(self.stimulus_duration, 'seconds'))
        self._presentation.go()
        
        
        self.send_parallel(self.STIMULUS[self._current_level-1][currentStimulus] + target_add)
        self.logger.info("[TRIGGER] %d" % (self.STIMULUS[self._current_level-1][currentStimulus] + target_add))
        # reset to normal:
        self._ve_oscillator.set(on=False) 
        if self.visual_stimulation:
            self.stimulus(currentStimulus, False)

        # present interstimulus:
        self._presentation.set(go_duration=(self.interstimulus_duration, 'seconds'))
        self._presentation.go()

        self.send_parallel(self.STIMULUS[self._current_level-1][currentStimulus] + target_add)
        self.logger.info("[TRIGGER] %d" % (self.STIMULUS[self._current_level-1][currentStimulus] + target_add))
        
        if self.debug:
            self.on_control_event({'cl_output':(self.random.random(), currentStimulus+1)})
                
        ## toDO: check here for classification, if you want to use adaptive stopping of the trial!!!!
        
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
            for i in xrange(self._nr_letters):
                self._ve_letters[i].set( on=False)
            for i in xrange(self._nr_elements):
                self._ve_letters[self._nr_letters + i].set(on=False)
               
    
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
        #self.__countdown_level2()
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
#            self._state_countdown = True
        else:
            if self._current_level == 1:
                # continue with level2 trial:
                self._state_trial = True
            elif not self.offline:
                # start countdown
                self._state_countdown = True
            
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
            
    

    def _init_classifier_output(self):
        ## Empty lists
        self._classifier_output = [list() for _ in xrange(self._nr_elements)]

    def eyetracker_input(self):
        # Control eye tracker
        if self.et.x is None:
            #print("[ERP Hex] No eyetracker data received!")
            self.logger.warning("[EYE_TRACKER] No eyetracker data received!")
        self.et_currentxy = (self.et.x, self.et.y)
        self.et_duration = self.et.duration
        tx, ty = self._centerPos[0], self._centerPos[1]
        cx, cy = self.et_currentxy[0], self.et_currentxy[1]
        dist = math.sqrt(math.pow(tx - cx, 2) + math.pow(ty - cy, 2))
        self.logger.info("[EYE_TRACKER] position=(%f,%f)" % (self.et.x, self.et.y))
        # Check if current fixation is outside the accepted range 
        if dist > self.et_range:
            # Check if the off-fixation is beyond temporal limits
            if self.et_duration > self.et_range_time:
                # Break off current trial !!
                if platform == 'win32':
                    winsound.PlaySound(self.soundfile, winsound.SND_ASYNC)

                self._presentation.set(go_duration=(1., 'seconds'))
                self._presentation.go()

                # Send break-off trigger
                self.send_parallel(self.INVALID_FIXATION)
                self.logger.info("[TRIGGER] %d" % self.INVALID_FIXATION)
                return False
        return True
    
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
            if iSubstim in range(1,self._nr_elements+1):
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
    def init_pygame(self):
        pygame.mixer.quit()
        #pygame.mixer.init(frequency=15000, size=-16, channels=2, buffer=512)
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512) # negative bit-size for signed values

        # The reduced buffer of 1024bits (512) is very important. With the standard 
        # value the latency is varying dramatically

        sample_rate, bit_rate, channels = pygame.mixer.get_init()
        self.logger.info('sampling rate = ' + str(sample_rate) )

        #self.audioChannel = pygame.mixer.find_channel() ## TODO
        self.audioChannel = pygame.mixer.Channel(1) 
        
    def playSoundObject(self, sound):
        """
        play the sound with the given filename or mixer.Sund object with the class-intern channel 
        """
        if os.path.isfile(sound): #conversion if filename was given
            s = pygame.mixer.Sound(sound)
            print "sound loaded from path"
        else:
            s = sound
        print 'PLAY'
        self.audioChannel.play(s)
        pygame.event.pump()
        
    
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
    
    def stimulus(self, i_element, on=True):
        '''
        turn on/off the stimulus elements and turn off/on the normal elements.
        overwrite this function in subclass.
        '''    
        pass
    def stimulus_A_only(self, on=True):
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
    def start_auditory_cue(self,stimulusnum):
        """
        performs a single trial -> one auditory stimulus
        """     
        
        sound = self.soundlist[stimulusnum]
        
        #self.audioChannel.play(sound) 
        sound.play()
        pygame.event.pump()
    
    def loadAuditoryStimuli(self, sounds):
        
        path = os.path.dirname( globals()["__file__"] ) 
        #dummy
#        sound0 = pygame.mixer.Sound(os.path.join(path, sounds[0])) 

        ##high pitch
        sound1 = pygame.mixer.Sound(os.path.join(path, sounds[0])) 
        sound2 = pygame.mixer.Sound(os.path.join(path, sounds[1])) 
        sound3 = pygame.mixer.Sound(os.path.join(path, sounds[2])) 
        #medium pitch
#        sound4 = pygame.mixer.Sound(os.path.join(path, sounds[3])) 
#        sound5 = pygame.mixer.Sound(os.path.join(path, sounds[4])) 
#        sound6 = pygame.mixer.Sound(os.path.join(path, sounds[5])) 
        
        #low pitch
        sound4 = pygame.mixer.Sound(os.path.join(path, sounds[3]))#kick_l.wav")) 
        sound5 = pygame.mixer.Sound(os.path.join(path, sounds[4]))#kick_m.wav")) 
        sound6 = pygame.mixer.Sound(os.path.join(path, sounds[5])) 
        sound7 = pygame.mixer.Sound(os.path.join(path, sounds[6]))#kick_m.wav")) 
        sound8 = pygame.mixer.Sound(os.path.join(path, sounds[7])) 
        self.soundlist = [sound1, sound2, sound3, sound4, sound5, sound6,sound7,sound8]
           

def animate(pos_start, pos_end, dt):
    return NP.add(pos_start, NP.multiply(NP.subtract(pos_end, pos_start), dt))

def animate_sinusoid(pos_start, pos_end, dt):
    xpos = dt * math.pi - math.pi/2 # Blend between -pi and pi
    ypos = math.sin(xpos)/2 + .5         # 0 to 1
    return NP.multiply(pos_start, 1.-ypos) + NP.multiply(pos_end,ypos)
    #return NP.add(pos_start, NP.divide(NP.subtract(pos_end, pos_start), (1. + NP.exp(-accel*dt + 0.5*accel))))

def animate_sigmoid(pos_start, pos_end, dt, accel=10.):
    return NP.add(pos_start, NP.divide(NP.subtract(pos_end, pos_start), (1. + NP.exp(-accel*dt + 0.5*accel))))

def animate_sigmoid2D(pos_start, pos_end, t, T, accel=10.):
    return (pos_start[0] + (pos_end[0]-pos_start[0])/(1. + NP.exp(-accel*(t/T) + 0.5*accel)),
            pos_start[1] + (pos_end[1]-pos_start[1])/(1. + NP.exp(-accel*(t/T) + 0.5*accel)))

if __name__ == '__main__':
    import VA_CenterSpeller
    logging.basicConfig(level=logging.DEBUG)
    fb = VA_CenterSpeller.VA_CenterSpeller()
    fb.on_init()
    fb.on_play()
