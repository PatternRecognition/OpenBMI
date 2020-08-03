'''Feedbacks.BGUI.BGUIBaseHybrid
# Copyright (C) 2012  "Javier Pascual"
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

Created on Jan 01, 2012

@author: "Javier Pascual"

'''

from time import clock
from FeedbackBase.MainloopFeedback import MainloopFeedback
from lib.P300Aux.P300Functions import random_flash_sequence
#from lib.eyetracker import EyeTracker

from VisionEgg.Core import Screen
from VisionEgg.Core import Viewport
from VisionEgg.FlowControl import Presentation
from VisionEgg.MoreStimuli import Target2D
from VisionEgg.Text import Text
from VisionEgg import logger
from VisionEgg.Gratings import *

import numpy as NP
import random, pygame, os, math
import logging
import copy
import csv
import os

from MenuElement import *
from LevelContainer import *
from VisionEgg.Textures import *

from sys import platform, maxint


class BGUIBaseHybrid(MainloopFeedback):
    '''
    Visual Speller with six circles like the classical HexOSpell.
    '''

    # Triggers:
    RUN_START, RUN_END = 252, 255
    EMERGENCY_BREAK = 300
    
    COUNTDOWN_START = 240

    TARGET_ADD = 20
    ERROR_ADD = 100
    
    STATE_IDLE      = 0 
    STATE_COUNTDOWN = 1
    STATE_TRIAL     = 2
    STATE_CLASSIFY  = 3
    STATE_FEEDBACK  = 4        
    
    def init(self):
        '''
        initialize parameters
        '''
        self.g_rel =  0.0025
        self.FPS = 5
        self._levels = None
        self.letterbox_size = [60, 60] 
# biglab
        self.screenPos = [-2560,-390, 2500, 1600]

        self.screenPos = [1280,-512, 1920, 1200]
        self.screenPos = [0,0, 800, 800]
        self.step = 0
        self.step_idle = 0
        self.trial_elapsed_time = 0
        
        self.log_filename = "bgui.log"
        self._centerPos = (self.screenPos[2]/2., (self.screenPos[3] - self.letterbox_size[1])/2.)        
        
        logger.setLevel(logging.ERROR)
        handler = logging.FileHandler(self.log_filename, 'w')
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("BGUI: %(asctime)s: %(message)s")
        handler.setFormatter(formatter)
        
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(handler)                                        
        self.random = random.Random(clock())
           
        assert(len(self.log_filename)!=0) # 'log_filename' must not be empty string!
        self._message_txt = ' '
        self._message = None 
        self._pause_after = 4 
        self._pause_time = 15       
        self.num_trials = 0;
            
                    
    def init_textures(self, maxsize):
                                       
        self.min_dist = 2 # Min number of intermediate flashes bef. a flash is repeated twice           
   
        self._levels = None
        self._message =   Text( position=(self._centerPos[0], self._centerPos[1]+400),
                                   text="",
                                   font_size=self.font_size["msg"],
                                   color=[0.8, 0.45, 1],
                                   anchor='center', 
                                   on=False)
        
        # STIMULI TEXTURES
        self.textures = list()
        self.desired_actions_textures = list()
        self.desired_actions_textures2 = list()
        self.foo_textures = list()
           
        for i in xrange(len(self.images_set)):
            
            full_path = os.path.dirname(__file__) + "\\" + self.images_set[i]

            foo = TextureStimulus( texture=Texture(full_path) )
                
            h = float(foo.parameters.texture.size[1]) 
            w = float(foo.parameters.texture.size[0])
            
            if (h < maxsize and w < maxsize):
                h2 = maxsize
                w2 = h2*(w/h)
            elif(h>maxsize and w>maxsize):
                if(h>=w):
                    h2 = maxsize
                    w2 = h2*(w/h)
                else:
                    w2 = maxsize
                    h2 = w2*(h/w)
                                
            
            self.textures.append(TextureStimulus( texture=Texture(full_path),
                                                            size=(w2,h2),
                                                            position=[0,0], 
                                                            anchor='center', 
                                                            max_alpha=1.0,
                                                            internal_format = gl.GL_RGBA,
                                                            on = False))
            
            self.desired_actions_textures.append(TextureStimulus( texture=Texture(full_path),
                                                            size=(w2*0.8,h2*0.8),
                                                            position=[0,0], 
                                                            anchor='center', 
                                                            max_alpha=1.0,
                                                            internal_format = gl.GL_RGBA,
                                                            on = False))  
            
            self.desired_actions_textures2.append(TextureStimulus( texture=Texture(full_path),
                                                            size=(w2*0.8,h2*0.8),
                                                            position=[0,0], 
                                                            anchor='center', 
                                                            max_alpha=1.0,
                                                            internal_format = gl.GL_RGBA,
                                                            on = False))  
        for i in xrange(len(self.foo_images_set)):
            
            full_path = os.path.dirname(__file__) + "\\" + self.foo_images_set[i]           
            
            self.foo_textures.append(TextureStimulus( texture=Texture(full_path),
                                                            size=(100,100),
                                                            position=[0,0], 
                                                            anchor='center', 
                                                            max_alpha=1.0,
                                                            internal_format = gl.GL_RGBA,
                                                            on = False))
            
         
        
    def pre_mainloop(self):
        
        self.suc = 0
        self.err = 0
        self.miss = 0
        self.start = clock()                                    
        self._confirmed = False                
        self._finished = False  
               
        self._classified_action = -1
        self._classified_element = -1                            
        
        self._spelled_actions = []
        
        if not self.free:
            self._actions_to_select = FooMenuElement("", None);
        else:
            self._actions_to_select  = None
                      
        self._current_action = 0;                                                            
        self._current_sequence = 0          # Index of current sequence
        self._current_stimulus = 0          # Index of current stimlus
        self._current_countdown = 0
                        
        self._state = self.STATE_IDLE
        
        self._ve_elements = []              ## init containers for VE elements:     
        
        self.prepare_mainloop()             ## call subclass-specific pre_mainloop:
        
        self.__init_screen()                ## build screen elements
            
        ## send start trigger:
        self.send_parallel(self.RUN_START)
        self.logger.info("[TRIGGER] %d - %.3f" % (self.RUN_START, clock()))
        
    
    
    def post_mainloop(self):  
                        
        pygame.time.wait(100)

        if(not self.free):            
            self.send_parallel(self.RUN_END)
            self.logger.info("[TRIGGER] %d - %.3f" % (self.RUN_END, clock()))
            
        pygame.time.wait(100)
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
        
        self._spelled_actions = FooMenuElement("", None); 
        self._ve_spelled_actions = [];        
        self.init_textures(100.0)
        
        ## create countdown:
        self._ve_countdown = Text(position=self._centerPos,
                                  text=" ",
                                  font_size=self.font_size_countdown,
                                  color=self.countdown_color,
                                  anchor='center')  
        
        ## create letter box on top:
        if not self.free:
            print("init screen in not free mode: %d elems" %(len(self.desired_actions)))
            self._ve_letterbox = Target2D(position=(self._centerPos[0], self.screenPos[3] * (1 - 0.01) - self.letterbox_size[1]/2. - 28),
                                         size=(110, 110),
                                         color=self.letterbox_color)
                      
            ve_innerbox = Target2D(position=(self._centerPos[0], self.screenPos[3] * (1 - 0.01) - self.letterbox_size[1]/2.  - 28),
                                   size=(110-9, 110-9),
                                   color=self.bg_color)
        
            self._current_letter_position = (self._centerPos[0], self.screenPos[3] * (1 - 0.015) - self.letterbox_size[1]/2.)                                           
                
            ## add symbol box to elementlist:

            self._ve_elements.extend([self._ve_letterbox, ve_innerbox])
                     
            
            desired = list(self.desired_actions)
            random.shuffle(desired)
        
            for i in xrange(len(self.desired_actions)):                
                t = self.desired_actions_textures[desired[i]]
                
                full_path = os.path.dirname(__file__) + "\\" + self.images_set[desired[i]]
                foo = TextureStimulus( texture=Texture(full_path) )
                
                h = float(foo.parameters.texture.size[1]) 
                w = float(foo.parameters.texture.size[0])
                maxsize = 100.0
                if (h < maxsize and w < maxsize):
                    h2 = maxsize
                    w2 = h2*(w/h)
                elif(h>maxsize and w>maxsize):
                    if(h>=w):
                        h2 = maxsize
                        w2 = h2*(w/h)
                    else:
                        w2 = maxsize
                        h2 = w2*(h/w)
                    
                t = TextureStimulus( texture=Texture(full_path),
                                                            size=(w2,h2),
                                                            position=[0,0], 
                                                            anchor='center', 
                                                            max_alpha=1.0,
                                                            internal_format = gl.GL_RGBA,
                                                            on = False)     
                elem = MenuElementImage( id = desired[i],
                                         parent=self._actions_to_select,
                                         position=(self._centerPos[0] + 100*(i+1) + self.letterbox_size[0]/2. - 130, self._current_letter_position[1] - 20),
                                         texture = t);
                                                 
                elem.texture.set(on = True)
                elem.set_max_alpha(0.4)
                                                                                    
                self._actions_to_select.append(elem);
                self._ve_elements.append(elem.texture)                                            

            self._actions_to_select.subelems[self._current_action].set_max_alpha(1.0)
                 
        ## create shapes and letters:
        self.init_screen_elements()
        
        ## add remaining elements to element list:     
        self._ve_elements.extend([self._ve_countdown, self._message])
        
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
        if self._state == self.STATE_COUNTDOWN:
            self.start_time = clock()
            self.pre__countdown()
            self.__countdown()
            self.post__countdown()
        elif self._state == self.STATE_TRIAL:
            self.pre__trial()
            self.__trial()
            self.post__trial()
        elif self._state == self.STATE_CLASSIFY:
            self.pre__classify()
            self.__classify()
            self.post__classify()
        elif self._state == self.STATE_FEEDBACK:
            self.pre__feedback()
            self.__feedback()
            self.post__feedback()
        else:
            self.pre__idle()
            self.__idle()
            self.post__idle()
        self.post_play_tick()
    
    
    
    def __stop(self, *args):
        self.on_stop()
    
    
    
    def __show_symbols(self, symbols, text):
       
        self._message.set(on=True, text=self._message_txt)
                       
        if(self._levels is None):
            self._levels = LevelContainer(self.nCountdown)               
                    
            if(self.selection_mode == "P300"):
                m = LevelBase.P300;
            elif(self.selection_mode == "MI"):
                m = LevelBase.MI; 
            elif(self.selection_mode == "MISync"):
                m = LevelBase.MI_SYNC;
            else:
                m = LevelBase.LIN_CTRL;
                
            t = list()
                  
            for i in xrange(len(symbols)):
                t.append(self.textures[symbols[i]]);
            
            if(self.with_emergency_exit):
                t.append(self.textures[6])                   # emergency break
            
            nelems = len(t) 
            
            if (len(t) < self.n_elems_seq):
                for i in xrange(self.n_elems_seq - len(t)):
                    t.append(self.foo_textures[i]);       
                   
            if(len(symbols)>1):
                self._levels.AddLevel(self._centerPos, self.stimuli_colors, m, t, self.speller_radius, self._ve_elements,  self.confirmation_text, self.font_size["msg"], self.color_msg, nelems, self.fbclasses, self.screenPos[2] )
            else:
                self._levels.AddLevel(self._centerPos, self.stimuli_colors, m, t, self.speller_radius, self._ve_elements,  "", self.font_size["msg"], self.color_msg, 2, self.fbclasses, self.screenPos[2])
            
            t = list()
            
            ## Creates the container for the "confirmation" level (if necessary)
            if(self.with_confirmation and (len(symbols)>1)):
                if(self.confirmation_mode == "P300"):
                    m = LevelBase.P300;
                    t.append(self.textures[10])
                    t.append(self.textures[11])
                    
                    if(self.with_emergency_exit):
                        t.append(self.textures[6])              # emergency break
                    
                    nelems = len(t) 
                        
                    t.append(self.foo_textures[0])
                    t.append(self.foo_textures[1])                                                   
                    t.append(self.foo_textures[2])
                    t.append(self.foo_textures[3])  
                elif (self.confirmation_mode == "MI"):                
                    m = LevelBase.MI;                     
                    t = [self.textures[10], self.textures[11]];
                    if(self.with_emergency_exit):
                        t.append(self.textures[6])              # emergency break
                    
                    nelems = len(t)
                else:
                    m = LevelBase.MISync;                     
                    t = [self.textures[10], self.textures[11]];
                    if(self.with_emergency_exit):
                        t.append(self.textures[6])              # emergency break
                    
                    nelems = len(t)
                
                self._levels.AddLevel(self._centerPos, self.stimuli_colors, m, t, self.speller_radius, self._ve_elements, "" , self.font_size["msg"], self.color_msg, nelems, self.fbclasses, self.screenPos[2])
    
            if(self._levels.GetMode() == LevelContainer.MI):
                self._presentation.set(go_duration=(0.05, 'seconds'))
      
        self._classified_element = -1; 
        self._state = self.STATE_COUNTDOWN
                  
        if(self.num_trials % self._pause_after == 0):
            self._current_countdown = self._pause_time
        else:
            self._current_countdown = self._levels.GetCountDown()
                
                        
        
    def __idle(self):

        if(self.free):
            if(self._levels is not None or self.show_command_recieved):
                self.__show_symbols(self._actions_to_select2, "")
                self.show_command_recieved = False
                return                
                                    
            self._presentation.set(go_duration=(0.5, 'seconds'))
            self._presentation.go()
        else:
            self.__show_symbols(self.actions_available, "")        
        
            
            
    def __countdown(self):
               
        if (self._current_countdown == self._levels.GetCountDown() or self._current_countdown == self._pause_time):
            self.send_parallel(self.COUNTDOWN_START)       
            self.logger.info("[TRIGGER] %d - %.3f" % (self.COUNTDOWN_START, clock()))
            self.set_countdown_screen()
            self._ve_countdown.set(on=True)
            self._presentation.set(go_duration=(1, 'seconds'))            
            
        self._ve_countdown.set(text="%d" % self._current_countdown)
        self._presentation.go()
        self._current_countdown = self._current_countdown-1
        
        if self._current_countdown == 0:            
            
            self._ve_countdown.set(on=False, text=" ")                   
            self.set_standard_screen()
                
            pygame.time.wait(20)
            
            self._state = self.STATE_TRIAL
            self._first_trial_tick = True
            self.trial_elapsed_time = 0
            self.logger.info("[TRIGGER] %d - %.3f" % (self._levels.GetBeginLevelMarker(), clock()))    
            self.send_parallel(self._levels.GetBeginLevelMarker())
                                         
     
    
    
    def __trial(self):
        if(self._levels.GetMode() == LevelContainer.P300):
            self.__trial_P300()
        else:
            
            if(self._first_trial_tick):
                self.trial_elapsed_time = 0                                            
                self._first_trial_tick = False
                self._presentation.set(go_duration=(1, 'seconds'))
                self._presentation.go()
                self._classified_element = -1
                self.last_dt = clock()
                self.timeout = False
            else:                 
                self.__trial_MI()                   
                                                                                          

            
    
    def __trial_MI(self):     

        self._presentation.set(go_duration=(0.03, 'seconds'))
        self._presentation.go()
            
        if self._classified_element == -1:
                                
            self._classified_element = self._levels.update(clock())
            self.trial_elapsed_time +=  clock() - self.last_dt;
            self.last_dt = clock()
                
            if(self.trial_elapsed_time > self.trial_timeout):
                if(not self.timeout):
                    self.timeout = True
                    self.miss += 1
                        
                    self.send_parallel(self._levels.GetTimeoutMarker())
                    self._state = self.STATE_FEEDBACK
                    print("TIMEOUT!!!!")
        else:
            # send trigger:       
            self.logger.info("[TRIGGER] %d - %.3f" % (self._levels.GetEndLevelMarker(), clock()))      
            self.send_parallel(self._levels.GetEndLevelMarker())
            # next state
            self._state = self.STATE_CLASSIFY                                                            

            

    def __trial_P300(self):
        
        if self._current_sequence==0 and self._current_stimulus==0:
            self.nnn = 0;
            # generate random sequences:
           
            self.flash_sequence = []
            for _ in range(self.nr_sequences):
                random_flash_sequence(self,
                                        set = range(self.n_elems_seq),
                                        min_dist = self.min_dist,
                                        seq_len = self.n_elems_seq)
                pygame.time.wait(166)

                                 
        currentStimulus = self.flash_sequence[self._current_sequence*self.n_elems_seq + self._current_stimulus]
        
        # set stimulus:
        self.stimulus(currentStimulus, True)
     
        # check if current stimulus is target and then send trigger:
        target_add = 0
        
        # In calibration, we stop when we finish all the "desired actions"
        if not self.free:
            if (self._current_action < len(self._actions_to_select.subelems)):
                if self._levels.GetState() == LevelContainer.STATE_SELECT:
                    if self._actions_to_select.subelems[self._current_action].id == self._levels.get_elem(currentStimulus).id:                    
                        # current stimulus is target group:
                        target_add = self.TARGET_ADD
                             
        marker = self._levels.GetMarker(currentStimulus) + target_add
        self.send_parallel(marker)
        self.logger.info("[TRIGGER] %d" % marker)
            
        # present stimulus:
        self._presentation.set(go_duration=(self.stimulus_duration, 'seconds'))
        self._presentation.go()

        # reset to normal:           
        self.stimulus(currentStimulus, False)

        # present interstimulus:
        self._presentation.set(go_duration=(self.interstimulus_duration, 'seconds'))
        self._presentation.go()

        self.nnn = self.nnn + 1;
        # increase  
        self._current_stimulus = (self._current_stimulus+1) % self._levels.GetNumElems()
        if self._current_stimulus == 0:
            self._current_sequence = (self._current_sequence+1) % self.nr_sequences

        # check for end of trial:
        if self.nnn == self.n_elems_seq*self.nr_sequences:
            
            # send trigger:           
            self.send_parallel(self._levels.GetEndLevelMarker())
                
            # next state
            self._state = self.STATE_CLASSIFY
            #self._classified_element2 = self._classified_element
            
            if( self.with_emergency_exit and (self._classified_element == self._levels.GetNumRealElements()-1)):
                self.__emergency_break()


    
    def __emergency_break(self):
        print("####################################### EMERGENCY BREAK SELECTED #######################################")
        self.send_parallel(self.EMERGENCY_BREAK)
        self.logger.info("[TRIGGER] %d" % self.EMERGENCY_BREAK)        
        self._finished = True   
        self.on_stop()  
        

        
    def __classify(self):

        print("classify self._classified_element= %d, self._classified_action =%d" %(self._classified_element, self._classified_action))            
        if self.offline:          
      
            if(self._levels.GetState() == LevelContainer.STATE_SELECT):             
                self._classified_element = self._actions_to_select.subelems[self._current_action].id
                print("_classify offline _classified_element=%d" % (self._classified_element))                
            elif(self._levels.GetState() == LevelContainer.STATE_CONFIRM):
                self._confirmed = True                                       
                            
        else:
            if(self._levels.GetMode() == LevelContainer.P300):
                i = 1
                while((self._classified_element == -1) and (i<30)):                
                    i = i + 1
                    self._presentation.set(go_duration=(0.2, 'seconds'))
                    self._presentation.go()  
                    print "Waiting for classifier output..."
            #else:
            #   self._classified_element = self._classified_element2
               
            if self._classified_element > -1:           
                print "\n**** Class: %d \n" % (self._classified_element+1)
                    
                if(self._levels.GetState() == LevelContainer.STATE_CONFIRM):
                    if (self._classified_element == 0):
                        self._confirmed = True
                    else:
                        self._confirmed = False
                        
                else:        
                    self._classified_action = self._classified_element
                                                                        
                print("current_level_state = %d, classified = %d, confirmed = %d" % (self._levels.GetState(), self._classified_element, self._confirmed))
                error_add = 0
        
                ## evaluate classification:
                if(self._levels.GetState() == LevelContainer.STATE_SELECT):            
                    if not self.free:                        
                        if (self._levels.get_elem(self._classified_element).id != self._actions_to_select.subelems[self._current_action].id):
                            error_add = self.ERROR_ADD
                            self.err += 1
                        else:
                            self.suc += 1
                    
                    self.send_parallel(self._levels.GetResponseMarker(self._classified_element) + error_add)
                             
                elif(self._levels.GetState() == LevelContainer.STATE_CONFIRM):           
        
                    # send confirmation trigger
                    #if(self._classified_element < self._levels.GetNumRealElements()):
                        self.send_parallel(self._levels.GetResponseConfirmationMarker(self._confirmed, self._classified_element))
                        
                        # check if the correct action was selected
                        if not self.free: 
                            selected_action =  self._levels.get_level_elem(0,self._classified_action).id
                            
                            print ("----> classif = %d, desired = %d" % (selected_action, self._actions_to_select.subelems[self._current_action].id))
                                                
                            if (self._current_action < len(self._actions_to_select.subelems)) and selected_action != self._actions_to_select.subelems[self._current_action].id and self._confirmed:
                                error_add = self.ERROR_ADD
                                self.err = self.err + 1
                                print("ERROR: WRONG action selected");
                            elif (self._current_action < len(self._actions_to_select.subelems)) and selected_action == self._actions_to_select.subelems[self._current_action].id and not self._confirmed:
                                error_add = self.ERROR_ADD
                                self.err += 1
                                print("ERROR: CORRECT action CANCELED");
                            else:
                                self.suc += 1
                                print("SUCCESS: CORRECT action selected and CONFIRMED or WRONG action CANCELED");   
                                
                            self.send_parallel(self._levels.GetResponseMarker(self._classified_element) + error_add)                                                      
            else:
                print("ERROR: NO classifier output recieved !!!!!")
                self._finished = True     
                self.on_stop()
       
        print("    ### Hits = %d Misses = %d" % (self.suc, self.err))     
        self._state = self.STATE_FEEDBACK



    def __feedback(self):        
        
        if(self.timeout):
            pygame.time.wait(700)        
            self._levels.set_on(False)         
            self._message.set(on=True, text=("Miss"))                       
            self._presentation.set(go_duration=(self.feedback_attr["duration"], 'seconds'))
            self._presentation.go()                            
            self._message.set(on=False)                
        
            pygame.time.wait(200)
        else:
            if(not (self.free and (self._levels.GetState() == LevelContainer.STATE_SELECT) and (self._classified_element >= self._levels.GetNumRealElements()) )):
                self.feedback()
                                   
        if (( (self._levels.GetState() == LevelContainer.STATE_SELECT and not self.with_confirmation) or (self._levels.GetState() == LevelContainer.STATE_CONFIRM and self.with_confirmation and self._confirmed) )): # only update, if we are at the end of level 2, or at the end of level 1 without confirmation                     
                                
            e = copy.copy(self._levels.get_elem(self._classified_element))        
            
            self._spelled_actions.append(e);
            self._ve_spelled_actions.append(e);
            
            self._current_action = self._current_action + 1
          
            if not self.free:
                if self._current_action >= self._actions_to_select.num_elements():
                    self._finished = True

                    if not self.offline:          
                        if(self.trial_timeout>0):
                            self._message.set(on=True, text=("END: Hits = %d Error = %d Misses = %d" % (self.suc, self.err, self.miss)))
                        else:
                            self._message.set(on=True, text=("END: Hits = %d Error = %d" % (self.suc, self.err)))
                            
                        #self._levels.show_message(True)
                        self._presentation.set(go_duration=(8, 'seconds'))
                        self._presentation.go()  

                    self.on_stop()            
                else:
                    for i in xrange(self._actions_to_select.num_elements()):
                        self._actions_to_select.subelems[i].original_pos = (self._actions_to_select.subelems[i].original_pos[0]-100, self._actions_to_select.subelems[i].original_pos[1])
                        self._actions_to_select.subelems[i].texture.set(position=self._actions_to_select.subelems[i].original_pos, max_alpha=0.4);                          
                
                    self._actions_to_select.subelems[self._current_action].texture.set(max_alpha=1.0)                                  
             
        # set new level:
                
        if(self.free):
            if(self.with_confirmation):
                if(self._levels.GetState() == LevelContainer.STATE_SELECT):
                    if((self._classified_element < self._levels.GetNumRealElements())):
                        if((self._levels.GetNumRealElements()>2)):
                            self._levels.SetState(LevelContainer.STATE_CONFIRM)
                        else:
                            self.send_parallel(self.RUN_END)                                                                                
                            self._levels= None 
                                                 
                elif(self._levels.GetState() == LevelContainer.STATE_CONFIRM):                   
                    if(self._confirmed): 
                        self.send_parallel(self.RUN_END)                                                                                
                        self._levels= None                 
                    else:
                        self._levels.SetState(LevelContainer.STATE_SELECT);  
                        
                    self._confirmed = False     
            else:
                self._levels.SetState(LevelContainer.STATE_SELECT);
        else:
            if(self.with_confirmation):
                if(self._levels.GetState() == LevelContainer.STATE_SELECT):                                        
                    self._levels.SetState(LevelContainer.STATE_CONFIRM);                                   
                elif(self._levels.GetState() == LevelContainer.STATE_CONFIRM):                                                                                                                                               
                    self._levels.SetState(LevelContainer.STATE_SELECT);                                                                             
            else:
                self._levels.SetState(LevelContainer.STATE_SELECT);
      
        #print("    TRIAL EXECUTION TIME: %.2f\n" % (clock() - self.start_time)*1000)
             
        self._state = self.STATE_IDLE
        
        self._classified_element = -1        
        self.start_time = clock()
        self.step = 0
        self.num_trials += 1
    

    def keyboard_input(self, event):
        
        if event.key == pygame.K_ESCAPE: 
            self.__emergency_break()     
        elif event.key == pygame.K_a:
            self.on_control_event({'Show': '''Drink,Brush,Touch face with a sponge,Touch a body part,Touch another person,Press a button'''})
        elif event.key == pygame.K_s:
            self.on_control_event({'Show': 'Grasp cup'})
        elif event.key == pygame.K_d:
            self.on_control_event({'Show': 'Stop drinking'}) 
        elif event.key == pygame.K_f:
            self.on_control_event({'Show': 'clear'})       
        elif event.key == pygame.K_g:
            self.on_control_event({'ShowText': 'hola'})       
        elif self.debug:                
            if not self.offline:
                if(self._levels is not None):
                    if(self._levels.GetMode() == LevelBase.P300):
                        if (event.key >= pygame.K_0 and event.key <= pygame.K_9):
                            self.on_control_event({'cfy_erp': int(chr(event.key))})
                    else:
                        if (event.key >= pygame.K_0 and event.key <= pygame.K_9):
                            if(int(chr(event.key)) == 1):
                                self.step -= 0.1       
                                self.on_control_event({'cfy_best_comb': self.step})                        
                            elif(int(chr(event.key)) == 2):
                                self.step += 0.1
                                self.on_control_event({'cfy_best_comb': self.step})
                            elif(int(chr(event.key)) == 3):
                                self.step_idle -= 0.1
                                self.on_control_event({'cfy_idle': self.step_idle})
                            elif(int(chr(event.key)) == 4):
                                self.step_idle += 0.1
                                self.on_control_event({'cfy_idle': self.step_idle})
                            
                          

#    "images_set": ["glass_straw_bw.png","scratch.png","brush.png","sponge.png","help.png","handshake.png","exit.png", "grasp.png", "back.png", "repeat.png", "ok.png", "cancel.png"],
 

    #<Action>Drink</Action> "glass_straw_bw.png"                    0
    #<Action>Brush</Action> "brush.png"                             2
    #<Action>Touch face with a sponge</Action> "sponge.png"         3
    #<Action>Touch a body part</Action> "scratch.png"               1
    #<Action>Touch another person</Action> "handshake.png"          5
    #<Action>Press a button</Action> "help.png"                     4

    def translate_string_to_action(self, s):
        if(s == "Drink"):
            return 0;
        elif(s == "Brush"):
            return 2;
        elif(s == "Touch face with a sponge"):
            return 3;
        elif(s == "Touch a body part"):
            return 1;
        elif(s == "Touch another person"):
            return 5;
        elif(s == "Press a button"):
            return 4;
        elif(s == "Exit"):
            return 6;
        elif(s == "Grasp cup"):
            return 7;
        elif(s == "Grasp sponge"):
            return 7;
        elif(s == "Grasp brush"):
            return 7;
        elif(s == "Go back to rest"):
            return 8;
        elif(s == "Repeat action"):
            return 9;        
        elif(s == "Stop touching face with the sponge" or s == "Stop drinking" or s == "Stop brushing" or s =="Stop touching another person" or s=="Stop touching body part"):
            return 10;
        else:
            return -1;
        
    
    def on_control_event(self, data, keyboard = False):
    
        #self.logger.info("[CONTROL_EVENT] %s" % str(data))                                      
        if data.has_key('erp_output'):
                        
            # classification output was sent:
            if(self._levels.GetMode()==LevelContainer.P300):
                
                self._classified_element = data['erp_output'] - 1
                        
                print("############################### [CONTROL_EVENT] class %d " % self._classified_element)                                   
        elif data.has_key('cfy_erp') :
                        
            # classification output was sent:
            if(self._levels.GetMode()==LevelContainer.P300):
                

                self._classified_element = data['cfy_erp'] - 1
                        
                print("############################### [CONTROL_EVENT] class %d " % self._classified_element)                                          
        elif data.has_key('Show'):
            print("\n###############################  SHOW \n");            
            if data['Show'] != 'clear':
                s = []
                reader = csv.reader([data['Show']], skipinitialspace=True)
                for r in reader:
                    for e in r:
                        
                            t = self.translate_string_to_action(e)
                            if t>=0:
                                print("   '%s' = %d!!" % (e,t))
                                s.append(t)
                            else:
                                print("ERROR: no translation for '%s' found!!" % (e))
                
                self._actions_to_select2 = s
                
                if(len(s)==1):
                    self._actions_to_select2 = [10]
                
                self.show_command_recieved = True
            else:
                self._message_txt = ' '
                if(self._message is not None):
                    print("on_control_event: message set ''")
                    self._message.set(on=False, text=' ')

                          
        elif data.has_key('ShowText'):
            print("\n###############################  SHOW TEXT: '%s' \n" % data['ShowText']);
            self._message_txt = data['ShowText']
            if(self._message is not None): 
                self._message.set(on=True, text=data['ShowText'])
               
        else:
           
            if self._state == self.STATE_TRIAL:
                if(self._levels is not None):
                    if(self._levels.GetMode()!=LevelContainer.P300):
                        #print("cl_output = %2f" % (data["cl_output"] ))
                        for key in data.keys():
                        #    print "key: %s , value: %s" % (key, data[key])

                            self._levels.setClassifierOutput(key, data[key])    
    
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
    

def animate_sigmoid(pos_start, pos_end, dt, accel=10.):
    return NP.add(pos_start, NP.divide(NP.subtract(pos_end, pos_start), (1. + NP.exp(-accel*dt + 0.5*accel))))
