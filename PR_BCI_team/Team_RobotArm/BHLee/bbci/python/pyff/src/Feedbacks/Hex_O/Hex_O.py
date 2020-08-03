#!/usr/bin/env python
 
# Hex_O.py -
# Copyright (C) 2009  Maria Kramarek
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

from VisionEgg import *
from VisionEgg.WrappedText import WrappedText
import numpy as np
import pylab as p
from Figure_Many import *
from Top_Letters import *
from Figure_One import *
from VisionEgg.FlowControl import Presentation, FunctionController
from VisionEgg.Core import *
from VisionEgg.MoreStimuli import *
from VisionEgg.Text import *
#from lib.eyetracker import EyeTracker
from FeedbackBase.MainloopFeedback import MainloopFeedback
       
class Hex_O(MainloopFeedback):
    # TRIGGER VALUES FOR THE PARALLEL PORT (MARKERS)
    START_EXP, END_EXP = 252, 253
    COUNTDOWN_START = 0
    #different from pilot - here ALWAYS the start of trial animation is 36 
    START_ANIMATION = 36  
    START_TRIAL = 37
    SHORTPAUSE_START = 249
    INVALID = 66
    DISPLAY_START = 45
    WELCOME = 150

    def init(self):
        """ initializes all the settings of the figure(s)"""
        
        #variables to be set constantly by matlab
        #self.classifier = np.zeros((10, 6))
        self.classifier = [] #np.random.randn(10, 6)
        #self.classifier = np.asarray(self.classifier)
        
        self.f_index= [0, 2, 3, 4, 5, 6]
        self.classifier2 = []
        self.avg_class = 0
        #settings about the word to be spelled and its display
        self.word_to_spell_over = "Everything_has_got_a_moral,_if_only_you_can_find_it"
        self.word_to_spell_cover = "__________________________________________________"
        

        self.top_word_main_color = (0.0,0.0,0.0)
        self.top_word_special_color = (1.0,0.0,0.0)  
        self.top_word_font_size = 18
        self.display_font_size = 75
                               
        #settings about the figures to be drawn
        self.no_of_figures = 6
        self.no_of_corners_in_figure = 6 
        self.radius_of_figure = 50  
        self.distance_between_figures = 150 
        self.color_of_figure_on = (1.0,1.0,1.0)
        self.color_of_figure_off = (0.0,0.0,0.0)
        self.one_letter_font_size = 18 
        self.many_letters_font_size = 18
        self.size_of_change = 5
                
        #settings about countdown
        self.color_countdown = (1.0,1.0,1.0)   
        self.round_count_down = 1    
        self.count_down_font_size = 200  
        self.class_dot_color = (1.0,1.0,1.0)
        self.class_dot_on = False
                        
        #other settings about display
        self.color_of_background = (0.5,0.5,0.5)  
        #self.add_fixation_spot_in_center = True  
        self.fixation_spot_color = (255,0,0,0) 
        self.fixation_spot_size = (8, 8)
        self.error_message = "Look in the middle"
        self.error_message2 = "Look at the target"
        self.overt_message = "Now you will see 6 hexagons. Each time the letter will be shown - you must LOOK at the hexagon displaying that letter, if not, error message will be shown and you will have to repeat the trial. When you are ready click any button and good luck"""
        self.covert_message = "Now you will see 6 hexagons. Each time the letter will be shown - you must look in the middle and CONCENTRATE at the hexagon displaying that letter, if not, error message will be shown and you will have to repeat the trial. When you are ready click any button and enjoy"""

        self.animate_hex = True
                
        #settings of monitor  
        self.screen_size = [800, 600]  #[width, height]
        self.monitor_refresh_rate = 120  
        self.same_screen = True
        self.full_screen = False #not adequate if two screens used  
        self.platform = "windows" # "linux"

        #settings of the frequencies & timing
        self.time_of_trial = 10 #given in seconds
        self.time_of_pause = 0
        self.time_of_count_down = 5
        self.time_word_display = 1
        self.flicker_rate = [4.0, 6.0, 7.5, 8.0, 10.0, 12.0]     
        self.time_animate = 4.5     
        
        # Eye tracker
        self.use_eyetracker = False
        self.et_fixate_center = True   # Whether center or fixdot 
                                       #has to be fixated   
        self.et_currentxy = (0, 0)      # Current fixation
        self.et_duration = 100
        self.et_targetxy = (100, 100)       # Target coordinates
        self.et_range = 200        # Maximum acceptable distance between target
                                   # and actual fixation (200)
        self.et_range_time = 200    # maximum acceptable fixation off the 
                                    #designated point 

        #self.center_figure_positions
        #self.figure_to_be_used
        self.figure_to_be_used = 0
        self.next_letter = "_"
        
    def pre_mainloop(self):
        """ prepares the feedback for being run """  
        #settings not to be changed normally
        if self.use_eyetracker:
            self.et = EyeTracker()
            self.et.start()
        if self.et_fixate_center == True:
            self.word_to_spell = self.word_to_spell_cover
            self.add_fixation_spot_in_center = True 
        else:
            self.word_to_spell = self.word_to_spell_over 
            self.add_fixation_spot_in_center = False 

        self.classifier_reset = 0                 
        self.number_of_trials = len(self.word_to_spell)
        self.trial_number = 0
        self.display_full = 1
        self.presentation_stopped = False
        self.another_trial = self.number_of_trials
        self.number_of_trials = len(self.word_to_spell)
        self.next_to_display = self.word_to_spell 
        self.use_letters = "abcdef"
        self.next_letter = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ><-&' 
        self.present_count_down = None
        self.time_used = 0
        self.animation_time = 0.0
        self.cross_size = self.fixation_spot_size
        self.play_ok = True
        self.look_in_the_middle = False
        if self.time_of_count_down > 0:
            self.countdown = True
        else:
            self.countdown = False  
        
        #initialize everything         
        self.init_graphics()
        self.shortBreakInit()
        self.send_parallel(Hex_O.START_EXP)
        self.welcome()
        #self.sound_invalid = pygame.mixer.Sound("Feedbacks\P300\winSpaceCritStop.wav")
        

    def post_mainloop(self):
        """ end the whole experiment """
        self.send_parallel(Hex_O.END_EXP)
        self.on_quit()
        
    def tick(self):
        print "tick"
        
    def play_tick(self):
        """ main experiment control"""
        print "play tick"
        if self.countdown:
            self.play_ok = False
            #self.play_sound("windVinciSysStart.wav")
            self.start_counting() 
            self.play_ok = True           
        else:
            #shows the full display
            if self.display_full == 1:
                #self.show_display()
                self.display_full = 0
 
            if self.another_trial > 0:
                #self.classifier2 = []
                self.classifier = [] #np.random.randn(10, 6)
                #self.classifier = np.asarray(self.classifier)
                self.many_letters_trial()
                #print self.avg_class, "class average"
                self.calc_average()
                self.another_trial = self.another_trial - 1
                #self.classifier2 = []
                #print self.avg_class, "class average"
                self.classifier = [] #np.random.randn(10, 6)
                #self.classifier = np.asarray(self.classifier)
                self.one_letter_trial()

                
            if self.another_trial == 0:   
                self.next_to_display = "the end"
                self.show_display()          
                self.post_mainloop()
                #self.play_sound("winSpaceClose.wav")
            #self.word_to_spell[self.trial_number] = self.next_letter
            self.trial_number = self.trial_number + 1
                
    def many_letters_trial(self):
        """ display the text and the figures for the many letters trial """

        #self.figure_to_be_used = self.use_hex
        #self.use_letters = self.letters_in_each[self.figure_to_be_used]

        if self.use_eyetracker:
            # Determine whether fixate center or fixate target
            if self.et_fixate_center:
            # Fixate center
                self.et_targetxy = (self.screen_size[0] / 2, self.screen_size[1] / 2)
            else:
                # Fixate target (change the positions of the target in runtime)
                self.et_targetxy = self.center_figure_positions[self.figure_to_be_used]
                #self.presentation_stopped
        
        self.presentation_stopped = True
        while self.presentation_stopped == True:
            self.look_in_the_middle = False
            print "here again"
            self.presentation_stopped = False
            
            #self.next_to_display = self.word_to_spell[self.trial_number]
            #self.show_display()
            #self.figure_many.go_duration(self.time_animate, 'seconds')
            #self.time_used = self.time_animate
            #self.figure_one.parameters.go_duration = (0,'frames')
            self.animation_time = self.time_animate
            self.cross_size = [0, 0]        
            self.send_parallel(Hex_O.START_ANIMATION)
            self.figure_many.parameters.go_duration = (self.time_animate, 'seconds') 
            self.figure_many.go()
            self.animation_time = 0.0
            self.cross_size =  self.fix_size
            self.send_parallel(Hex_O.START_TRIAL)
            self.look_in_the_middle = True
            self.figure_many.parameters.go_duration = (self.time_of_trial, 'seconds') 
            self.figure_many.go()
            #self.presentation_stopped = False
        
    def one_letter_trial(self):
        """ display the text and the figures for the one letter trial """

                #self.presentation_stopped
                
        #self.show_display() 
        #letter_index = self.next_letter.find(self.next_to_display.upper())
        #self.figure_to_be_used = int(letter_index / (self.no_of_corners_in_figure-1))
        self.figure_to_be_used = self.use_hex
        self.use_letters = self.letters_in_each[self.figure_to_be_used]
        
#        self.send_parallel(Hex_O.START_TRIAL_ANIMATION)                
#        self.figure_one.go()

        if self.use_eyetracker:
            # Determine whether fixate center or fixate target
            if self.et_fixate_center:
            # Fixate center
                self.et_targetxy = (self.screen_size[0] / 2, self.screen_size[1] / 2)
            else:
                # Fixate target (change the positions of the target in runtime)
                self.et_targetxy = self.center_figure_positions[int(letter_index % (self.no_of_corners_in_figure-1))]

        self.presentation_stopped = True
        while self.presentation_stopped == True:
            self.look_in_the_middle = False
            self.presentation_stopped = False
            self.animation_time = self.time_animate
            self.cross_size = [0, 0]
            self.send_parallel(Hex_O.START_ANIMATION)
            self.figure_one.parameters.go_duration = (self.time_animate, 'seconds') 
            self.figure_one.go()
            self.animation_time = 0.0
            self.cross_size = self.fix_size
            self.look_in_the_middle = True
            self.send_parallel(Hex_O.START_TRIAL)
            self.figure_one.parameters.go_duration = (self.time_of_trial, 'seconds') 
            self.figure_one.go()
            #self.presentation_stopped = False
        #self.next_to_display = self.center_figure_positions[int(self.use_hex % (self.no_of_corners_in_figure-1))]
        #print self.next_to_display
        self.calc_average()
        self.next_to_display = self.use_letters[self.use_hex]
        self.next_letter = self.next_to_display
        print self.next_to_display
        self.show_display()
        self.short_pause_tick()

    def welcome(self):
        """this will display the welcome message"""
        if self.et_fixate_center == True:
            msg = self.covert_message
        else:
            msg = self.overt_message
            
        text = WrappedText(text=msg,
                    color=self.color_countdown, # alpha is ignored (set with max_alpha_param)                    position=(self.screen.size[0]/2,self.screen.size[1]/2),
                    font_size= self.display_font_size,
                    position=(10,self.screen.size[1]-10))
        
        viewport = Viewport(screen=self.screen,
                    size=self.screen.size,
                    stimuli=[text])
        
        self.welcome_you = Presentation(go_duration=('forever',),viewports=[viewport])
        self.welcome_you.parameters.handle_event_callbacks = [(pygame.locals.KEYDOWN, self.keydown)] 
#        text_controller = FunctionController(\
#            during_go_func = self.button_controller,\
#            eval_frequency = Controller.EVERY_FRAME,\
#            temporal_variables = \
#            Controller.TIME_SEC_SINCE_GO)
#        self.welcome_you.add_controller(text,'font_size', text_controller)
        self.welcome_you.go()
#        
#
#    def button_controller(self, t = None):
#
#      if (pygame.event.type == KEYUP) or (pygame.event.type == KEYDOWN):
#         self.welcome_you.parameters.go_duration = (0,'frames')
#
#      return self.display_font_size
        
        
    def play_sound(self, file_name):
        """ plays given sound """ 
        file= file_name
        print "playing ", file
        if self.platform == "windows":
           from winsound import PlaySound, SND_FILENAME, SND_ASYNC
           PlaySound(file, SND_FILENAME|SND_ASYNC)
        elif self.platform == "linux":
           from wave import open as waveOpen
           from ossaudiodev import open as ossOpen
           s = waveOpen('tada.wav','rb')
           (nc,sw,fr,nf,comptype, compname) = s.getparams( )
           dsp = ossOpen('/dev/dsp','w')
           try:
             from ossaudiodev import AFMT_S16_NE
           except ImportError:
             if byteorder == "little":
               AFMT_S16_NE = ossaudiodev.AFMT_S16_LE
             else:
               AFMT_S16_NE = ossaudiodev.AFMT_S16_BE
           dsp.setparameters(AFMT_S16_NE, nc, fr)
           data = s.readframes(nf)
           s.close()
           dsp.write(data)
           dsp.close()  
             
    def start_counting(self):
        """ initiates the start of count down"""
        self.countdown = False
        self.send_parallel(Hex_O.COUNTDOWN_START)
        self.present_count_down.go()
        
    def short_pause_tick(self):
        """ make a short break - of time defined in init section """
        self.send_parallel(Hex_O.SHORTPAUSE_START)
        self.short_break.go()      
            
    #def write_log(self):
    #    pass
        
    #def read_log(self):
    #    pass
                
    # ------------------- figure section ------------------------  
    def init_figures(self):
        """ initializes figures """ 
        #calculates the position for each of the figures
        figure_positions = self.calculate_center_of_each_figure(
                        self.no_of_figures, 
                        self.radius_of_figure, 
                        self.distance_between_figures,
                        [self.screen_size[0]/2, self.screen_size[1]/2])
        self.center_figure_positions = figure_positions
        self.letters_in_each = []
        letters_all = []
        letters_one_by_one = []
        figures_many_letters = []
        figures_one_letter = []
        letters_used = 0
        letters_number = self.no_of_corners_in_figure - 1
        
        #initizlize the figures       
        for i in range(0,len(figure_positions)):    
            now_letters_to_use = self.next_letter[i * letters_number:i * 
                                 letters_number + letters_number]  
            now_letters_to_use = now_letters_to_use[:(i + 
                                 self.no_of_corners_in_figure / 2 ) % 
                                 self.no_of_corners_in_figure] + " " \
                                 + now_letters_to_use[(i + 
                                 self.no_of_corners_in_figure / 2) % 
                                 self.no_of_corners_in_figure:]
            self.letters_in_each.append(now_letters_to_use)
         
            figures_many_letters.append(Figure_Many(color_on = self.color_of_figure_on,
                               color_off = self.color_of_figure_off,
                               center_position = figure_positions[i],
                               radius_size = self.radius_of_figure, 
                               corners_number = self.no_of_corners_in_figure,
                               time_passed = 0,
                               frequency_used = self.flicker_rate[i], 
                               letters_to_use = now_letters_to_use,
                               empty_letter_spot = i,
                               animate = self.time_animate,
                               screen_center = (self.screen.size[0]/2,self.screen.size[1]/2)
                               ))
            figures_one_letter.append(Figure_One(color_on = self.color_of_figure_on,
                               color_off = self.color_of_figure_off,
                               center_position = figure_positions[i],
                               radius_size = self.radius_of_figure, 
                               corners_number = self.no_of_corners_in_figure,
                               time_passed = 0,
                               frequency_used = self.flicker_rate[i], 
                               letters_to_use = now_letters_to_use,
                               figure_number = i,
                               screen_center = (self.screen.size[0]/2,self.screen.size[1]/2)
                               ))
        #initialize the fixation spot          
        if (self.add_fixation_spot_in_center):   
            self.fix_size = self.fixation_spot_size   
        else:
            self.fix_size = (0,0)  

        if (self.class_dot_on):   
            dot_size = self.fixation_spot_size   
        else:
            dot_size = (0,0)  
            
        fixation_spot = FixationSpot(position=(self.screen.size[0]/2,self.screen.size[1]/2),
                             anchor='center',
                             color=self.fixation_spot_color,
                             size = self.fix_size)
        
        classification_dot = FixationSpot(position=(self.screen.size[0]/2,self.screen.size[1]/2),
                             anchor='center',
                             color=self.class_dot_color,
                             size = dot_size)
                
        #initialize the word to be spelled
        word_on_the_top = [] #self.presentation_stopped
        position2 = self.screen.size[0]/2 - len(self.word_to_spell)*5
        for i in range (0, len(self.word_to_spell)):
            position2 
            word_on_the_top.append(Top_Letters(character=str(self.word_to_spell[i]),
                        special_color = self.top_word_special_color ,
                        my_number = i,
                        color=self.top_word_main_color, # alpha is ignored 
                                                        #(set with max_alpha_param)
                        center_position=(position2,self.screen.size[1]-35)))
            position2 = position2 + 10
                           
        figures_many_letters = figures_many_letters + [fixation_spot] + word_on_the_top + [classification_dot]
        figures_one_letter = figures_one_letter + [fixation_spot] + word_on_the_top + [classification_dot]
    
       # adds all to the presentations
        viewport_many_letters=Viewport(screen=self.screen,
                          size=self.screen.size,
                          stimuli= figures_many_letters)

        viewport_one_letter=Viewport(screen=self.screen,
                          size=self.screen.size,
                          stimuli= figures_one_letter)   

#        self.figure_many = Presentation(go_duration=(self.time_of_trial + self.time_animate, 'seconds'), 
#                                        viewports=[viewport_many_letters])   
        self.figure_many = Presentation(go_duration=('forever',), 
                                        viewports=[viewport_many_letters])  
        self.figure_one = Presentation(go_duration=(self.time_of_trial + self.time_animate, 'seconds'), 
                                       viewports=[viewport_one_letter])          
        
        #creates the controllers for the objects
        frequency_controllers = []
        radius_controller = []
        font_controller_many = []
        font_controller_one = []
        letter_controller = []
        animation_controller = FunctionController(during_go_func = self.get_animation_time)
                         
        for i in range(0,len(figure_positions)): 
            frequency_controllers.append(FunctionController(\
            during_go_func = self.get_frequency_time,\
            eval_frequency = Controller.EVERY_FRAME,\
            temporal_variables = \
            Controller.TIME_SEC_SINCE_GO))

            #add each stimulus to the presentation set to be visible
            self.figure_many.add_controller(figures_many_letters[i], 'time_passed',\
                   frequency_controllers[i]) 
            self.figure_one.add_controller(figures_one_letter[i], 'time_passed',\
                   frequency_controllers[i])
            self.figure_many.add_controller(figures_many_letters[i], 'animate',\
                   animation_controller) 
            self.figure_one.add_controller(figures_one_letter[i], 'animate',\
                   animation_controller)
        
        self.figure_many.parameters.handle_event_callbacks = [(pygame.locals.QUIT, self.quit),
                                       (pygame.locals.KEYDOWN, self.keydown)]   

        self.figure_one.parameters.handle_event_callbacks = [(pygame.locals.QUIT, self.quit),
                                       (pygame.locals.KEYDOWN, self.keydown)]   
        k = 0
        for i in range(0,len(figure_positions)): 
            radius_controller.append(FunctionController(\
            during_go_func = self.get_radius,\
            eval_frequency = Controller.EVERY_FRAME,\
            temporal_variables = \
            Controller.TIME_SEC_SINCE_GO))

            font_controller_one.append(FunctionController(\
            during_go_func = self.get_font_one,\
            eval_frequency = Controller.EVERY_FRAME,\
            temporal_variables = \
            Controller.TIME_SEC_SINCE_GO))

            font_controller_many.append(FunctionController(\
            during_go_func = self.get_font_many,\
            eval_frequency = Controller.EVERY_FRAME,\
            temporal_variables = \
            Controller.TIME_SEC_SINCE_GO))

            letter_controller.append(FunctionController(\
                during_go_func = self.get_letters_used,\
                eval_frequency = Controller.EVERY_FRAME,\
                temporal_variables = \
                Controller.TIME_SEC_SINCE_GO))
    
            #add each stimulus to the presentation set to be visible
            self.figure_many.add_controller(figures_many_letters[i], 'radius_size',\
                   radius_controller[i])  
            self.figure_many.add_controller(figures_many_letters[i], 'font_size',\
                   font_controller_many[i])    
              
            self.figure_one.add_controller(figures_one_letter[i], 'radius_size',\
                   radius_controller[i])  
            self.figure_one.add_controller(figures_one_letter[i], 'font_size',\
                   font_controller_one[i]) 
            self.figure_one.add_controller(figures_one_letter[i], 'letters_to_use',\
                   letter_controller[i]) 
            
            k = i     
        color_top_controller = FunctionController(\
            during_go_func = self.get_letter_used_no,\
            eval_frequency = Controller.EVERY_FRAME,\
            temporal_variables = \
            Controller.TIME_SEC_SINCE_GO) 

        fix_cross_controller = FunctionController(during_go_func = self.fix_cross_controller)

        self.figure_one.add_controller(figures_one_letter[k+1], 'size',\
               fix_cross_controller)                         
        self.figure_many.add_controller(figures_one_letter[k+1], 'size',\
               fix_cross_controller)          
                  
        for i in range((k+2),(k+len(self.word_to_spell))+2):
            self.figure_one.add_controller(figures_one_letter[i], 'letter_no_used',\
                   color_top_controller)                         
            self.figure_many.add_controller(figures_one_letter[i], 'letter_no_used',\
                   color_top_controller)  
        """
        calc_class_dot_pos_contr = FunctionController(during_go_func = self.calc_class_dot_pos)
        self.figure_one.add_controller(figures_one_letter[(k+len(self.word_to_spell))+2], 'position',\
            calc_class_dot_pos_contr)                         
        self.figure_many.add_controller(figures_one_letter[(k+len(self.word_to_spell))+2], 'position',\
            calc_class_dot_pos_contr)          
        """
    def fix_cross_controller (self, t = None):
        #returns size of the spot in the center#
        if t > 0.66 * self.time_animate:
            self.cross_size = self.fix_size
        return self.cross_size
    
    def get_animation_time(self, t = None):
        "returns set animation time"           
        return self.animation_time
    
    def get_letter_used_no(self,t = None):
        """ function used in controller - returns current trial"""
        return self.trial_number
    
    def get_letters_used(self,t = None):
        """ function used in controller - returns the letters 
        of the current session to be used"""
        return str(self.use_letters)
    
    def get_radius(self, t = None):
        """ function used in controller - returns radius of figure"""
        return self.radius_of_figure

    def get_font_one(self, t = None):
        """ function used in controller - returns font for one letter figures"""
        return self.one_letter_font_size
    
    def get_font_many(self, t = None):
        """ function used in controller - return font for many letter figures"""
        return self.many_letters_font_size

    def get_frequency_time(self, t):
        """ function used in controller - return time which elapsed"""
        # Control eye tracker
        #self.classifier = np.random.randn(10, 6)
        try:
            #print ">>>"
            OUT= self._data.get("cl_output")
            #print OUT
            OUT_arr = np.asarray(OUT)
            new_input = OUT_arr[np.asarray(self.f_index)]
            if self.classifier_reset == 1:
                print "classifier on, self.classifier reset"
            if len(self.classifier) == 0 or self.classifier_reset == 1:
                self.classifier = [0, 0, 0, 0, 0, 0]
                self.classifier_reset = 0
            else:
                self.classifier = np.add(self.classifier,np.multiply(0.05, (np.subtract(new_input, self.old_classifier))))
                
                #self.classifier.append([])            
            #self.classifier2.append(OUT)
            #OUT_used = []
            #for i in len(self.f_index):
            #    OUT_used.append(OUT[self.f_index[i]])
            #print OUT_used, "out used"
            
            self.old_classifier = new_input
            #self.classifier.append(new_input)
        except:
            #pass
            OUT = [0, 0, 0, 0, 0, 0, 0, 0]

        #print self.classifier
        #print new_input
        
        if self.look_in_the_middle == True:
            if self.use_eyetracker:
              
                if self.et.x is None:
                    #self.logger.error("No eyetracker data received!")
                    print "No eyetracker data received!"
                    self.on_stop()
                    return t
                self.et_currentxy = (self.et.x, self.et.y)
                self.et_duration = self.et.duration
                tx, ty = self.et_targetxy[0], self.et_targetxy[1]
                cx, cy = self.et_currentxy[0], self.et_currentxy[1]
                dist = math.sqrt(math.pow(tx - cx, 2) + math.pow(ty - cy, 2))
    
                #self.et_outside = 0
                # Check if current fixation is outside the accepted range    
                if dist > self.et_range:
                    
                    #self.et_outside = 1
                    # Check if the off-fixation is beyond temporal limits
                    
                    if self.et_duration > self.et_range_time:
    
                        self.send_parallel(self.INVALID)
                        self.figure_one.parameters.go_duration = (0,'frames')
                        self.figure_many.parameters.go_duration = (0,'frames')
    #                    self.figure_one.parameters.go_duration = (self.time_of_count_down,
    #                                            'seconds')
    #                    self.figure_many.parameters.go_duration = (self.time_of_count_down,
    #                                            'seconds')
                        if self.use_eyetracker == True:
                            self.next_to_display = self.error_message
                        else:
                            self.next_to_display = self.error_message2
                        self.show_display()
                        self.presentation_stopped = True
                        #self.sound_invalid.play()
                        self.play_sound("winSpaceCritStop.wav")
                        self.look_in_the_middle = False
                        #show_message(self, "Bad fixation, we have to restart...")
                        #wait_for_key()
                        #self.screen.blit(self.background, self.background_rect)
                        # Send break-off trigger
                        #print " self INVALID"
                        
#        if t > self.time_used:
#            #self.figure_many.quit()
#            #self.figure_one.quit()
##            self.figure_many.parameters.handle_event_callbacks = [(pygame.locals.QUIT, self.quit)]  
##            self.figure_one.parameters.handle_event_callbacks = [(pygame.locals.QUIT, self.quit)]  
#             self.figure_one.parameters.go_duration = (0,'frames')
#             self.figure_many.parameters.go_duration = (0,'frames') 
#             print t
#             print self.time_used
#             print
        return t

    def calc_average(self):
        #if len(self.classifier) == 0:
        #    self.classifier.append([0,3,2,2,2,2])
##        self.avg_class = np.average(np.asarray(self.classifier), 0)
##        print self.avg_class, "average"
##        maxim = max(self.avg_class)
##
##        self.avg_class = self.avg_class.tolist()
##        self.use_hex = self.avg_class.index(maxim)
##        print self.use_hex, "use_hex"
        
        maxim = np.max(self.classifier)
        #print maxim, "maxim"
        print self.classifier, "self.classifier"
        self.classifier = self.classifier.tolist()
        self.use_hex = self.classifier.index(maxim)
        #print self.use_hex, "self.use_hex"
        
    def calc_class_dot_pos(self, t):
        #self.screen_size[]

        maxim = max(np.average(abs(np.asarray(self.classifier)),0))
        if (maxim == 0):
            maxim = 1
        avg_class = (np.average(abs(self.classifier), 0)/maxim) / 2
        self.avg_class = avg_class
 
        #avg_class = np.zeros((1,6))
        #avg_class[:,1] = 0.5
        
        trans_center = np.transpose(self.center_figure_positions)
        relative_y = [m - self.screen_size[0]/2 for m in trans_center[0]]
        relative_x = [m - self.screen_size[1]/2 for m in trans_center[1]]
        #print relative_x, "rela"
        #print avg_class, "avg"
        
        #avg_trans = np.array(np.transpose(avg_class))
        x = sum(relative_x* avg_class)
        y = sum(relative_y* avg_class)
        #print x, "x"
        #print
        position = (x + self.screen_size[0]/2, y + self.screen_size[1]/2)
        
        #for i in self.center_figure_positions:

        return position
        #return position
    def set_classifying_dot(self):
        """this function will be called at each time point. the  """
        pass 
               
    def show_display(self):
        """displays the text on the screen"""
        self.send_parallel(Hex_O.DISPLAY_START)
        #if self.play_ok == True:
        #    self.play_sound("winSpaceDefault.wav")
        self.present_display.go()
             
    def keydown(self, event):
        """controls the press of the keyboard keys"""
        global up, down, left, right
        self.welcome_you.parameters.go_duration = (0,'frames')
        if event.key == pygame.locals.K_ESCAPE:
            quit(event)
        elif event.key == pygame.locals.K_EQUALS:
            self.radius_of_figure = self.radius_of_figure + self.size_of_change
            self.many_letters_font_size = self.many_letters_font_size + self.size_of_change
            self.one_letter_font_size = self.one_letter_font_size + self.size_of_change
        elif event.key == pygame.locals.K_MINUS:
            self.radius_of_figure = self.radius_of_figure - self.size_of_change
            self.many_letters_font_size = self.many_letters_font_size - self.size_of_change
            self.one_letter_font_size = self.one_letter_font_size - self.size_of_change            
        else:
            print event.key
            
    def quit(event):
        """controls quit"""
        self.figure_one.parameters.go_duration = (0,'frames')
        self.figure_many.parameters.go_duration = (0,'frames') 
            
    def calculate_center_of_each_figure(self, number_of_figures, radius_of_figure, 
                                        distance_between, center, start_at = np.pi/2):
        """calculates the center for the figures """
        all_positions = []
        if number_of_figures == 1:
            all_positions.append((self.screen_size[0]/2, self.screen_size[1]/2))
            return(all_positions)
        else:
            angle_between = 2 * np.pi / number_of_figures
            circle_size = (radius_of_figure + distance_between) * number_of_figures
            radius = circle_size/ (np.pi*2)
            a = center[0]
            b = center[1]
            for i in range(0, number_of_figures):
                all_positions.append((a + radius *p.cos(i * angle_between + start_at),
                                      b + radius *p.sin(i * angle_between + start_at))) 
            return all_positions    
        
    # ------------------- count down section ------------------------    
    def control_count_down(self, t):
        """ define counting as a function of time """
        if self.round_count_down == 0:
            return str(int(self.use_time - t) + 1)
        else:
            return str(round(self.use_time - t, self.round_count_down))
        
    def init_count_down(self):
        """ initializes count down """
        self.use_time = self.time_of_count_down
        text = Text(text=str(self.use_time),
                    color=self.color_countdown, # alpha is ignored (set with max_alpha_param)                    position=(self.screen.size[0]/2,self.screen.size[1]/2),
                    font_size= self.count_down_font_size,
                    anchor='center',
                    position = (self.screen.size[0]/2,self.screen.size[1]/2))
        
        viewport = Viewport(screen=self.screen,
                    size=self.screen.size,
                    stimuli=[text])
        self.present_count_down = Presentation(go_duration=(self.time_of_count_down,
                                            'seconds'),viewports=[viewport])

        #create the frequency controller for the countdown   
        text_controller = FunctionController(\
            during_go_func = self.control_count_down,\
            eval_frequency = Controller.EVERY_FRAME,\
            temporal_variables = \
            Controller.TIME_SEC_SINCE_GO)
        self.present_count_down.add_controller(text,'text', text_controller)
        
    # -------------------- init section ----------------------------        
    def init_graphics(self):
        """ here all the graphics is being initialized """
        self.init_screen()
        self.init_count_down()
        self.init_text_to_display()
        self.init_figures()

    def control_display(self, t = None):
        return self.next_to_display
        
    def init_text_to_display(self):
        """ initializes count down """
        text = Text(text=str(self.word_to_spell),
                    color=self.top_word_main_color, # alpha is ignored (set with max_alpha_param)
                    position=(self.screen.size[0]/2,self.screen.size[1]/2),
                    font_size= self.display_font_size,
                    anchor='center')           
        viewport = Viewport(screen=self.screen,
                    size=self.screen.size,
                    stimuli=[text])
        self.present_display = Presentation(go_duration=(self.time_word_display,
                                                         'seconds'),viewports=[viewport])
        text_controller = FunctionController(\
            during_go_func = self.control_display,\
            eval_frequency = Controller.EVERY_FRAME,\
            temporal_variables = \
            Controller.TIME_SEC_SINCE_GO)       
        self.present_display.add_controller(text,'text', text_controller)
          
    def init_screen(self):
        """ initiates settings of the screen """
        VisionEgg.start_default_logging(); 
        VisionEgg.watch_exceptions()
        VisionEgg.config.VISIONEGG_FRAMELESS_WINDOW = 1
        VisionEgg.config.VISIONEGG_MONITOR_REFRESH_HZ = self.monitor_refresh_rate    
        VisionEgg.config.VISIONEGG_GUI_INIT = 0
        VisionEgg.config.VISIONEGG_SCREEN_W = self.screen_size[0]
        VisionEgg.config.VISIONEGG_SCREEN_H = self.screen_size[1]
        #if feedback on another screen - put it there
        if self.same_screen == False:
            os.environ['SDL_VIDEO_WINDOW_POS']="-800,1000"
        else:
            os.environ['SDL_VIDEO_CENTERED']="center"
        if self.full_screen == True:
            VisionEgg.config.VISIONEGG_FULLSCREEN = 1
        self.screen = get_default_screen()
        self.screen.parameters.bgcolor = self.color_of_background
    
    def shortBreakInit(self):
        # Create an instance of the Target2D class with appropriate parameters.
        stimulus = Target2D(size  = (40.0, 20.0),
                  color      = (1.0, 0.0, 0.0, 1.0), 
                  orientation = -45.0, 
                  on = False)
        viewport = Viewport(screen=self.screen, stimuli=[stimulus] )     
        self.short_break = Presentation(go_duration=(self.time_of_pause,
                                                     'seconds'),viewports=[viewport])
   
if __name__ == '__main__':
    ho = Hex_O()
    ho.on_init()
    ho.on_play()
    print "left all"
  
