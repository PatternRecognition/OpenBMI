#!/usr/bin/env python
 
# Ssvep_dots.py -
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

#'4 Hz','5 Hz','6 Hz','7.5 Hz','8 Hz','10 Hz','12 Hz','15 Hz','20 Hz','24 Hz','30 Hz'
from VisionEgg import *
from VisionEgg.WrappedText import WrappedText
import numpy as np
import pylab as p

from VisionEgg.FlowControl import Presentation, FunctionController
from VisionEgg.Core import *
from VisionEgg.MoreStimuli import *
from VisionEgg.Text import *
from Points import *
from Animate import *
from VisionEgg.Textures import *
from random import *
#from lib.eyetracker import EyeTracker
from FeedbackBase.MainloopFeedback import MainloopFeedback

    
class Ssvep_dots(MainloopFeedback):
    # TRIGGER VALUES FOR THE PARALLEL PORT (MARKERS)
    START_EXP, END_EXP = 252, 253
    COUNTDOWN_START = 60
    #different from pilot - here ALWAYS the start of trial animation is 36 
    START_ANIMATION = 36  
    START_TRIAL = 37
    SHORTPAUSE_START = 249
    INVALID = 66
    DISPLAY_START = 45
    WELCOME = 150
    COLOR1, COLOR2, COLOR3, COLOR4 = 22, 23, 24, 25
    #in markers of stimuli: 1st frequency used 2nd corner used, eg: 12 - first freq, 2nd corner

    def init(self):
        """ initializes all the settings of the figure(s)"""
        #self.pictures = ["D:\pics_ssvep\prom.JPG", "D:\pics_ssvep\cross.JPG", "D:\pics_ssvep\ertical.JPG", "D:\pics_ssvep\horizontal.JPG"]#, "ludzik.JPG", "rabbit.JPG"]
        #------those variables must be changed each time!!-----------------
        #self.frequencies = (4.0, 6.0, 7.5, 8.0)
        #self.frequencies = (30.0, 30.0, 30.5, 30.0)
        use_case = 1
        self.covert = 0
        self.no_of_classes = 4
        self.training = 1
        times_in_corner = 4
        
        if use_case == 1:         
            self.use_solid = True
            self.pix_size = 6 #1
            pic_small = True
            self.use_all = True
        elif use_case == 2:
            self.use_solid = True
            self.pix_size = 5 #1
            pic_small = True
            self.use_all = False
        elif use_case == 3:
            self.use_solid = False
            self.pix_size = 1 #1
            pic_small = True
            self.use_all = True             
        elif use_case == 4:
            self.use_solid = False
            self.pix_size = 1 #1
            pic_small = True
            self.use_all = False
        elif use_case == 5:
            self.use_solid = False
            self.pix_size = 5 #1
            pic_small = True
            self.use_all = True  
        elif use_case == 6:
            self.use_solid = False
            self.pix_size = 5 #1
            pic_small = True
            self.use_all = False  
        elif use_case == 7:
            self.use_solid = True
            self.pix_size = 5 #1
            pic_small = False
            self.use_all = True
        elif use_case == 8:
            self.use_solid = True
            self.pix_size = 5 #1
            pic_small = False
            self.use_all = False             
        #-------------------------------------------------------------------

        #if self.use_solid == 0:
        #    use_solid = False
        #else:
        #    use_solid = True

        #if self.pic_small == 0:
        #    pic_small = False
        #else:
        #    pic_small = True

        #if self.use_all == 0:
        #    self.use_all = False
        #else:
        #    self.use_all = True              
       
        if self.no_of_classes == 4:
            self.pictures = ["D:\pics_ssvep\horizontal.JPG", "D:\pics_ssvep\diag_left.JPG", "D:\pics_ssvep\ertical.JPG", "D:\pics_ssvep\diag_right.JPG"]
            #self.pictures = ["horizontal.JPG", "diag_left.JPG", "vertical.JPG", "diag_right.JPG"]
            
        else:
            self.pictures = ["D:\pics_ssvep\cross2.JPG", "D:\pics_ssvep\cross.JPG"]

        #print "self.covert: ", self.covert, "self.use_solid: ", self.use_solid, "self.pix_size: ", self.pix_size, "self.pic_small; ", self.pic_small      
        self.colors = ((0.0, 1.0, 0.0), (1.0, 0.5, 0.1), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0))        
        if self.use_solid:
            self.dot_sizes = 4
        else:
            self.dot_sizes = 1
            
        self.numbers_in_class = [1, 2, 3, 4]
        if pic_small:
            self.pic_size = (200, 200)#if the given picture is not of this size it will be normalized to it     
            
        else:
            self.pic_size = (400, 400)                                            
        self.display_font_size = 75   
             
        #settings about countdown
        self.color_countdown = (1.0,1.0,1.0)   
        self.round_count_down = 1  
        self.count_down_font_size = 200  
        self.class_dot_color = (1.0,1.0,1.0)
        self.class_dot_on = False
        #self.startFreq = 4.0
        self.startFreq = 8.0
        self.endFreq = 16.00
        self.stepFrequency = 0.01
        
        if self.no_of_classes == 4:
            self.classes = ("ABCDEFGHI", "JKLMNOPQR", "STUVWXYZ_", "#  123.,?")
        else:
            self.classes = ("", "", "", "")
        self.special_char = "<"
                        
        #other settings about display
        self.color_of_background = (0.0,0.0,0.0) 
        if not self.training:
            self.color_upper_txt = (1.0, 1.0, 1.0)
        else:
            self.color_upper_txt = self.color_of_background
            
        #self.fixation_spot_color = (255,0,0,0) 
        self.fixation_spot_size = (8, 8)
        #self.error_message = "Look in the middle"
        #self.error_message2 = "Look at the target"
        if self.training:
            self.covert_message = "Please concentrate on the figure with the letter displayed beforhand"
        elif self.no_of_classes == 2:
            self.covert_message = "Please concentrate on one of the figure, and state if the feedback was correct"
        else:
            self.covert_message = "Please spell: aaa by concentrating on the figure with the appropriate letter"
                
        #self.overt_message = "Now you will see 4 figures. Each time the letter will be shown - you must LOOK at the hexagon displaying that letter, if not, error message will be shown and you will have to repeat the trial. When you are ready click any button and good luck"""
        #self.covert_message = "Please concentrate on the symbol shown"
                
        #settings of monitor  
        self.screen_size = [1280, 800]  #[width, height]
        self.top_letter_position = (self.screen_size[0] / 2.0, self.screen_size[1] - 20)
        self.monitor_refresh_rate = 120
        self.same_screen = True
        self.full_screen = False #not adequate if two screens used  
        self.platform = "windows" # "linux"

        #settings of the frequencies & timing
        self.time_of_trial = 5 #given in seconds
        self.time_of_pause = 0


        self.time_to_present = 1

        self.time_of_count_down = 5
        if self.no_of_classes == 2 or self.training:
            self.time_animation = 0
            self.time_preanimate = 0
        else:
            self.time_preanimate = 3
            if self.covert == 1:
                self.time_animation = 1
            else:
                self.time_animation = 0
                
        #self.time_word_display = 1 
        if self.training:
            self.set_FlickerRate()
            self.set_corners()
            self.number_of_trials = p.size(self.frequencies)* 4 * times_in_corner
        else:
            self.number_of_trials = 1
            self.classifier = []
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
        x = self.screen_size[0]
        y = self.screen_size[1]     
        self.starting_position = [[1.0/4 * x, 3.0/4 * y], [3.0/4 * x, 3.0/4 * y], [1.0/4 * x, 1.0/4 * y], [3.0/4 * x, 1.0/4 * y]]
        self.ending_position = p.ones([4, 2]) * (x/2.0, y/2.0)                      

        self.letter_position = self.starting_position
        if self.covert == 0:
            self.ending_position = self.starting_position
            
        self.to_display = self.classes
        self.level = 1
        self.up_level_char = "<"
        self.remove_char = "#  "
        #self.center_figure_positions
        #self.figure_to_be_used
        self.characters_to_spell = "abecadlo"
        self.letters_spelled = ""
                
    def pre_mainloop(self):
        """ prepares the feedback for being run """  
        #settings not to be changed normally
        if self.use_eyetracker:
            self.et = EyeTracker()
            self.et.start()
        if self.et_fixate_center == True:
            self.add_fixation_spot_in_center = True 
        self.another_trial = self.number_of_trials    
        self.trial_number = 0
        self.display_full = 1
        self.presentation_stopped = False

        self.present_count_down = None
        self.time_used = 0
        #self.cross_size = self.fixation_spot_size
        self.play_ok = True
        #self.look_in_the_middle = False
        if self.time_of_count_down > 0:
            self.countdown = True
        else:
            self.countdown = False  
        
        #initialize everything         
        self.init_graphics()
        
        self.shortBreakInit()
        self.send_parallel(Ssvep_dots.START_EXP)
        self.welcome()
        #self.sound_invalid = pygame.mixer.Sound("Feedbacks\P300\winSpaceCritStop.wav")
        #self.order_of_pics = [i+1 for i in range(0, p.size(self.pictures))]
        #self.order_of_pics = [1, 2, 3, 2]

        self.f_used = -1
        #shuffle(self.order_of_pics)
        self.next_pic = -1
        self.all_markers = []
    #def post_mainloop(self):
    #    """ end the whole experiment """
    #    self.send_parallel(Ssvep_dots.END_EXP)
    #    self.on_quit()
        
    def tick(self):
        print "tick"
        
    def play_tick(self):
        """ main experiment control"""
        print "play tick"
        if self.countdown:
            #self.play_ok = False
            self.start_counting() 
            #self.play_ok = True           
        else:
            #shows the full display
            if self.display_full == 1:
                self.display_full = 0
            if self.another_trial > 0:

                if self.training:
                    if self.another_trial % len(self.frequencies) == 0:
                        print "changing the corners!!!"
                        self.set_chosen_figure()                    
                    
                    self.trial_tick()
   
                else:
                    self.trial_tick()
                #self.another_trial = self.another_trial - 1
                    self.show_next()

            if self.another_trial == 0:   
                #self.next_to_display = "the end"
                #self.show_display()          
                self.post_mainloop()

            self.trial_number = self.trial_number + 1
    
    def show_next(self):
        """display next figure to concentrate on"""
#        self.next_pic = (self.next_pic + 1) % 4
#        self.send_parallel(self.order_of_pics[self.next_pic])
        present = self.stat_img[self.chosen_figure]
        present.go()
         
                    
    def trial_tick(self):
        """ display the text and the figures for the many letters trial """
        
        self.presentation_stopped = True
        
        while self.presentation_stopped == True:
            self.presentation_stopped = False

            #self.cross_size =  self.fixation_spot_size
            if (self.time_animation + self.time_preanimate) > 0.0:
                  self.send_parallel(Ssvep_dots.START_ANIMATION)
                  self.animate.go() 
            if self.training:  
                self.f_used = (self.f_used + 1) % len(self.frequencies)  
                print "self.f_used", self.f_used
                    
                idx, = np.where(np.array(self.marker_freq) == self.frequencies[self.f_used]) 
                    
                #idx = idx.tolist()
                self.c_used = self.corners[self.f_used]
                self.chosen_figure = self.c_used - 1
                self.show_next()
                for i in range(4):
                    if self.c_used - 1 == i: 
                        self.stimulus[i].parameters.frequency_used = self.frequencies[self.f_used]
                    else:
                        if self.use_all == False:
                            self.stimulus[i].parameters.frequency_used = 0
                        else:
                            self.stimulus[i].parameters.frequency_used = self.frequencies[(self.f_used + i + 1) % len(self.frequencies)]
                marker = (idx + 1) * 10 + self.c_used
                [marker] = marker.tolist()
                print "next marker", marker
                self.send_parallel(marker)
                self.all_markers.append(marker)
                if (self.another_trial % (4 * len(self.frequencies))) == 0:
                    shuffle(self.corners)
            else:
                self.send_parallel(Ssvep_dots.START_TRIAL)

            #self.look_in_the_middle = True
            #self.figure_many.parameters.go_duration = (self.time_of_trial, 'seconds') 
            #self.figure_many.go()
            self.flickers.go()
            if self.training:
                self.send_parallel(Ssvep_dots.SHORTPAUSE_START)
                self.another_trial = self.another_trial - 1
            else:
                self.calc_average()
                self.classifier = []
                self.send_parallel(self.chosen_figure + 1)
                
                if self.no_of_classes == 2:
                    self.another_trial = self.another_trial - 1
                else:
                    # chosen figure
                    if self.level == 2:
                        self.previous_level = self.to_display
                    #self.chosen_figure = 
                    if  self.to_display[self.chosen_figure] == self.up_level_char:
                        #go level up
                        
                        if self.level == 2:
                            #self.level = 1
                            self.to_display = self.classes
                        else:
                            self.to_display = self.previous_level
                        
                        self.level = self.level - 1
                        print "go level up"
                    elif self.to_display[self.chosen_figure] == self.remove_char:
                        # remove previous letter
                        print "remove character"
                        if len(self.letters_spelled) > 0:
                            self.letters_spelled = self.letters_spelled[0:len(self.letters_spelled) - 1]
                            self.another_trial = self.another_trial + 1
                        self.to_display = self.classes
                        self.level = 1
                    elif self.level == 3 or len(self.to_display[self.chosen_figure]) == 1:
                        self.letters_spelled = self.letters_spelled + self.to_display[self.chosen_figure]
                        self.to_display = self.classes
                        self.another_trial = self.another_trial - 1
                        self.level = 1
                        # display the chosen letter on the top
                    else:
        #                print "chosen_figure", chosen_figure
                        # get the new display letters
                        chosen = self.to_display[self.chosen_figure]
                        print "chosen", chosen
                        print self.level, "level"
                        if self.level == 1:
                            self.to_display = (chosen[0:3], chosen[3:6], chosen[6:9], "<")
                        else:
                            self.to_display = (chosen[0], chosen[1], chosen[2], "<")
                        self.level = self.level + 1
    #                    self.to_display = self.to_display[len(self.to_display) * (chosen_figure-1)/4:len(self.to_display) * (chosen_figure)/4]
    #            print "self.to_display",self.to_display
    #            #self.presentation_stopped = False

    def welcome(self):
        """this will display the welcome message"""
        msg = self.covert_message

            
        text = WrappedText(text=msg,
                    color=self.color_countdown, # alpha is ignored (set with max_alpha_param)                    position=(self.screen.size[0]/2,self.screen.size[1]/2),
                    font_size= self.display_font_size,
                    position=(10,self.screen.size[1]-10))
        
        viewport = Viewport(screen=self.screen,
                    size=self.screen.size,
                    stimuli=[text])
        
        self.welcome_you = Presentation(go_duration=('forever',),viewports=[viewport])
        self.welcome_you.parameters.handle_event_callbacks = [(pygame.locals.KEYDOWN, self.keydown)] 

        self.welcome_you.go()     
        
#    def play_sound(self, file_name):
#        """ plays given sound """ 
#        file= file_name
#        print "playing ", file
#        if self.platform == "windows":
#           from winsound import PlaySound, SND_FILENAME, SND_ASYNC
#           PlaySound(file, SND_FILENAME|SND_ASYNC)
#        elif self.platform == "linux":
#           from wave import open as waveOpen
#           from ossaudiodev import open as ossOpen
#           s = waveOpen('tada.wav','rb')
#           (nc,sw,fr,nf,comptype, compname) = s.getparams( )
#           dsp = ossOpen('/dev/dsp','w')
#           try:
#             from ossaudiodev import AFMT_S16_NE
#           except ImportError:
#             if byteorder == "little":
#               AFMT_S16_NE = ossaudiodev.AFMT_S16_LE
#             else:
#               AFMT_S16_NE = ossaudiodev.AFMT_S16_BE
#           dsp.setparameters(AFMT_S16_NE, nc, fr)
#           data = s.readframes(nf)
#           s.close()
#           dsp.write(data)
#           dsp.close()  
             
    def start_counting(self):
        """ initiates the start of count down"""
        self.countdown = False
        self.send_parallel(Ssvep_dots.COUNTDOWN_START)
        self.present_count_down.go()
        
    def short_pause_tick(self):
        """ make a short break - of time defined in init section """
        self.send_parallel(Ssvep_dots.SHORTPAUSE_START)
        self.short_break.go()      

    # ------------------- figure section ------------------------  
    def init_figures(self):
        """ initializes figures """ 
        self.stimulus = []
        self.stat_img = []
        # initiate blinking figures
        k = 0  
        text = Text(text=self.letters_spelled,
            color= self.color_upper_txt, # alpha is ignored (set with max_alpha_param)
            position=(self.top_letter_position),
            font_size=50,
            anchor='center')

        view_text = Viewport(screen=self.screen,
                size=self.screen.size,
                stimuli=[text])
            
        for pict in self.pictures:
            k = k + 1
            stim = Points(position                = self.ending_position[k - 1] ,
                              anchor                  = 'center',
                              size                    = self.pic_size,
                              dot_size                = self.dot_sizes,
                              color                   = self.colors[k - 1], 
                              number                  = self.numbers_in_class[k - 1],
                              pixel_size              = self.pix_size,
                              no_of_all               = self.no_of_classes,
                              picture                 = pict, 
                              frequency_used          = self.frequencies[k-1],
                              use_solid               = self.use_solid
                              #time_animation          = self.time_animation
                              )
            self.stimulus.append(stim)
        viewport = Viewport( screen=self.screen, stimuli=self.stimulus )
        self.flickers = Presentation(go_duration=(self.time_of_trial,'seconds'),viewports=[viewport, view_text])
        
        # initiate animation figures
        animation = []
        if (self.time_animation + self.time_preanimate) > 0.0:
            k = 0  
            
            for pict in self.pictures:
                k = k + 1
                stim = Animate(starting_position          = self.starting_position[k - 1],
                                  ending_position         = self.ending_position[k - 1],
                                  anchor                  = 'center',
                                  size                    = self.pic_size,
                                  dot_size                = self.dot_sizes,
                                  color                   = self.colors[k - 1], 
                                  number                  = self.numbers_in_class[k - 1],
                                  no_of_all               = self.no_of_classes,
                                  pixel_size              = self.pix_size,
                                  picture                 = pict,  
                                  time_animation          = self.time_animation,
                                  time_before             = self.time_preanimate,
                                  display                 = self.classes,
                                  level                   = 1,
                                  letter_position         = self.letter_position[k - 1]
                                  )
                animation.append(stim)
            viewport = Viewport( screen=self.screen, stimuli=animation )
            self.animate = Presentation(go_duration=(self.time_animation + self.time_preanimate,'seconds'),viewports=[viewport, view_text])            
        k = 0
        display = []
        for pict in self.pictures:
            k = k + 1
            stim = Animate(starting_position          = self.ending_position[k - 1],
                              ending_position         = self.ending_position[k - 1],
                              anchor                  = 'center',
                              size                    = self.pic_size,
                              dot_size                = self.dot_sizes,
                              color                   = self.colors[k - 1], 
                              number                  = self.numbers_in_class[k - 1],
                              no_of_all               = self.no_of_classes,
                              picture                 = pict,  
                              time_animation          = self.time_to_present,
                              time_before             = 0,
                              display                 = self.classes,
                              level                   = 1,
                              letter_position         = self.ending_position[k - 1]
                              )
            display.append(stim)
            viewports = (Viewport( screen=self.screen, stimuli=[stim] ))

            self.stat_img.append(Presentation(go_duration=(self.time_to_present,'seconds'),viewports=[viewports])) 
                    
        frequency_controllers = []
        display_controllers = []
        level_controllers = []
        for i in range(0,len(self.stimulus)): 

            frequency_controllers.append(FunctionController(\
            during_go_func = self.get_frequency_time,\
            eval_frequency = Controller.EVERY_FRAME,\
            temporal_variables = \
            Controller.TIME_SEC_SINCE_GO))
            if self.no_of_classes == 4:
                display_controllers.append(FunctionController(\
                during_go_func = self.get_display,\
                eval_frequency = Controller.EVERY_FRAME,\
                temporal_variables = \
                Controller.TIME_SEC_SINCE_GO))

            level_controllers.append(FunctionController(\
            during_go_func = self.get_level,\
            eval_frequency = Controller.EVERY_FRAME,\
            temporal_variables = \
            Controller.TIME_SEC_SINCE_GO))
    
            #add each stimulus to the presentation set to be visible
            self.flickers.add_controller(self.stimulus[i], 'time_passed',\
                   frequency_controllers[i])
            if p.size(animation) > 0:
                self.animate.add_controller(animation[i], 'time_passed',\
                       frequency_controllers[i])    

                self.animate.add_controller(animation[i], 'level',\
                       level_controllers[i])  
                self.animate.add_controller(display[i], 'time_passed',\
                       frequency_controllers[i])    
                if self.no_of_classes == 4:
                    self.animate.add_controller(display[i], 'display',\
                           display_controllers[i])   
                self.animate.add_controller(animation[i], 'display',\
                       display_controllers[i])   
                                    
                self.animate.add_controller(display[i], 'level',\
                       level_controllers[i])                  
                

        text_controllers = (FunctionController(\
        during_go_func = self.get_text,\
        eval_frequency = Controller.EVERY_FRAME,\
        temporal_variables = \
        Controller.TIME_SEC_SINCE_GO)) 
        if self.time_animation + self.time_preanimate > 0:
            self.animate.add_controller(text, 'text',\
                       text_controllers)                
                            
 
    def post_mainloop(self):
        """ end the whole experiment """
        if self.training:
            print "freq marker used: ", self.all_markers
        
        if self.no_of_classes == 2:
            write_text = "Thank you"
        else:
            write_text = "Thank you\nYou spelled:\n\n" + self.letters_spelled
        text = WrappedText(text= write_text ,
                    color=self.color_upper_txt, # alpha is ignored (set with max_alpha_param)                    position=(self.screen.size[0]/2,self.screen.size[1]/2),
                    font_size= self.display_font_size,
                    position=(150,self.screen.size[1]- 150))
        
        viewport = Viewport(screen=self.screen,
                    size=self.screen.size,
                    stimuli=[text])
        the_end = Presentation(go_duration=('forever',),viewports=[viewport])
        
        self.send_parallel(Ssvep_dots.END_EXP)
        #self.on_quit()
        the_end.go()

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
        
#        if self.look_in_the_middle == True:
#            if self.use_eyetracker:
#              
#                if self.et.x is None:
#                    #self.logger.error("No eyetracker data received!")
#                    print "No eyetracker data received!"
#                    self.on_stop()
#                    return t
#                self.et_currentxy = (self.et.x, self.et.y)
#                self.et_duration = self.et.duration
#                tx, ty = self.et_targetxy[0], self.et_targetxy[1]
#                cx, cy = self.et_currentxy[0], self.et_currentxy[1]
#                dist = math.sqrt(math.pow(tx - cx, 2) + math.pow(ty - cy, 2))
#    
#                #self.et_outside = 0
#                # Check if current fixation is outside the accepted range    
#                if dist > self.et_range:
#                    
#                    #self.et_outside = 1
#                    # Check if the off-fixation is beyond temporal limits
#                    
#                    if self.et_duration > self.et_range_time:
#    
#                        self.send_parallel(self.INVALID)
#                        self.figure_one.parameters.go_duration = (0,'frames')
#                        self.figure_many.parameters.go_duration = (0,'frames')
#    #                    self.figure_one.parameters.go_duration = (self.time_of_count_down,
#    #                                            'seconds')
#    #                    self.figure_many.parameters.go_duration = (self.time_of_count_down,
#    #                                            'seconds')
#                        if self.et_fixate_center == True:
#                            self.next_to_display = self.error_message
#                        else:
#                            self.next_to_display = self.error_message2
#                        #self.show_display()
#                        self.presentation_stopped = True
#                        #self.sound_invalid.play()
#                        self.play_sound("winSpaceCritStop.wav")
#                        self.look_in_the_middle = False

        return t
    
    def get_text(self, t):
        return self.letters_spelled
    
    def get_display(self, t):
        return self.to_display

    def get_level(self, t):
        return self.level    

    
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

    def calc_average(self):
        if len(self.classifier) == 0:
            #self.classifier.append([0,3,2,2,2,2])
            if self.no_of_classes == 4:
                self.classifier = [0, 1, 2, 3]
            else:
                self.classifier = [0, 1]
            shuffle(self.classifier)
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
        try:
            self.classifier = self.classifier.tolist()
        except:
            pass
        self.chosen_figure = self.classifier.index(maxim)
        #self.use_hex = 3
        print self.chosen_figure, "self.chosen_figure"
        
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
        self.send_parallel(Ssvep_dots.DISPLAY_START)
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
        self.init_figures()

    def control_display(self, t = None):
        return self.next_to_display
        

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
    
    def set_chosen_figure(self):
        self.corners = np.array(self.corners)
        self.corners = (self.corners % 4) + 1
        self.corners = self.corners.tolist()
        
        
    def set_corners(self):
        self.corners = [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4]
        shuffle(self.corners)
            
    def set_FlickerRate(self):
        i = self.startFreq     
        self.frequencies = []
        self.marker_freq = []
        while i < self.endFreq + self.stepFrequency:
            doNotAdd = 0
            #for p in range(len(self.flicker_rate)):
            if (self.monitor_refresh_rate % i) == 0:
                doNotAdd = 1
                    #break
            if doNotAdd == 1:
                self.frequencies.append(round(i,2)) 
                self.marker_freq.append(round(i,2))
            i = round(i,2) + self.stepFrequency
        print "new flickers: " + str(self.frequencies)  
        #print "to markers: " + str(self.marker_numbers)  
        shuffle(self.frequencies)
        print "played in order: " + str(self.frequencies)   
        
if __name__ == '__main__':
    sd = Ssvep_dots()
    sd.on_init()
    sd.on_play()
    print "left all"
  
