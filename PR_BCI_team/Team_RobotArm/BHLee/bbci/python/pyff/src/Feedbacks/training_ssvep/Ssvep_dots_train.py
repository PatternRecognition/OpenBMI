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

# frequencies used: [4.0, 5.0, 6.0, 7.5, 8.0, 10.0, 12.0, 15.0, 20.0, 24.0, 30.0]
from VisionEgg import *
from VisionEgg.WrappedText import WrappedText
import numpy as np
import pylab as p

from VisionEgg.FlowControl import Presentation, FunctionController
from VisionEgg.Core import *
from VisionEgg.MoreStimuli import *
from VisionEgg.Text import *
from Points import *
from VisionEgg.Textures import *
import random as ran
from random import *
#from lib.eyetracker import EyeTracker
from FeedbackBase.MainloopFeedback import MainloopFeedback
       
class Ssvep_dots(MainloopFeedback):
    # TRIGGER VALUES FOR THE PARALLEL PORT (MARKERS)
    START_EXP, END_EXP = 252, 253
    COUNTDOWN_START = 60
    #different from pilot - here ALWAYS the start of trial animation is 36 
    #frequencies: 1, 2, 3, 4, 5, 6...
    START_ANIMATION = 36  
    START_TRIAL = 37
    SHORTPAUSE_START = 249
    INVALID = 66
    DISPLAY_START = 45
    WELCOME = 150
    COLOR1, COLOR2, COLOR3, COLOR4 = 22, 23, 24, 25

    def init(self):
        """ initializes all the settings of the figure(s)"""
        #self.pictures = ["D:\pics_ssvep\prom.JPG", "D:\pics_ssvep\cross.JPG", "D:\pics_ssvep\ertical.JPG", "D:\pics_ssvep\horizontal.JPG"]#, "ludzik.JPG", "rabbit.JPG"]
        self.pictures = ["cross.JPG", "cross_hor.JPG"] #, "vertical.JPG", "horizontal.JPG"]#, "ludzik.JPG", "rabbit.JPG"]
        #self.pictures = ["abc.JPG", "red_cross.JPG", "house.JPG", "smiley.JPG"]
        self.colors = ((0.0, 1.0, 0.0), (1.0, 0.5, 0.1))
        #self.frequencies = (4.35, 5.0, 8.5, 10.0, 17.0, 21.5)
        self.startFreq = 4.0
        self.endFreq = 35.0
        self.stepFrequency = 0.01
        self.no_of_classes = 2
        self.dot_sizes = 1
        self.numbers_in_class = [1]
        self.pic_size = (100, 100) #if the given picture is not of this size it will be normalized to it                            
        self.display_font_size = 75        
        #settings about countdown
        self.color_countdown = (1.0,1.0,1.0)   
        self.round_count_down = 1  
        self.count_down_font_size = 200  
        self.class_dot_color = (1.0,1.0,1.0)
        self.class_dot_on = False
        
        #self.classes = ("ABCDEFGHI", "JKLMNOPQR", "STUVWXYZ_", "#  123.,?")
        #self.special_char = "<"
        
                        
        #other settings about display
        self.color_of_background = (0.0,0.0,0.0) 
        self.color_upper_txt = (1.0, 1.0, 1.0)
        self.fixation_spot_color = (255,0,0,0) 
        self.fixation_spot_size = (8, 8)
                
        #settings of monitor  
        self.screen_size = [1280, 800]  #[width, height]
        self.monitor_refresh_rate = 120
        self.same_screen = True
        self.full_screen = False #not adequate if two screens used  
        self.platform = "windows" # "linux"

        #settings of the frequencies & timing
        self.time_of_trial = 3 #given in seconds
        self.time_of_pause = 1
        self.time_of_count_down = 1  
        self.number_of_trials = 2
        
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
        self.ending_position = (x/2.0, y/2.0)
                
    def pre_mainloop(self):
        """ prepares the feedback for being run """  
        #settings not to be changed normally
        self.to_show = 0
        self.set_FlickerRate()
        if self.use_eyetracker:
            self.et = EyeTracker()
            self.et.start()
        if self.et_fixate_center == True:
            self.add_fixation_spot_in_center = True 
        print "frequency lenght, ", len(self.frequencies)
        self.another_trial = self.number_of_trials * len(self.frequencies) #p.size(self.pictures)       
        self.trial_number = 0
        self.display_full = 1
        self.presentation_stopped = False
       
        self.present_count_down = None
        self.time_used = 0
        self.cross_size = self.fixation_spot_size
        self.play_ok = True
        self.look_in_the_middle = False
        if self.time_of_count_down > 0:
            self.countdown = True
        else:
            self.countdown = False  
        
        #initialize everything   
        self.f_used = 0      
        self.init_graphics()
        
        self.shortBreakInit()
        self.send_parallel(Ssvep_dots.START_EXP)
        #print "got to the start"
        #self.welcome()
        #self.sound_invalid = pygame.mixer.Sound("Feedbacks\P300\winSpaceCritStop.wav")
        self.order_of_pics = [i+1 for i in range(0, p.size(self.pictures))]
        self.order_of_pics = [1, 2, 3, 2]
        


        #shuffle(self.order_of_pics)
        self.next_pic = -1

    def post_mainloop(self):
        """ end the whole experiment """
        text = WrappedText(text="Thank you\n        The end",
                    color=self.color_upper_txt, # alpha is ignored (set with max_alpha_param)                    position=(self.screen.size[0]/2,self.screen.size[1]/2),
                    font_size= self.display_font_size,
                    position=(self.screen.size[0]/2 - 150,self.screen.size[1]/2))
        
        viewport = Viewport(screen=self.screen,
                    size=self.screen.size,
                    stimuli=[text])
        the_end = Presentation(go_duration=('forever',),viewports=[viewport])
        
        self.send_parallel(Ssvep_dots.END_EXP)
        #self.on_quit()
        the_end.go()
        
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

                self.trial_tick()
                self.another_trial = self.another_trial - 1

            if self.another_trial == 0:   
                #self.next_to_display = "the end"
                #self.show_display()          
                self.post_mainloop()

            #self.trial_number = self.trial_number + 1
                    
    def trial_tick(self):
        """ display the text and the figures for the many letters trial """
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
            self.presentation_stopped = False

            # self.send_parallel(Ssvep_dots.START_ANIMATION)

            self.next_pic = (self.next_pic + 1) % 2
            self.cross_size =  self.fixation_spot_size  
            self.send_parallel(Ssvep_dots.START_TRIAL)
            self.f_used = (self.f_used + 1) % len(self.frequencies)

            idx, = np.where(np.array(self.marker_numbers) == self.frequencies[self.f_used])
            self.send_parallel(idx[0] + 1)

            self.look_in_the_middle = True
            #self.figure_many.parameters.go_duration = (self.time_of_trial, 'seconds') 
            #self.figure_many.go()
            print "next frequency: ", self.frequencies[self.f_used]
            print "out of: ", self.marker_numbers
            self.flickers[self.next_pic].go()
            self.short_break.go()
            to_show = (self.to_show + 1) % 2

            
            # chosen figure
            #self.previous_level = self.to_display
            #chosen_figure = self.order_of_pics[self.next_pic] - 1
        
        
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
        self.send_parallel(Ssvep_dots.COUNTDOWN_START)
        self.present_count_down.go()
        
    def short_pause_tick(self):
        """ make a short break - of time defined in init section """
        self.send_parallel(Ssvep_dots.SHORTPAUSE_START)
        self.short_break.go()      

    # ------------------- figure section ------------------------  
    def init_figures(self):
        """ initializes figures """ 
        stimulus = []
        self.stat_img = []
        # initiate blinking figures
        k = -1 

        self.flickers = []
        k = k + 1
        stimulus1 = Points(position                = self.ending_position ,
                          anchor                  = 'center',
                          size                    = self.pic_size,
                          dot_size                = self.dot_sizes,
                          color                   = self.colors[k - 1], 
                          number                  = self.numbers_in_class[k - 1],
                          no_of_all               = self.no_of_classes,
                          picture                 = self.pictures[0], 
                          frequency_used          = self.frequencies[k-1], 
                          #time_animation          = self.time_animation
                          )
        #stimulus.append(stim)
        viewport = Viewport( screen=self.screen, stimuli=[stimulus1])
        self.flickers.append(Presentation(go_duration=(self.time_of_trial,'seconds'),viewports=[viewport]))
        k = k +1
        stimulus2 = Points(position                = self.ending_position ,
                          anchor                  = 'center',
                          size                    = self.pic_size,
                          dot_size                = self.dot_sizes,
                          color                   = self.colors[k - 1], 
                          number                  = self.numbers_in_class[k - 1],
                          no_of_all               = self.no_of_classes,
                          picture                 = self.pictures[1], 
                          frequency_used          = self.frequencies[k-1], 
                          #time_animation          = self.time_animation
                          )
        viewport = Viewport( screen=self.screen, stimuli=[stimulus2])
        self.flickers.append(Presentation(go_duration=(self.time_of_trial,'seconds'),viewports=[viewport]))
        # initiate animation figures 
            
        frequency_controllers = []
        display_controllers = []
        freq_controllers = []
        #for i in range(0,len(stimulus1)): 

        freq_controllers = FunctionController(\
        during_go_func = self.get_frequency,\
        eval_frequency = Controller.EVERY_FRAME,\
        temporal_variables = \
        Controller.TIME_SEC_SINCE_GO)
        frequency_controllers = FunctionController(\
        during_go_func = self.get_frequency_time,\
        eval_frequency = Controller.EVERY_FRAME,\
        temporal_variables = \
        Controller.TIME_SEC_SINCE_GO)
        
#        display_controllers.append(FunctionController(\
#        during_go_func = self.get_display,\
#        eval_frequency = Controller.EVERY_FRAME,\
#        temporal_variables = \
#        Controller.TIME_SEC_SINCE_GO))


        #add each stimulus to the presentation set to be visible
        
        self.flickers[0].add_controller(stimulus1, 'time_passed',\
               frequency_controllers)
        self.flickers[1].add_controller(stimulus2, 'time_passed',\
               frequency_controllers)       
        self.flickers[0].add_controller(stimulus1, 'frequency_used',\
               freq_controllers)
        self.flickers[1].add_controller(stimulus2, 'frequency_used',\
               freq_controllers)                      
                            


    def get_frequency(self, t):
        return self.frequencies[self.f_used]
    
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
                        if self.et_fixate_center == True:
                            self.next_to_display = self.error_message
                        else:
                            self.next_to_display = self.error_message2
                        #self.show_display()
                        self.presentation_stopped = True
                        #self.sound_invalid.play()
                        self.play_sound("winSpaceCritStop.wav")
                        self.look_in_the_middle = False

        return t

    
    def fix_cross_controller (self, t = None):
        #returns size of the spot in the center#
        if t > 0.66 * self.time_animate:
            self.cross_size = self.fix_size
        return self.cross_size
    

    def calc_average(self):
        if len(self.classifier) == 0:
            #self.classifier.append([0,3,2,2,2,2])
            self.classifier = [[0, 3, 2, 2, 2, 2]]
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
        #self.use_hex = self.classifier.index(maxim)
        self.use_hex = 3
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
        self.send_parallel(Ssvep_dots.DISPLAY_START)
        #if self.play_ok == True:
        #    self.play_sound("winSpaceDefault.wav")
        self.present_display.go()
            
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
    
    def shortBreakInit(self):
        print "short(break)"       
        print "countdown(start)"
        
        # Create an instance of the Target2D class with appropriate parameters.
        stimulus = Target2D(size  = (40.0, 20.0),
                  color      = (1.0, 0.0, 0.0, 1.0), 
                  orientation = -45.0, 
                  on = False)
        ###############################################################
        #  Create viewport - intermediary between stimuli and screen  #
        ###############################################################
        
        viewport = Viewport(screen=self.screen, stimuli=[stimulus] )
        
        ########################################
        #  Create presentation object and go!  #
        ########################################
        
        self.h = Presentation(go_duration=(self.pauseTime,'seconds'),viewports=[viewport])
        
    def shortBreak(self):  
        self.send_parallel(training_ssvep_july.SHORTPAUSE_START)      
        self.h.go()
        self.h.between_presentations()
        #self.send_parallel(training_ssvep_july.SHORTPAUSE_END)
            
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
   
    def set_FlickerRate(self):
        i = self.startFreq     
        self.frequencies = []
        self.marker_numbers = []
        while i < self.endFreq + self.stepFrequency:
            doNotAdd = 0
            #for p in range(len(self.flicker_rate)):
            if (self.monitor_refresh_rate % i) == 0:
                doNotAdd = 1
                    #break
            if doNotAdd == 1:
                self.frequencies.append(round(i,2)) 
                self.marker_numbers.append(round(i,2))
            i = round(i,2) + self.stepFrequency
        print "new flickers: " + str(self.frequencies)  
        print "to markers: " + str(self.marker_numbers)  
        ran.shuffle(self.frequencies)
        print "played in order: " + str(self.frequencies)
        

if __name__ == '__main__':
    sd = Ssvep_dots()
    sd.on_init()
    sd.on_play()
    print "left all"
  

 
        
 