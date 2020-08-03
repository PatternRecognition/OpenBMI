#!/usr/bin/env python

 
# B3Hex.py -
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

"""B4HexAll BCI Feedback. - Flashes 6 hexagons at different frequencies into the screen"""
from FeedbackBase.MainloopFeedback import MainloopFeedback
import random as ran
from Hexagon import *
from VisionEgg.Core import *
from HighlightingController import *
import pylab as p
from VisionEgg.MoreStimuli import *
import ImageDraw
    
    
class B_hex_all_july(MainloopFeedback):
    
    # TRIGGER VALUES FOR THE PARALLEL PORT (MARKERS)
    START_EXP, END_EXP = 252, 253
    COUNTDOWN_START = 0
    #different from pilot - here ALWAYS the start of trial animation is 36 
    START_TRIAL_ANIMATION = 36  
    SHORTPAUSE_START = 249
    
    #to send the marker when used, use:
    #self.send_parallel(self.marker_name)
    
    def on_play(self):
        print "on play called"
        MainloopFeedback.on_play(self)
        
    def on_init(self):
                                     #at the moment, more than 5 will not work   
        self.timeForOneFruequency = 20       #time of trial is given in seconds        
        
        print "initializing"
        self.screenSize = [600, 800]
        
        #----------------------------------
        #times of different sequences in the program
        self.pauseTime = 1           #in secs
        self.countDownTime = 0       #one countdown / sec, 
                                     #at the moment, more than 5 will not work  
        self.time_of_trial = 3       #time of trial is given in seconds 
        #self.time_of_trial = 5       #time of trial is given in seconds
        #!!!!!!!!!!!!!!!!here the frequencies for each of the hexagons are set

        self.number_of_trials = 15   
        #----------------------------------        
        #frequencies to be used
        self.flicker_rate = [3.0, 4.0, 5.0, 7.0, 11.0, 13.0]  # cycles per second  
                            
        self.monitor_Refresh = 120 
        #---------------------------------- 
        #---------------------------------- 
        #other variables to be initialized
        #optionally by user
        self.bgColor = (0.5,0.5,0.5,0.0)   #background color
        self.hexagonSize = int(self.screenSize[0] * 0.15)  
        self.distanceBetweenHexagons = int(self.screenSize[0] * 0.40)         
        self.same_screen = True    
        self.add_cross_in_center = True
        self.cross_color = (0.0, 0.0, 0.0)
        self.cross_radius_and_width = (10.0, 1.0)
        #will later be set to the presentation of flashing stimuli and for break
        self.p = None
        self.h = None
        self.screen = None    
        self.chosenSection = 1
        self.sequenceOfLetters = "not yet"
        

    def pre_mainloop(self):
        self.initScreen()
        #self.countdown()
        #self.init_graphics()
        #self.shortBreakInit()
        #self.check_if_frequencies_are_correct(self.flicker_rate)
        #run trials as many times as you are told
        self.countdown()    #runs predefined countdown()        
        for k in range(0, self.number_of_trials):
            self.init_graphics()
            self.shortBreakInit()
       
            
            self.sequenceOfLettersSwitch()
            print "from main: " + self.sequenceOfLetters
            self.init_run()
            self.shortBreak()
        self.send_parallel(B_hex_all_july.END_EXP)


    def post_mainloop(self):
        print "post mainloop accessed"
        self.send_parallel(B_hex_all_july.END_EXP)
        
   
    def init_run(self):
        #calls the presentation of hexagon flashing to be run
        self.send_parallel(B_hex_all_july.START_TRIAL_ANIMATION)
        self.p.go()
        self.p.between_presentations() 
        
    #checks if the frequencies are correct for monitor refresh rate
    #if they are not it gives worning message, but still pass them on
    def check_if_frequencies_are_correct(self, stimuli_frequencies):
        #gets current monitor refresh rate
       monitor_refresh = VisionEgg.config.VISIONEGG_MONITOR_REFRESH_HZ
       if filter((lambda x: x > monitor_refresh), stimuli_frequencies):
           print "one of your frequencies is larger than your refresh rate"
           print "your monitor refresh rate:"
           print monitor_refresh
           print "requested frequencies:"
           print stimuli_frequencies
       else:
           # check for every specified frequency
           for frequency in stimuli_frequencies:
               # if we are theoretically able to render it
               if (monitor_refresh % frequency) != 0:
                   print "one of your frequency rate is not correct: "
                   print frequency

    def pause_tick(self):
        print("Pause")
        
    def on_pause(self):
        pass       

    def addPauses(self, matrix):
        pass
    
    def pause_tick(self):
        self.send_parallel(B_hex_all_july.END_EXP)
        self.on_quit()
    
    def _MainloopFeedback__paused(self):
        print "pause3"
    
    #those are the controllers for each of the hexagons        
    def on_or_off1(self, t):
        return int(t*self.flicker_rate[0]*2.0) % 2      
    
    def on_or_off2(self, t):
        return int(t*self.flicker_rate[1]*2.0) % 2
    
    def on_or_off3(self, t):
        return int(t*self.flicker_rate[2]*2.0) % 2
    
    def on_or_off4(self, t):
        return int(t*self.flicker_rate[3]*2.0) % 2
    
    def on_or_off5(self, t):
        return int(t*self.flicker_rate[4]*2.0) % 2
    
    def on_or_off6(self, t):
        return int(t*self.flicker_rate[5]*2.0) % 2                

    #initiate the screen
    def initScreen(self):
        self.send_parallel(B_hex_all_july.START_EXP)  #not sure if that's the right place for it..
        VisionEgg.config.VISIONEGG_GUI_INIT = 0
        
        VisionEgg.config.VISIONEGG_SCREEN_W = self.screenSize[1]
        VisionEgg.config.VISIONEGG_SCREEN_H = self.screenSize[0]
        if self.same_screen == False:
            os.environ['SDL_VIDEO_WINDOW_POS']="-800,1000"
        else:
            os.environ['SDL_VIDEO_CENTERED']="center"
        #VisionEgg.config.VISIONEGG_FULLSCREEN = 1
        self.screen = get_default_screen()
        self.screen.parameters.bgcolor = self.bgColor
        
    def charactersDispalyed(self, t):
        return self.sequenceOfLetters
    
#switches if all the letters or only random ones will be displayed 
    def sequenceOfLettersSwitch(self):
        self.chosenSection = int(p.rand(1) * 10)
        while self.chosenSection > 5 or self.chosenSection == 0:
            self.chosenSection = int(p.rand(1) * 10)
        if self.sequenceOfLetters == "all":
            self.sequenceOfLetters = str(self.chosenSection)
        else:
            self.sequenceOfLetters = "all"

# ------------------------------------------------------------  
    #defines the controller for each of the bars in the countdown screen      
    def on_or_offCount1(self, t):
        if t > 1:
            return False
        else:
            return True

    def on_or_offCount2(self, t):
        if t > 2:
            return False
        else:
            return True
        
    def on_or_offCount3(self, t):
        if t > 3:
            return False
        else:
            return True
        
    def on_or_offCount4(self, t):
        if t > 4:
            return False
        else:
            return True
        
    def on_or_offCount5(self, t):
        if t > 5:
            return False
        else:
            return True  
                           
    #specifies the countdow screen
    def countdown(self):

        middleScreen = (self.screenSize[1] / (self.countDownTime + 1))

        yPosition = (self.screenSize[0]) / 2     
        stimulus = []
        viewports = []
        onOrOffControllers = []
        
        for g in range(0, self.countDownTime):   
            #display the bars in the middle of the screen the same distance from each other
            xPosition =  middleScreen * (g + 1)   
            # Create an instance of the Target2D class with appropriate parameters.           
            stimulus.append(Target2D(size  = (64.0, 250.0),
                      color      = (0.0,0.0,0.0,1.0), # Set the target color (RGBA) black
                      orientation = 0.0,
                      position = (xPosition, yPosition)))
            
            ###############################################################
            #  Create viewport - intermediary between stimuli and screen  #
            ###############################################################
            viewports.append(Viewport(screen=self.screen, stimuli=[stimulus[g]]))
            
        f = Presentation(go_duration=(self.countDownTime,'seconds'),viewports=viewports)            
        for g in range(0, self.countDownTime):  
            func = None
            
            if g == 0:   
                func = self.on_or_offCount1
            elif g == 1:
                func = self.on_or_offCount2
            elif g == 2:
                func = self.on_or_offCount3                
            elif g == 3:
                func = self.on_or_offCount4                
            elif g == 4:
                func = self.on_or_offCount5            
                    
            #create the controller for each of the bars    
            onOrOffControllers.append(FunctionController(\
            during_go_func = func,\
            eval_frequency = Controller.EVERY_FRAME,\
            temporal_variables = \
            Controller.TIME_SEC_SINCE_GO))
            
            #add each stimulus to the presentation set to be visible
            f.add_controller(stimulus[g], 'on',\
                   onOrOffControllers[g])
        ########################################
        #  Create presentation object and go!  #
        ########################################
        
        #countdown runs only once so not necessary to create separate function for it
        
        self.send_parallel(B_hex_all_july.COUNTDOWN_START)
        f.go()
        f.between_presentations()
    
    
        
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
        self.send_parallel(B_hex_all_july.SHORTPAUSE_START)      
        self.h.go()
        self.h.between_presentations()
        
    def insertCharacterAtGivenPosition(inputString, char, position):
        if position > len(inputString):
            return inputString
        else:
            newString = ""
            for i in range(position):
                newString += inputString[i]
            newString += str(char)
            for i in range(position, len(inputString)):
                newString += inputString[i]    
            return newString
      
     
    def init_graphics(self):
        #this function defines all the stimulus for the hexagons, and controllers to flash them
        #and adds them to the presentation p
               
        hexagons = []        

        #create 6 instances of the Hexagon class giving each of them the number
        #this number will then let know the Hexagon instance where it should
        #appear on the screan           
        #fix_cross = VisionEgg.Textures.FixationCross(on = True, position = (self.screenSize[1]/ 2, self.screenSize[0]/ 2))
        if self.add_cross_in_center == True:
            fix_cross1 = VisionEgg.Textures.TextureStimulus(position = (self.screenSize[1]/ 2, (self.screenSize[0]/ 2)),
                                                             color = self.cross_color, size = self.cross_radius_and_width)
            fix_cross2 = VisionEgg.Textures.TextureStimulus(position = ((self.screenSize[1]/ 2), self.screenSize[0]/ 2),
                                                             angle = 90.0, color = self.cross_color, size = self.cross_radius_and_width)
            fix_cross3 = VisionEgg.Textures.TextureStimulus(position = (self.screenSize[1]/ 2, (self.screenSize[0]/ 2)),
                                                             angle = 180.0, color = self.cross_color, size = self.cross_radius_and_width)
            fix_cross4 = VisionEgg.Textures.TextureStimulus(position = ((self.screenSize[1]/ 2), self.screenSize[0]/ 2),
                                                             angle = 270.0, color = self.cross_color, size = self.cross_radius_and_width)
        
        #fix_cross = VisionEgg.Gratings.SinGrating2D()
        viewport=Viewport(screen=self.screen)#, stimuli=[hexagons[s],fix_cross]))
        for s in range(0,6):                
            hexagons.append(Hexagon(radius = self.hexagonSize,
                               screenX = self.screenSize[0],
                               screenY = self.screenSize[1],
                               distance = self.distanceBetweenHexagons, 
                               hexNumber = s,
                               on_or_off = False
                               #fix_cross = self.add_cross_in_center
                               ))
            #create Viewport using each of these hexagons
            viewport.parameters.stimuli.append(hexagons[s])
        if self.add_cross_in_center == True:
            viewport.parameters.stimuli.append(fix_cross1)
            viewport.parameters.stimuli.append(fix_cross2)
            viewport.parameters.stimuli.append(fix_cross3)
            viewport.parameters.stimuli.append(fix_cross4)            
        #if self.add_cross_in_center == True:
        #    viewport.parameters.stimuli.append(rect1)
        #    viewport.parameters.stimuli.append(rect2)
        #self.parameters.fix_cross == True: 
            #self.fix_cross.draw()
             
        #create the presentation with all the hexagons created                      
        self.p = Presentation(go_duration=(self.time_of_trial, 'seconds'), viewports=[viewport]) 

        letter_controller = FunctionController(\
               during_go_func = self.charactersDispalyed,\
               eval_frequency = Controller.EVERY_FRAME,\
               temporal_variables = \
               Controller.TIME_SEC_SINCE_GO)

        #add controller to each of the stimuli
        frequency_controllers = []
        for s in range(0, 6):
            if s == 0:
                func = self.on_or_off1
            elif s == 1:
                func = self.on_or_off2
            elif s == 2:                
                func = self.on_or_off3
            elif s == 3:            
                func = self.on_or_off4
            elif s == 4:   
                func = self.on_or_off5
            elif s == 5:                   
                func = self.on_or_off6  
                              
            #create the frequency controller for each of the stimuli
            #functionController is predefined and uses given function                
            frequency_controllers.append(FunctionController(\
               during_go_func = func,\
               eval_frequency = Controller.EVERY_FRAME,\
               temporal_variables = \
               Controller.TIME_SEC_SINCE_GO))

            
            
            self.p.add_controller(hexagons[s], 'on_or_off',\
               frequency_controllers[s]) 
            self.p.add_controller(hexagons[s], 'sequenceOfLetters', letter_controller)      
