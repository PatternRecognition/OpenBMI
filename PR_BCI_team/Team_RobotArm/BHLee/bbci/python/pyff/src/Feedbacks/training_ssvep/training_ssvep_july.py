#!/usr/bin/env python

# training_ssvep_july.py -
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

#training_ssvep_july
#displays one hexagon in the middle of the screen 
#in each trial with different of given frequencies
#and different sequence of symbols chosen randomly

try:
    from FeedbackBase.MainloopFeedback import MainloopFeedback
except ImportError:
    print("Error loading modules1")

try:
    from Figure import *
except ImportError:
    print("Error loading modules2")
    
try:
    from VisionEgg.Core import *
except ImportError:
    print("Error loading modules3")

try:
    from HighlightingController import *
except ImportError:
    print("Error loading modules4")
    
try:
    import pylab as p
except ImportError:
    print("Error loading modules5")
    
try:
    from VisionEgg.MoreStimuli import *
except ImportError:
    print("Error loading modules")

try:
     import os
except ImportError:
    print("Error loading modules")   
    
try:
    import random as ran
except ImportError:
    print("Error loading modules")  
      
#class training_ssvep_july(MainloopFeedback):
class training_ssvep_july(MainloopFeedback):
        
    # TRIGGER VALUES FOR THE PARALLEL PORT (MARKERS)
    START_EXP, END_EXP = 252, 253
    COUNTDOWN_START = 100
    START_TRIAL_ANIMATION = 1  
    SHORTPAUSE_START = 249
    
    #1 - 3, 2- 3.75, 3 - 4,
    #4 - 5, 5 - 6, 6 - 7.5,
    #7 - 8, 8 - 10, 9 - 12,
    #10 - 15, 11 - 20, 12 - 24,
    #13 - 30 
    
    def on_play(self):
        print "on play called"
        MainloopFeedback.on_play(self)
        
    def on_init(self):
        print "initializing"
        self.screenSize = [600, 800]
        
        #----------------------------------
        #times of different sequences in the program
        self.pauseTime = 1           #in secs
        self.countDownTime = 5       #one countdown / sec, 
                                     #at the moment, more than 5 will not work   
        self.timeForOneFruequency = 20       #time of trial is given in seconds
        #----------------------------------        
        #frequencies to be used
        self.increment = 0 #if increment set to 0
        #   frequencies taken from the flicker_rate, otherwise
        #   they are incremented from startingFrequency to endingFrequency in steps of 
        #   stepFrequency
        #either the set of the frequencies can be given (separated by commas)
        self.flicker_rate = []  # cycles per second   
        #or the sequence of frequencies can be given from starting to 
        #ending frequency in given step
        self.startingFrequency = 4        
        self.endingFrequency = 15
        self.stepFrequency = 0.01                             
        #all of the frequencies will be checked for the correctness with the 
        #monitor refreshing rate, and the wrong ones will be eliminated
        self.monitor_Refresh = 120 
        #---------------------------------- 
        #other variables to be initialized
        #optionally by user
        self.bgColor = (0.5,0.5,0.5,0.0)   #background color
        self.hexagonSize = int(self.screenSize[0] * 0.15)  
        self.distanceBetweenHexagons = int(self.screenSize[0] * 0.40)         
        self.numberOfLoops = 3 
        self.same_screen = True    
               
        #just initialised:
        self.p = None
        self.h = None
        self.screen = None
        self.sequenceOfLetters = "1"
        self.currentFrequency = self.startingFrequency

    def pre_mainloop(self):
        self.initScreen()   #initializes screen
        if self.increment == 0:
            self.set_FlickerRate()
        self.check_if_frequencies_are_correct(self.flicker_rate)
        
        for k in range (0, self.numberOfLoops):
            self.init_graphics()
            self.shortBreakInit()
            self.countdown()    #runs predefined countdown()
            #training_ssvep_july.START_TRIAL_ANIMATION = 1
                       
            for i in range(0, len(self.flicker_rate)):
                
                self.currentFrequency = self.flicker_rate[i]    
                print "next frequency run: " + str(self.currentFrequency)
                self.sequenceOfLettersSwitch()                 
                
                training_ssvep_july.START_TRIAL_ANIMATION = self.marker_numbers.index(self.currentFrequency) + 1
                print training_ssvep_july.START_TRIAL_ANIMATION                            

                self.init_run()
                if i < len(self.flicker_rate) - 1:
                    self.shortBreak() 
                #training_ssvep_july.START_TRIAL_ANIMATION = training_ssvep_july.START_TRIAL_ANIMATION + 1 
            #shuffle the list of frequencies
            #ran.shuffle(self.flicker_rate)
        self.send_parallel(training_ssvep_july.END_EXP)
        
    def post_mainloop(self):
        print "post mainloop accessed"
        self.send_parallel(training_ssvep_july.END_EXP)
        
   
    def init_run(self):
        #calls the presentation of hexagon flashing to be run
        self.send_parallel(training_ssvep_july.START_TRIAL_ANIMATION)
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
        
    #sets the array of frequencies starting from startingFrequency, ending on
    #endingFrequency in the step of stepFrequency and eliminating all the frequencies
    #which have harmonics with the previous ones
    def set_FlickerRate(self):
        i = self.startingFrequency
        self.marker_numbers = []
        while i < self.endingFrequency + self.stepFrequency:
            doNotAdd = 0
            #for p in range(len(self.flicker_rate)):
            if (self.monitor_Refresh % i) == 0:
                doNotAdd = 1
                    #break
            if doNotAdd == 1:
                self.flicker_rate.append(round(i,2)) 
                self.marker_numbers.append(round(i,2))
            i = round(i,2) + self.stepFrequency
        print self.flicker_rate  
        print "length of flickers " +str(len(self.flicker_rate))
        print "new flickers: " + str(self.flicker_rate)
        
    def charactersDispalyed(self, t):
        return self.sequenceOfLetters
    
#switches if all the letters or only random ones will be displayed 
    def sequenceOfLettersSwitch(self):
        self.chosenSection = int(p.rand(1) * 10)
        while self.chosenSection > 5 or self.chosenSection == 0:
            self.chosenSection = int(p.rand(1) * 10)
        self.sequenceOfLetters = str(self.chosenSection)
               
    def on_pause(self):
        pass       

    def addPauses(self, matrix):
        pass
    
    def pause_tick(self):
        self.send_parallel(training_ssvep_july.END_EXP)
        self.on_quit()
    
    def _MainloopFeedback__paused(self):
        print "pause3"
    
    #those are the controllers for black and white squres in the checkboard        
    def on_or_off1(self, t):
        return int(t*self.currentFrequency * 2.0) % 2      
    
    def on_or_off2(self, t):
        return not int(t*self.currentFrequency * 2.0) % 2
                  

    #initiate the screen
    def initScreen(self):
        self.send_parallel(training_ssvep_july.START_EXP)  #not sure if that's the right place for it..
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
                           
    #specifies the countdown screen
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
        
        self.send_parallel(training_ssvep_july.COUNTDOWN_START)
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
        self.send_parallel(training_ssvep_july.SHORTPAUSE_START)      
        self.h.go()
        self.h.between_presentations()
        #self.send_parallel(training_ssvep_july.SHORTPAUSE_END)
      
     
    def init_graphics(self):
        #this function defines all the stimulus for the hexagons, and controllers to flash them
        #and adds them to the presentation p
               
        
        viewports = []        

        #create instances of the Figure class giving each of them the number of row
        #and a number of column
        #this numbers will then let know the Figure instance where it should
        #appear on the screan  
        frequency_controllers = []
        h = 0         


        figure = (Figure(radius = self.hexagonSize,
                   screenX = self.screenSize[0],
                   screenY = self.screenSize[1],
                   distance = self.distanceBetweenHexagons, 
                   on_or_off = False
                   ))
        #create Viewport using hexagon
        viewports.append(Viewport(screen=self.screen, stimuli=[figure]))


        #add controller to each of the stimuli
        func = self.on_or_off1
   
                          
        #create the frequency controller for each of the stimuli
        #functionController is predefined and uses given function                
        frequency_controllers.append(FunctionController(\
           during_go_func = func,\
           eval_frequency = Controller.EVERY_FRAME,\
           temporal_variables = \
           Controller.TIME_SEC_SINCE_GO))
          

        letter_controller = FunctionController(\
               during_go_func = self.charactersDispalyed,\
               eval_frequency = Controller.EVERY_FRAME,\
               temporal_variables = \
               Controller.TIME_SEC_SINCE_GO)
        
        self.p = Presentation(go_duration=(self.timeForOneFruequency, 'seconds'), viewports=viewports) 
                    
        self.p.add_controller(figure, 'on_or_off',\
           frequency_controllers[0]) 

        self.p.add_controller(figure, 'sequenceOfLetters', letter_controller) 


       
