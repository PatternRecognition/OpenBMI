
#!/usr/bin/env python


# FeedbackCursorArrow.py -
# Copyright (C) 2008-2009  Bastian Venthur, Simon Scholler
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

"""CursorArrow BCI Feedback."""


import random
import sys
import math
import os

import pygame

from FeedbackBase.MainloopFeedback import MainloopFeedback


class FeedbackCursorArrow3(MainloopFeedback):

################################################################################
# Derived from Feedback
#vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

# Markers written to parallel port
#    1:  on first direction
#    2:  on second direction
#    3:  on third direction (if present)
#   11: trial ended with cursor correctly on the first direction
#   12: trial ended with cursor correctly on the second direction
#   13: trial ended with cursor correctly on the third direction
#   21: trial ended with cursor erroneously on the first direction
#   22: trial ended with cursor erroneously on the second direction
#   23: trial ended with cursor erroneously on the third direction
#   24: trial ended by time-out policy='reject'
#   30: countdown starts
#   33: in position control, cursor becomes active in center area
#   41: first touch with cursor correctly on the first direction
#   42: first touch with cursor correctly on the second direction
#   43: first touch with cursor correctly on the third directions
#   51: first touch with cursor erroneously on the first direction
#   52: first touch with cursor erroneously on the second direction
#   53: first touch with cursor erroneously on the third direction
#   70: start of adaptation phase 1 (rotating cursor)
#   71: end of adaptation phase 1
#  200: init of the feedback
#  210: game status changed to 'play'
#  211: game status changed to 'pause'
#  254: game ends

    #FEEDBACK GENERATION POLICIES
    AUTOMATIC_EQ_TRIALS_PER_CLASS = 1
    AUTOMATIC_DIFF_TRIALS_PER_CLASS = 2    
    AUTOMATIC_EQ_TRIALS_PER_CLASS_PER_BLOCK = 3
    
    INTRO_START = 241
    INTRO_END = 242
        
    #for calibration:
    START_TRIAL = 100
    START_POSITION_CONTROL = 101
    
    TARGET = [1, 2, 3]
    HIT_CURSOR = [11, 12, 13]
    
    MISS_CURSOR = [21, 22, 23]
        
    TRIAL_END_REJECT = 24

    COUNTDOWN_START = 30    

    HIT_FT_CURSOR = [41, 42, 43]
    MISS_FT_CURSOR = [51, 52, 53]

    INIT_FEEDBACK = 200
    
    GAME_STATUS_PLAY = 210
    # game_status_pause is never used
    GAME_STATUS_PAUSE = 211
    GAME_OVER = 254
    
    
    def on_init(self):
        """
        Initializes the variables and stuff, but not pygame itself.
        """
       
        self.g_rel =  1
        self.g_ab   = 1
        self.control = 'relative'
        self.eth = None
        
        self.introtext = [""]
        MainloopFeedback.on_init(self)
        self.logger.debug("on_init")
        self.send_parallel(self.INIT_FEEDBACK)
        
        self.restrict_control_signal = True
        self.durationUntilBorder = 100
        self.durationUntilBorderMsec = 2000
        
        self.durationIndicateGoal = 100
        
        self.pauseBeforeFixationCross = 2000
        self.durationFixationCross = 2000
        self.durationActivationPeriod = 4000
        
        self.hitMissDuration =  2500
        self.pauseDuration = 16000
        self.introDuration = 2000        

        self.countdownFrom = 15
        self.shortPauseCountdownFrom = 16
                    
        self.durationTrial = self.pauseBeforeFixationCross + self.durationFixationCross + self.durationActivationPeriod                            
    
        self.trialGenerationPolicy = self.AUTOMATIC_EQ_TRIALS_PER_CLASS_PER_BLOCK
        self.pauseAfter = 20        
        
        self.trialsPerClass = 50
        self.classes = ['left','right','foot']
        # if you have classes = ['left', 'foot'] and want foot to have marker 2 put classesDirections = ['left','foot'] and classesMarkers = [1, 2]
        # self.classesDirections and self.classesMarkers should correspond to self.classes
        self.classesDirections = ['left','right','foot']
        self.classesMarkers = [1, 2, 3]
        
        self.numClassTrials = [25, 25, 25, 25, 25, 25]
            
        self.FPS =  50
        self.fullscreen =  False
        self.screenWidth =  1900
        self.screenHeight = 1200        
        self.screenPos = [-2250, 200]
       
        self.dampedMovement = False
        #self.damping = 'linear'
        self.damping = 'distance'
        self.calibration = False
        
        self.intro = True
        self.pause = False
        self.quit = True
        self.quitting = True        
        self.gameover = False
        self.shortPause = False
        
        self.countdown = False
                
        self.hit = False
        self.miss = False
        
        self.indicateGoal = True
        
        self.firstTickOfTrial = True
        self.firstTickOfFixationCross = True
        self.firstTickOfArrow = True
        
        self.showsPause, self.showsShortPause = False, False
        self.showsHitMiss, self.showsGameover = False, False
        
        self.elapsed, self.trialElapsed, self.countdownElapsed, self.introElapsed = 0,0,0,0
        self.hitMissElapsed, self.shortPauseElapsed, self.indicateGoalElapsed = 0,0,0
        
        self.completedTrials = 0
        
        self.f = 0
        self.hitMiss = [0,0]
        self.resized = False
        self.pos = 0
        self.targetDirection = 0
        
        self.arrowPointlist = [(.5,0), (.5,.33), (1,.33), (1,.66), (.5,.66), (.5,1), (0,.5)]
        self.arrowColor = (127, 127, 127)
        self.borderColor = self.arrowColor
        self.backgroundColor = (64, 64, 64)
        self.cursorColor = (100, 149, 237)
        self.fontColor = self.cursorColor
        self.countdownColor = (237, 100, 148)
        self.borderWidthRatio = 0.4     # in pixels

        
        #=============
        # NOTE: only one of the following variables should be True
        self.hitIfLateral = True  
        self.reject = False
        #=============
     
        # How many degrees counter clockwise to turn an arrow pointing to the 
        # left to point at left, right and foot
        self.LEFT, self.RIGHT, self.DOWN, self.UP = 'left', 'right', 'foot', 'up'
        self.directions = {self.LEFT: 0, self.RIGHT: 180, self.DOWN: 90, self.UP: 270}
             
        #if not hasattr(self, 'trialGenerationPolicy'): 
        #self.trialGenerationPolicy = self.AUTOMATIC_EQ_TRIALS_PER_CLASS                              
        
                
        
    def generate_trials(self):
        """
        Generate a list of trial directions. Class probabilities are not retained within feedback blocks.
        """
        self.completedTrials = 0
        self.trialDirections = [0] * int(self.trialsPerClass*len(self.classes))
        self.totalTrials = len(self.trialDirections)
        
        for c in xrange(len(self.classes)): 
            self.trialDirections[int(self.trialsPerClass)*c:int(self.trialsPerClass)*(c+1)] = [c+1]* int(self.trialsPerClass)        
            self.markers[c]= self.TARGET[self.classesMarkers[c]-1]
         
        random.shuffle(self.trialDirections)
           
    
    def generate_trials2(self):
        """
        todo.
        """      
        self.totalTrials = 0
        self.completedTrials = 0
        
        aux = 0
        
        for c in xrange(len(self.numClassTrials)):
            aux = aux + self.numClassTrials[c]                    
               
        self.trialDirections = [0] * aux
        
        for c in xrange(len(self.numClassTrials)):
            self.markers[c]= self.TARGET[self.classesMarkers[c]-1]            
            self.trialDirections[self.totalTrials:self.totalTrials + self.numClassTrials[c]] = [self.markers[c]]* self.numClassTrials[c]        
            self.totalTrials = self.totalTrials + self.numClassTrials[c];
                                 
        self.totalTrials = len(self.trialDirections)        

        random.shuffle(self.trialDirections)            
            
            
    def generate_trials3(self):
        """
        Generate a list of trial directions. Class probabilities are retained within feedback blocks.
        """        
        nC = len(self.classes)
        self.completedTrials = 0
        self.trialDirections = []        
        self.totalTrials = int(self.trialsPerClass*nC)

        # rounding on self.pauseAfter so that the same nnumber of trials per class is possible in each block
        self.pauseAfter = self.pauseAfter - self.pauseAfter%nC;

        # rounding self.totalTrials so that is a multiple of self.pauseAfter. Put
        if self.totalTrials > self.pauseAfter:
            nB = self.totalTrials//self.pauseAfter
            self.totalTrials = nB * self.pauseAfter
            self.trialsPerClass = self.totalTrials/nC
        else:
            nB = int(1)            
            self.pauseAfter = self.totalTrials

        for c1 in xrange(nB):
            n = self.pauseAfter/nC
            trialDirectionsBlock = []
            for c2 in xrange(nC):                 
                trialDirectionsBlock[n*c2:n*(c2+1)-1] = [c2+1]* n     
                self.markers[c2] = self.TARGET[self.classesMarkers[c2]-1]         
                random.shuffle(trialDirectionsBlock)                   
            self.trialDirections[c1*self.pauseAfter:(c1+1)*self.pauseAfter-1] = trialDirectionsBlock                    
        print self.markers

    def pre_mainloop(self):
        print "#### PREMAINLOOP ####"
     
        self.markers = [0] * len(self.classes)        
   
        if self.trialGenerationPolicy == self.AUTOMATIC_EQ_TRIALS_PER_CLASS:                    
            self.generate_trials()
        elif self.trialGenerationPolicy == self.AUTOMATIC_DIFF_TRIALS_PER_CLASS:
            self.generate_trials2()        
        else:
            self.generate_trials3()

        print("    trials = %d" % self.totalTrials);
        print("    classes = %s, directions = %s" % (self.classes, self.classesDirections));        
        print("    trialDirections = %s" % self.trialDirections);  
        print("    classesMarkers = %d, %d" % (self.classesMarkers[0], self.classesMarkers[1]));

        self.logger.debug("on_play")
        self.init_pygame()
        self.init_graphics()
        self.send_parallel(self.GAME_STATUS_PLAY)


    def post_mainloop(self):
        self.logger.debug("on_quit")        
        if self.eth is not None:
          self.eth.cleanup()

        pygame.quit()


    def tick(self):
        self.process_pygame_events()
        pygame.time.wait(10)
        self.elapsed = self.clock.tick(self.FPS)


    def play_tick(self):
        """
        Decides in wich state the feedback currently is and calls the apropriate
        tick method.
        """
        if self.intro:
            self.intro_tick();
        elif self.pause:
            self.pause_tick()
        elif self.hit or self.miss:
            self.hit_miss_tick()
        elif self.gameover:
            self.end_of_experiment_tick()
        elif self.countdown:
            self.countdown_tick()
        elif self.shortPause:
            self.short_pause_tick()
        elif self.indicateGoal:
            self.indicate_goal_tick()
        else:
            if(self.calibration == True):
                self.calib_trial_tick()
            else:
                self.feedback_trial_tick()


    def on_control_event(self, data):           
        self.f = data["cl_output"]
    
               
    def calib_trial_tick(self):
        """
        One tick of the trial loop.
        """
        self.trialElapsed += self.elapsed

        # Teste ob erster Tick im Trial
        if self.firstTickOfTrial:
 
            self.send_parallel(self.START_TRIAL)
                
            self.firstTickOfTrial = False            
            self.trialElapsed = 0
            self.pos = 0
            self.counter = 1
            self.nDampTicks = 0     
            self.cursorTransition = False   
            self.firstTickOfFixationCross = True        
            self.firstTickOfArrow = True 
        # Teste ob zeit fuer alten Trial abgelaufen ist
        if self.trialElapsed >= self.durationTrial:
                
            self.miss = True
            #self.check_for_hit_miss(); 
            return
       
        if self.trialElapsed < self.pauseBeforeFixationCross:
            self.draw_all(False, False)
        elif self.trialElapsed < self.pauseBeforeFixationCross + self.durationFixationCross:
            if self.firstTickOfFixationCross:
                self.firstTickOfFixationCross = False
                self.send_parallel(self.START_POSITION_CONTROL)
                
            self.draw_all(False, True)
            
        elif self.trialElapsed < self.pauseBeforeFixationCross + self.durationFixationCross + self.durationActivationPeriod:
            if self.firstTickOfArrow:
                print("class %s" % self.classes[self.targetDirection])
                self.firstTickOfArrow = False
                print self.markers
                print self.targetDirections
                self.send_parallel(self.markers[self.targetDirection])           
                            
            self.draw_all(True, False)        
        else:
            self.draw_all()
                   
    def feedback_trial_tick(self):
        """
        One tick of the trial loop.
        """
        self.trialElapsed += self.elapsed

        # Teste ob erster Tick im Trial
        if self.firstTickOfTrial:
            self.firstTickOfTrial = False            
            self.trialElapsed = 0
            self.pos = 0
            self.counter = 1
            self.nDampTicks = 0            
            self.cursorTransition = False           
            self.firstTickOfFixationCross = True   
            self.firstTickOfArrow = True

        # Teste ob zeit fuer alten Trial abgelaufen ist
        if self.trialElapsed >= self.durationTrial:            
            self.check_for_hit_miss(); 
            return
        
        if self.trialElapsed < self.pauseBeforeFixationCross:
            self.draw_all(False, False)
        elif self.trialElapsed < self.pauseBeforeFixationCross + self.durationFixationCross:
            if self.firstTickOfFixationCross:
                self.firstTickOfFixationCross = False
                self.send_parallel(self.markers[self.targetDirection])                
                
            self.draw_all(True, True)
        elif self.trialElapsed == self.pauseBeforeFixationCross + self.durationFixationCross:
            self.send_parallel(self.START_POSITION_CONTROL)

        else:
            # Calculate motion of cursor
            #self.f = (self.targetDirection-0.5)*2   #TODO: remove HACK
            
             # Move bar according to classifier output
            if self.control == "absolute":
                self.pos = self.f * self.g_ab 
            elif self.control == "relative":
                self.g_rel= self.s1/(self.FPS*self.durationUntilBorderMsec/1000.0)
                if self.restrict_control_signal:
                    self.f= max(-1, min(1, self.f))
                self.pos += self.f * self.g_rel
                
            # send marker if cursor hits the border for the first time
            if abs(self.pos)>self.s1 and not self.cursorTransition:
                self.cursorTransition = True
                if self.pos<0 and self.targetDirection==0:
                    self.send_parallel(self.HIT_FT_CURSOR[self.markers[0]-1])
                elif self.pos>0 and self.targetDirection==1:
                    self.send_parallel(self.HIT_FT_CURSOR[self.markers[1]-1])    
                elif self.pos<0 and self.targetDirection==1:    
                    self.send_parallel(self.MISS_FT_CURSOR[self.markers[0]-1])
                else:
                    self.send_parallel(self.MISS_FT_CURSOR[self.markers[1]-1])   
                                         
            self.update_cursor()
            
            if abs(self.pos) >= self.size/2:
                self.check_for_hit_miss(); return
            
            self.draw_all()


    def indicate_goal_tick(self):    
        """
        Indicate goal before start of trial (i.e. before position control).
        """
        if self.indicateGoalElapsed==0:

            self.targetDirection = self.trialDirections[self.completedTrials] - 1
            
            #print("Completed trials = %d, targetDirection = %d, len(classes) = %d" % (self.completedTrials, self.targetDirection, len(self.classes)))
            self.myarrow = pygame.transform.rotate(self.arrow, self.directions[self.classesDirections[self.classesMarkers[self.targetDirection]-1]])
            
            self.myarrowRect = self.myarrow.get_rect(center=self.screen.get_rect().center)
            self.cursorRect.center = self.screen.get_rect().center                      
            
                
        self.indicateGoalElapsed += self.elapsed
        if self.durationIndicateGoal<self.indicateGoalElapsed:
            self.indicateGoalElapsed = 0
            self.indicateGoal = False
        self.draw_all(False, False)
        
           
    def pause_tick(self):
        """
        One tick of the pause loop.
        """
        if self.showsPause:
            return
        self.do_print("Pause", self.fontColor, 30)
        self.showsPause = True

    def intro_tick(self):
        """
        One tick of the intro loop.
        """
        if(self.introElapsed == 0):
            self.send_parallel(self.INTRO_START)
            
        self.introElapsed += self.elapsed
        
        if self.introElapsed >= self.introDuration:
            self.showsIntro = False
            self.intro = False
            self.countdown = True
            self.shortIntroElapsed = 0
            self.elapsed = 0
            self.send_parallel(self.INTRO_END)
            return
        
        self.draw_init()
        #self.do_print(self.introtext, self.countdownColor, 50, True)
        self.do_print("%s vs. %s" % (self.classes[0], self.classes[1]), self.cursorColor, self.size/5, True)

    def short_pause_tick(self):
        """
        One tick of the short pause loop.
        use self.hitMissDuration which originally was used after each trials, to show the performance hit:miss after each block
        """
        self.shortPauseElapsed += self.elapsed
        if self.shortPauseElapsed >= self.pauseDuration + self.hitMissDuration + 500:
            self.showsShortPause = False
            self.shortPause = False
            self.shortPauseElapsed = 0
            return
        
        if self.shortPauseElapsed < self.hitMissDuration:
            self.do_print(["(%i : %i)" % (self.hitMiss[0], self.hitMiss[1])])            
        elif self.shortPauseElapsed > self.hitMissDuration + 500:
            t = (self.shortPauseCountdownFrom * 1000 - self.shortPauseElapsed + self.hitMissDuration + 500) / 1000
            self.draw_init()
            self.do_print(str(t), self.countdownColor, self.size/3, True)

    
    def countdown_tick(self):
        """
        One tick of the countdown loop.
        """
        if self.countdownElapsed==0:
            self.send_parallel(self.COUNTDOWN_START)
            
        self.countdownElapsed += self.elapsed
        if self.countdownElapsed >= self.countdownFrom * 1000:
            self.countdown = False
            self.indicateGoal = True
            self.countdownElapsed = 0
            
            return
        t = (self.countdownFrom * 1000 - self.countdownElapsed) / 1000
        self.draw_init()
        self.do_print(str(t), self.countdownColor, self.size/3, True)

        
    def end_of_experiment_tick(self):
        """
        One tick of the game over loop.
        """
        if self.showsGameover:
            return
        
        if self.calibration:
            self.do_print("END")
            self.send_parallel(self.GAME_OVER)
        else:
            self.do_print(["Game Over! (%i : %i)" % (self.hitMiss[0], self.hitMiss[1])])
            self.send_parallel(self.GAME_OVER)

        self.showsGameover = True

        
    def hit_miss_tick(self):
        """
        One tick of the Hit/Miss loop.
        """
        self.hitMissElapsed += self.elapsed
        if self.hitMissElapsed >= self.hitMissDuration:
            self.hitMissElapsed = 0
            self.hit, self.miss, self.reject = False, False, False            
            self.showsHitMiss = False
            self.indicateGoal = True
            return
        if self.showsHitMiss:
            return
        
        self.completedTrials += 1
        self.firstTickOfTrial = True
        
        if(self.calibration == False):
            s = ""
            if self.hit:
                self.hitMiss[0] += 1
            elif self.miss:
                self.hitMiss[-1] += 1

            if self.indicateGoal:
                if self.hit:
                    s = "Hit"
                elif self.miss:
                    s = "Miss"
                elif self.reject:
                    s = "Draw"

        if self.completedTrials % self.pauseAfter == 0:
            self.shortPause = True
            
        if self.completedTrials >= self.totalTrials:
            self.gameover = True
        
        self.draw_all(False, False)
        
        if(self.calibration == False):
            print("%d:%d" % (self.hitMiss[0], self.hitMiss[1])) 
            self.do_print(s, None, None, True)
            
        pygame.display.update()
        self.showsHitMiss = True
    
    def damp_movement(self):
        if self.damping == 'linear':
            if abs(self.pos)>self.s1:
                s = abs(self.pos)-self.s1
                t = self.t2 - (math.sqrt(4*self.s2*(self.s2-s))/self.v0)
                return self.v0 * ((self.t2-t)/self.t2)
            else:
                return self.v0
        elif self.damping == 'distance':
            if self.trialElapsed == 0:
                self.v0 = self.s1/(self.FPS*self.durationUntilBorder/1000.0)
            if abs(self.pos)>self.s1:
                s = abs(self.pos)-self.s1
                return (1.0*(self.borderWidth-s)/self.borderWidth)*self.v0
            else:
                return self.v0
            
    def check_for_hit_miss(self):
        if self.reject:
            if abs(self.pos) <= self.arrowRect.width/2:
                self.send_parallel(self.TRIAL_END_REJECT)
                self.reject = True; return
                
        if (self.pos<0 and self.targetDirection==0) or (self.pos>0 and self.targetDirection==1):

            if (self.pos<0 and self.targetDirection==0):
                self.send_parallel(self.HIT_CURSOR[self.markers[0]-1]) 
            else:
                self.send_parallel(self.HIT_CURSOR[self.markers[1]-1])
                 
            if self.hitIfLateral:
                self.hit = True
                
        else:
            self.miss = True
            if self.pos<0:
                self.send_parallel(self.MISS_CURSOR[self.markers[0]-1]) 
            else:
                self.send_parallel(self.MISS_CURSOR[self.markers[1]-1])
                 
    

    
    def update_cursor(self):
        """
        Update the cursor position.
        """
     
        self.cursorRect.center = self.backgroundRect.center
        self.direction = self.classes[0]
        
        if self.pos > 0:
            self.direction = self.classes[1]
            
        border = self.borderRect.width/2
        self.pos = min(border, max(-border, self.pos));
        arrowPos = { self.LEFT  : (-abs(self.pos),0),
                     self.RIGHT : (abs(self.pos),0),
                     self.DOWN  : (0,abs(self.pos)),
                     self.UP    : (0,-abs(self.pos))
                   }[self.direction]
        self.cursorRect.move_ip(arrowPos)
        
    
    def do_print(self, text, color=None, size=None, superimpose=False):
        """
        Print the given text in the given color and size on the screen.
        If text is a list, multiple items will be used, one for each list entry.
        """
        if not color:
            color = self.fontColor
        if not size:
            size = self.size/10
        
        font = pygame.font.Font(None, size)
        if not superimpose:
            self.draw_init()
        
        if type(text) is list:
            height = pygame.font.Font.get_linesize(font)
            top = -(2*len(text)-1)*height/2
            for t in range(len(text)):
                surface = font.render(text[t], 1, color)
                self.screen.blit(surface, surface.get_rect(midtop=(self.screenWidth/2, self.screenHeight/2+top+t*2*height)))
        else:
            surface = font.render(text, 1, color)
            self.screen.blit(surface, surface.get_rect(center=self.screen.get_rect().center))
        pygame.display.update()
        
        
    def init_graphics(self):
        """
        Initialize the surfaces and fonts depending on the screen size.
        """
        self.screen = pygame.display.get_surface()
        (self.screenWidth, self.screenHeight) = (self.screen.get_width(), self.screen.get_height())
        self.size = min(self.screen.get_height(), self.screen.get_width())
        self.borderWidth = int(self.size*self.borderWidthRatio/2)
        self.offsetX = (self.screenWidth-self.size)/2
        self.s1 = self.size/2-self.borderWidth
        self.s2 = self.borderWidth
            
        if self.dampedMovement:
            self.v0 = self.s2/(1.0*self.t2) + self.s1/(2.0*self.t1)
            self.dampingTime = self.durationTrial-self.durationUntilBorder
            self.t1 = self.FPS * self.durationUntilBorder/1000
            self.t2 = self.FPS * self.dampingTime/1000
        else:
            self.v0 = (self.size * 0.5) / (self.durationTrial*self.FPS/1000.0)
        
        # arrow
        scale = self.size / 3
        scaledArrow = [(P[0]*scale, P[1]*scale) for P in self.arrowPointlist]
        self.arrow = pygame.Surface((scale, scale))
        self.arrowRect = self.arrow.get_rect(center=self.screen.get_rect().center)
        self.arrow.fill(self.backgroundColor)
        pygame.draw.polygon(self.arrow, self.arrowColor, scaledArrow)
        
        # cursor
        scale = self.size / 5
        self.cursor = pygame.Surface((scale, scale))
        self.cursorRect = self.cursor.get_rect(center=self.screen.get_rect().center)
        self.cursor.set_colorkey((0,0,0))
        pygame.draw.line(self.cursor, self.cursorColor, (0,scale/2),(scale,scale/2), 10)
        pygame.draw.line(self.cursor, self.cursorColor, (scale/2,0),(scale/2,scale), 10)
        
        # background + border
        self.background = pygame.Surface((self.screen.get_width(), self.screen.get_height()))
        self.backgroundRect = self.background.get_rect(center=self.screen.get_rect().center)
        self.background.fill((0,0,0))
        self.border = pygame.Surface((self.size, self.size))
        self.border.fill(self.borderColor)
        self.borderRect = self.border.get_rect(center=self.screen.get_rect().center)
        self.inner = pygame.Surface((self.size-2*self.borderWidth, self.size-2*self.borderWidth))
        self.inner.fill(self.backgroundColor)
        self.innerRect = self.inner.get_rect(center=self.screen.get_rect().center)

      
            
        if self.resized:
            self.resized = False
            target = self.classes[self.targetDirection]
            self.pos = (1.0*self.size/self.size_old) * self.pos
            self.update_cursor()
            
            self.myarrow = pygame.transform.rotate(self.arrow, self.directions[target])
            self.myarrowRect = self.myarrow.get_rect(center=self.screen.get_rect().center)
            self.draw_all()
        
    def init_pygame(self):
        """
        Set up pygame and the screen and the clock.
        """
        os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (self.screenPos[0],
                                                        self.screenPos[1])
        pygame.init()
        pygame.display.set_caption('CursorArrow Feedback')
        if self.fullscreen:
            self.screen = pygame.display.set_mode((self.screenWidth, self.screenHeight), pygame.FULLSCREEN)
        else:
            self.screen = pygame.display.set_mode((self.screenWidth, self.screenHeight), pygame.RESIZABLE)
        self.w = self.screen.get_width()
        self.h = self.screen.get_height()
        self.clock = pygame.time.Clock()


    def draw_all(self, bDrawArrow=True, bDrawCursor=True):
        """
        Executes the drawing of all feedback components.
        """
        self.screen.blit(self.background, self.backgroundRect)
        self.screen.blit(self.border, self.borderRect)
        self.screen.blit(self.inner, self.innerRect)
        if(bDrawArrow == True):
            self.screen.blit(self.myarrow, self.myarrowRect)
            
        
        if(bDrawCursor == True):
            self.screen.blit(self.cursor, self.cursorRect)
            
        pygame.display.update()
        
    def draw_init(self):
        """
        Draws the initial screen.
        """
        self.screen.blit(self.background, self.backgroundRect)
        self.screen.blit(self.border, self.borderRect)
        self.screen.blit(self.inner, self.innerRect)
        
        
    def process_pygame_events(self):
        """
        Process the the pygame event queue and react on VIDEORESIZE.
        """
        for event in pygame.event.get():
            if event.type == pygame.VIDEORESIZE:
                self.resized = True
                self.size_old = self.size
                h = min(event.w, event.h)
                self.screen = pygame.display.set_mode((event.w, h), pygame.RESIZABLE) 
                self.init_graphics()
            elif event.type == pygame.QUIT:
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if(self.calibration == False):
                    step = 0
                    if event.unicode == u"a": step = -0.1
                    elif event.unicode == u"d" : step = 0.1
                    self.f += step
                    if self.f < -1: self.f = -1
                    if self.f > 1: self.f = 1

if __name__ == '__main__':
    ca = FeedbackCursorArrow3(None)
    
    #we want to send the following markers:
    # 1: MI first class
    # 2: MI second class               

    ca.trialGenerationPolicy = ca.AUTOMATIC_EQ_TRIALS_PER_CLASS_PER_BLOCK
    ca.classes =  ['left', 'right','foot']    
    ca.classesDirections =  ['left', 'right','foot']
    

    
    ca.on_init()
    ca.on_play()

