#!/usr/bin/env python
#
# Plays Videos. Needs VisionEgg and wxpython installed.
#
# Copyright (C) 2010  Helene Schmidt, Matthias Treder
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

import pygame
import os, time
import numpy as np
import random as r

import wx
import wx.media

from VisionEgg.Core import Viewport, Screen, SimplePerspectiveProjection
from VisionEgg.FlowControl import Presentation
from VisionEgg.MoreStimuli import FilledCircle
from VisionEgg.Textures import Texture, TextureStimulus3D 
from VisionEgg.Text import Text

from FeedbackBase.MainloopFeedback import MainloopFeedback
from lib.ExperimentalDesign.OrthogonalDesign import orthogonalDesign


class Video(MainloopFeedback):
    
    # Trigger 
    RUN_START, RUN_END = 252, 253
    TRIAL_START, TRIAL_END = 248,249 
    FIXATION = 18
    RESPONSE_1 , RESPONSE_2 = 16,17
    COUNTDOWN_START = 240

    def init(self):
        
        # Experimental design
        #self.nConditions = 2		# number of conditions
        self.nBlocks = 4		# number of blocks
        self.ranSym = (0,1)
        self.nImg = 1		        # Total number of images in each subdir

        # Timing (milliseconds)
        #self.tStim = 500 		# timing of stimulus
        self.tFixation = 500		# timing of fixation cross
        self.tBeforeTrial = 500 	# blank screen at begin of each trial (before first fix cross)
        self.tBlankAfterFix = 500 	# blank screen between fixation cross and stimulus
        self.tBlankAfterStim = 500	# blank screen after stimulus

        # Stimuli
        self.basedir = '/home/lena/Desktop/Perspektive_Movie' # Path to directory with stimuli
        self.postfix = '.avi'

        # Subjects and logfile
        self.subject = 'JohnDoe'        # Name of subject
        self.logdir = 'D:\stimuli\gert\logfiles' # Path to logfile directory
        self.logging = 1                # Switch logging on or off
        
        # Screen Settings
        self.screenPos = [100, 100, 1440, 900] #[100, 100, 1000, 700]
        self.bgColor = (0., 0., 0.)
        self.fullscreen = False
        
        # Fixation Point Settings
        self.fixpointSize = 4.0
        self.fixpointColor = (1.0, 1.0, 1.0)

        # Text Settings
        self.wordboxHeight = 60
        self.font_size_word = 65
        self.word_color = (0.2, 0.0, 1.0)
        self.pressButtonText = "Druk 1 of 2"

        # Countdown settings
        self.countdown_color = (0.2, 0.0, 1.0)
        self.nCountdown = 5
        self.font_size_countdown = 150

    def pre_mainloop(self):
        self.send_parallel(self.RUN_START)
      
        self.state_response = False
        self.state_trial = False
        self.state_pause = False
        self.state_countdown = True
        self.current_countdown = self.nCountdown
        self.currentTrial = 0
        self.currentBlock = 1
        self.key = None     # Pressed key is saved here
        
        self.nTrials = 2*4 
        #self.nTrials = len(self.ranSym)*len(self.angles)*self.nImg
        # get random sequence of stimuli
        #self.seq = np.random.permutation(self.nImg)
        #self.seq = orthogonalDesign([self.ranSym,self.angles,range(1,self.nImg+1)],self.nTrials)
        #self.seq = np.random.permutation(self.seq)
        #print "sequence ",self.seq
        
        # Logging
        #if self.logging:
        #    logfile = os.path.join(self.logdir,"%s.txt" % self.subject)
        #    try: 
        #        self.logfile = open(logfile, 'a')
        #        text = []
        #        text.append("# [SymmetryPerspective] Logfile for %s" % self.subject)
        #        text.append("# The following variables are logged in the data columns")
        #        text.append("# Column 1: number of trial")
        #        text.append("# Column 2: random (0) or symmetry (1)")
        #        text.append("# Column 3: slant angle")
        #        text.append("# Column 4: number of picture")
        #        text.append("# Column 5: subject response")
        #        self.logfile.write("\n".join(text))
        #    except IOError:
        #        print "Cannot open datafile ",logfile
        #        self.on_quit()


        # initialize screen elements
        self.__init_screen()


    def __init_screen(self):
        
        # make screen:
        screenWidth, screenHeight = self.screenPos[2:]
        self.screen = Screen(size=(screenWidth, screenHeight),
                             fullscreen=self.fullscreen,
                             bgcolor=self.bgColor,
                             sync_swap=True)

        # make countdown:
        self.ve_countdown = Text(position=(screenWidth/2., screenHeight/2.), 
                                 text=" ",
                                 font_size=self.font_size_countdown,
                                 color=self.countdown_color,
                                 anchor='center',
                                 on=False)

        # make fixation point:
        self.ve_fixpoint = FilledCircle(radius=self.fixpointSize,
                                        position=(screenWidth/2., screenHeight/2.), 
                                        color=self.fixpointColor,
                                        on=False)


        # make response
        self.ve_words = Text(position=(screenWidth/2., screenHeight/2.),
                             text=self.pressButtonText,
                             font_size=self.font_size_word,
                             color=self.word_color,
                             anchor='center',
                             on=False)

        # make break between blocks
        self.ve_pause = Text(position=(screenWidth/2., screenHeight/2.),
                             text="Pause. Weiter mit <ENTER>",
                             font_size=self.font_size_word,
                             color=self.word_color,
                             anchor='center',
                             on=False)

        # add elements to viewport:
        self.viewport_fixpoint = Viewport(screen=self.screen, stimuli=[self.ve_fixpoint])
        self.viewport_blank = Viewport(screen=self.screen)  
        self.viewport_response = Viewport(screen=self.screen, stimuli=[self.ve_words])
        self.viewport_countdown = Viewport(screen=self.screen, stimuli=[self.ve_countdown])
        self.viewport_pause = Viewport(screen=self.screen, stimuli=[self.ve_pause])
       

        self.presentation = Presentation(viewports = [self.viewport_fixpoint,
                                                      self.viewport_blank,
                                                      self.viewport_response,
                                                      self.viewport_countdown,
                                                      self.viewport_pause],
                                                      handle_event_callbacks=[(pygame.KEYDOWN, self.keyboard_input)])


    def tick(self):
        pass            

    def play_tick(self):
        if self.state_countdown:
            self.countdown()
        elif self.state_trial:  
            self.trial()            
        elif self.state_pause:
            self.pause()
            
    def post_mainloop(self):
       # pygame.quit()
        time.sleep(0.1)
        self.send_parallel(self.RUN_END)
        #if self.logging:
        #    try: self.logfile.close()
        #    except IOError: self.logger.warn("Could not close datafile")
        self.presentation.set(quit=True)
        self.screen.close()


    def countdown(self):
        if self.current_countdown == self.nCountdown:
            self.send_parallel(self.COUNTDOWN_START)
            self.ve_countdown.set(on=True)
            self.presentation.set(go_duration=(1., 'seconds'))

        self.ve_countdown.set(text="%d" % self.current_countdown)
        self.presentation.go()
        self.current_countdown = (self.current_countdown-1) % self.nCountdown
        if self.current_countdown == 0:
            self.current_countdown = self.nCountdown
            self.ve_countdown.set(on=False, text=" ")
            #pygame.time.wait(20)
            self.state_countdown = False
            self.state_trial = True

    def close_window(self, event):
        self.frame.Destroy()

    def trial(self):
        self.send_parallel(self.TRIAL_START)

        ## Blank before trial
        self.presentation.set(go_duration=(self.tBeforeTrial/1000., 'seconds'))
        self.presentation.go()

        ## Fixationpoint
        self.send_parallel(self.FIXATION)
        self.ve_fixpoint.set(on=True)
        self.presentation.set(go_duration=(self.tFixation/1000., 'seconds'))
        self.presentation.go()
        self.ve_fixpoint.set(on=False)        
        
        ## Blank after fixation
        self.presentation.set(go_duration=(self.tBlankAfterFix/1000., 'seconds'))
        self.presentation.go()

        ## Stimulus - Play Video

        # load stimuli
        #cond = self.seq[self.currentTrial]
        #ranSym = cond[0]
        #angle = cond[1]
        #pic_nr = cond[2]
        #print self.seq[self.currentTrial]
        #angle = r.choice(self.angles)
        #pic_nr = self.seq[self.currentTrial]

        self.path = os.path.join(self.basedir,'phy'+self.postfix)

        app = wx.PySimpleApp()
        self.frame = wx.Frame(None, -1,)
        #self.frame.CenterOnScreen()
        #self.frame.Maximize()

        mc = wx.media.MediaCtrl(self.frame)
        mc.Load(self.path)
        mc.Play()
        self.frame.ShowFullScreen(True)

        wx.media.EVT_MEDIA_STOP(self.frame,-1,self.close_window)
        app.MainLoop()


        ## Blank after stimulus
        self.presentation.set(go_duration=(self.tBlankAfterStim/1000., 'seconds'))
        self.presentation.go()

        ## Response
        self.ve_words.set(on=True) 
        self.state_response = True
        self.presentation.set(quit = False)
        self.presentation.run_forever()
        self.ve_words.set(on=False) 

        #if self.logging:
        #    self.log_data()
        #time.sleep(0.2)

        self.send_parallel(self.TRIAL_END)

        self.currentTrial += 1  
        if self.currentTrial == self.nTrials:
            self.on_stop()
            time.sleep(0.1)

        if self.currentTrial == self.currentBlock*(self.nTrials/self.nBlocks):
            self.state_trial = False
            self.state_pause = True

    def pause(self):
        self.currentBlock += 1
        self.ve_pause.set(on=True) 
        self.presentation.set(quit = False)
        self.presentation.run_forever()
        self.ve_pause.set(on=False)

    def keyboard_input(self, event):

        if (self.state_response) and (event.key == pygame.K_1 or event.key == pygame.K_2):
            if event.key == pygame.K_1:
                self.send_parallel(self.RESPONSE_1)
            else:
                self.send_parallel(self.RESPONSE_2)
            self.key = event.key
            self.presentation.set(quit = True)
            self.state_response = False
        
        elif (self.state_pause) and (event.key == pygame.K_RETURN):
            self.presentation.set(quit = True)
            self.state_pause = False
            self.state_countdown = True

        ## CHECK
        elif event.key == pygame.K_q or event.type == pygame.QUIT:
            if self.state_response:
                self.presentation.set(quit = True)
            self.on_stop()
            
    def log_data(self):
        ct = self.currentTrial
        cs = self.seq[self.currentTrial]
        text = "\n%d\t%d\t%d\t%d\t%s" % (ct+1,cs[0],cs[1],cs[2],self.key)
        print text
        self.logfile.write(text)

        

if __name__ == "__main__":
    feedback = Video()
    feedback.on_init()
    feedback.on_play()

