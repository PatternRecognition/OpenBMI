

import pygame
import os, time
import random as r
import math
from string import atof
from sys import platform
if platform == 'win32':
    import winsound

from VisionEgg.Core import Screen
from VisionEgg.Core import Viewport
from VisionEgg.FlowControl import Presentation
from VisionEgg.MoreStimuli import FilledCircle
from VisionEgg.Textures import TextureStimulus, Texture
from VisionEgg.Text import Text

#from MainloopFeedback import MainloopFeedback 
from FeedbackBase.MainloopFeedback import MainloopFeedback
from lib.marker import *
from lib.eyetracker import EyeTracker


class Symmetry2(MainloopFeedback):
    
    # Trigger
    REPEAT = 21
    TARGET = 20
    NONTARGET = 19
    INVALID_FIXATION = 11
    #FIXATION_START = 18
    #TRIAL_START, TRIAL_END = 248,249    
    #RUN_START, RUN_END = 252, 253
    RESPONSE_1 , RESPONSE_2 = 16,17
    NUMBER_ADD = 100   # Added to the entered number (sent as trigger)
    
    def init(self):

        # Experimental design        
        self.nTrials = 4  	# number of trials
        self.nImgSym = 10
        self.nImgRan = 10
        self.p = 0.0;
        self.audio = True
        self.sequence = [1,2,1,2]
        self.word = 'THEY'

        # Timing
        self.tStim = 1000		# timing of stimulus (ms)
        self.tStimJitter = [300,500]	# Random jitter added to stimulus duration
        #self.tFixation = [300,500] 	# timing of fixation cross
        self.tRepeat = 1000
        self.tBegin = 5000              # Delay before begin
        self.tWord = 2000
        self.tDiode = 200

	# Stimuli
        #self.basedir = '/home/lena/Desktop/Symmetrie/stimuli' 	# Path to directory with stimuli 	
        self.basedir = 'D:\\stimuli\\labrotation10_Helene'
        self.targetPrefix = 'symA'		
        self.nontargetPrefix = 'ranA'
        self.Prefix = 'noise'
        self.postfix = '.jpg'

        # Screen Settings
        self.screenPos = [100, 100, 1024, 768] #1440, 900]
        self.fullscreen = True
        self.bgColor = (0., 0., 0.)
        
        # Fixation Point Settings
        self.fixpointSize = 4.0
        self.fixpointColor = (1.0, 0.0, 0.0)

        # Diode Point Settings
        self.diode = True
        self.diodeSize = 22.0
        self.diodeColor = (1.0, 1.0, 1.0)
        self.diodex = 40
        self.diodey = 40

        # Text Settings
        self.wordboxHeight = 60
        self.font_size_word = 65
        self.word_color = (0.2, 0.0, 1.0)

        # Countdown settings
        self.countdown_color = (0.2, 0.0, 1.0)
        self.nCountdown = 5
        self.font_size_countdown = 150
        
        # Eyetracker settings
        self.use_eyetracker = False
        self.et_range = 150         # Maximum acceptable distance between target and actual fixation
        self.et_range_time = 200    # maximum acceptable fixation off the designated point


    def pre_mainloop(self):
        self.send_parallel(RUN_START)        

        # Get sequence of pics
        self.seq = self.sequence*(self.nTrials/4)
        self.sym_pic = range(1, self.nImgSym+1) #range(0,self.nImgSym)
        self.ran_pic = range(1, self.nImgRan+1)

        #self.state_stim = True
        self.state_response = False
        self.state_trial = False
        self.state_countdown = True
        self.current_countdown = self.nCountdown
        #self.state_verify = False

        self.currentTrial = 0
        self.repeat = 0
        self.number = ''

        # initialize screen elements
        self.__init_screen()
        time.sleep(self.tBegin/1000.0)

        # Show Word
        self.ve_word.set(on=True)
        self.presentation.set(go_duration=(self.tWord/1000., 'seconds'))
        self.presentation.go()
        self.ve_word.set(on=False)

        ## Eyetracker
        if self.use_eyetracker:
            self.et = EyeTracker()
            self.et.start()
            cwd = os.path.dirname(__file__)
            self.soundfile = os.path.join(cwd,'winSpaceCritStop.wav')

        # Save the stimulus sequence
        self.lastSequence = []



    def __init_screen(self):
        if not self.fullscreen:
            os.environ['SDL_VIDEO_WINDOW_POS'] = '%d, %d' % (self.screenPos[0], self.screenPos[1])
       
        screenWidth, screenHeight = self.screenPos[2:]
        self._centerPos = [screenWidth/2., screenHeight/2.]
        # make screen:
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

        # make diode point:
        self.ve_diode = FilledCircle(radius=self.diodeSize,
                                        position=(self.diodex, self.diodey), 
                                        color=self.diodeColor,
                                        on=False)
        
        # make stimuli:  
        self.ve_pic = TextureStimulus(position=(screenWidth/2.-500,screenHeight/2.-500),max_alpha = 0.9, on=False)

        # make word
        self.ve_word = Text(position=(screenWidth/2., screenHeight/2.), #- self.wordboxHeight/2.),
                             text=self.word,
                             font_size=self.font_size_word,
                             color=self.word_color,
                             anchor='center',
                             on=False)

        # make Response
        self.ve_words = Text(position=(screenWidth/2., screenHeight/2.), #- self.wordboxHeight/2.),
                             text="How many did you count?",
                             font_size=self.font_size_word,
                             color=self.word_color,
                             anchor='center',
                             on=False)


        # add elements to viewport:
        self.viewport_fixpoint = Viewport(screen=self.screen, stimuli=[self.ve_fixpoint])
        self.viewport_diode = Viewport(screen=self.screen, stimuli=[self.ve_diode])
        self.viewport_pic = Viewport(screen=self.screen, stimuli=[self.ve_pic])
        self.viewport_select = Viewport(screen=self.screen, stimuli=[self.ve_words])
        self.viewport_countword = Viewport(screen=self.screen, stimuli=[self.ve_word])
        self.viewport_countdown = Viewport(screen=self.screen, stimuli=[self.ve_countdown])

        self.presentation = Presentation(viewports = [#self.viewport_fixpoint,
                                                      self.viewport_pic,
                                                      self.viewport_fixpoint,
                                                      self.viewport_diode,
                                                      self.viewport_select,
                                                      self.viewport_countword,
                                                      self.viewport_countdown,
                                                      ], 
                                                      handle_event_callbacks=[(pygame.KEYDOWN, self.keyboard_input)]
                                                      )


    def tick(self):
        pass            


    def play_tick(self):

        if self.state_countdown:
            self.countdown()
        elif self.state_trial:  
            self.trial()
            



    def post_mainloop(self):
       # pygame.quit()
        time.sleep(0.1)
        if self.use_eyetracker:
            self.et.stop()
        self.send_parallel(RUN_END)
        print "USE_EYETRACKER: ",self.use_eyetracker
        #print 'Repetitions = ',self.repeat

    def trial(self):
        #self.send_parallel(TRIAL_START)

        ## Blank
        #self.presentation.set(go_duration=(self.tBeforeTrial/1000., 'seconds'))
        #self.presentation.go()

        ## Fixationpoint
        #duration = r.randint(self.tFixation[0],self.tFixation[1])
        #self.send_parallel(FIXATION_START)
        #self.ve_fixpoint.set(on=True)
        #self.presentation.set(go_duration=(duration/1000., 'seconds'))
        #self.presentation.go()
        #self.ve_fixpoint.set(on=False) 

	## Stimuli
        jitter = r.randint(self.tStimJitter[0],self.tStimJitter[1])
	if len(self.lastSequence) >= len(self.sequence) and (self.p > r.random()): #(self.currentTrial != 0) and 
            trigger = self.REPEAT
            self.repeat += 1
            #self.pic = os.path.join(self.basedir, self.Prefix+self.postfix)
            time_s = self.tRepeat+jitter
            snum = self.seq[self.currentTrial]
            for i in range(self.currentTrial-1,0,-1):
                if self.seq[i] == snum:
                    self.pic = self.lastSequence[-(self.currentTrial-i)]
                    break
                
            self.lastSequence.append(self.pic)
            self.lastSequence = self.lastSequence[1:]   # Remove first element
            
        else:
            if self.seq[self.currentTrial] == 1:
                trigger = self.NONTARGET
                idx = r.randint(0,len(self.ran_pic)-1)
                self.pic = os.path.join(self.basedir, self.nontargetPrefix+str(self.ran_pic[idx])+self.postfix) 
                self.ran_pic.remove(self.ran_pic[idx])
            elif self.seq[self.currentTrial] == 2:
                trigger = self.TARGET  
                idx = r.randint(0,len(self.sym_pic)-1)
                self.pic = os.path.join(self.basedir, self.targetPrefix+str(self.sym_pic[idx])+self.postfix) 
                self.sym_pic.remove(self.sym_pic[idx])
            # Save stimulus sequence
            if len(self.lastSequence)<len(self.sequence):
                self.lastSequence.append(self.pic)
            else:
                self.lastSequence.append(self.pic)
                self.lastSequence = self.lastSequence[1:]   # Remove first element
                
            #print "TIME: ",time_s

        print "Lastsequence: ",self.lastSequence

        self.currentTrial += 1
        time_s = self.tStim + jitter

        self.texture_pic = Texture(self.pic)
        self.ve_pic.set(texture=self.texture_pic)  
        
        self.ve_pic.set(on=True)

        self.send_parallel(trigger)
        if self.diode:
            # Show picture + diode
            self.ve_diode.set(on=True)
            self.presentation.set(go_duration=(self.tDiode/1000., 'seconds'))
            self.presentation.go()
            
            if self.use_eyetracker and not self.__eyetracker_input():
                # Send invalid fixation trigger
                self.send_parallel(self.INVALID_FIXATION)
                self.logger.info("[TRIGGER] %d" % self.INVALID_FIXATION)

            # After diode offset, show only picture
            self.ve_diode.set(on=False)
            self.presentation.set(go_duration=( (time_s-self.tDiode)/1000.,'seconds'))
            self.presentation.go()
        else:    
            self.presentation.set(go_duration=(time_s/1000., 'seconds'))
            self.presentation.go()
        
        self.ve_pic.set(on=False)
        

        #self.send_parallel(TRIAL_END)

        
        if (self.currentTrial == self.nTrials):
            #print "AUDIO IS TRUE"
            if (self.p == 0.0) and self.audio:
                ## Response
                self.ve_fixpoint.set(on=False)
                self.ve_words.set(on=True)
                self.state_response = True
                self.presentation.set(quit = False)
                self.presentation.run_forever()
                self.ve_words.set(on=False)

                ## Verifying Response
                #self.ve_words.set(on=True) 
                #self.state_verify = True
                #self.presentation.set(quit = False)
                #self.presentation.run_forever()  
                #self.ve_words.set(on=False) 
            
            self.on_stop()
            time.sleep(0.1)

        
    def countdown(self):
        if self.current_countdown == self.nCountdown:
            self.send_parallel(COUNTDOWN_START)
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
            self.ve_fixpoint.set(on=True)
            #if self.diode:
            #    self.ve_diode.set(on=True)
            #self.send_parallel(COUNTDOWN_END)

    def __eyetracker_input(self):
        # Control eye tracker
        if self.et.x is None:
            self.logger.warning("[EYE_TRACKER] No eyetracker data received!")
        tx, ty = self._centerPos[0], self._centerPos[1]
        cx, cy = self.et.x, self.et.y
        if type(self.et.x)!=int or type(self.et.y)!=int:
            self.logger.error("no eyetracker input. stopping...")
            self.on_stop()
        dist = math.sqrt(math.pow(tx - cx, 2) + math.pow(ty - cy, 2))
        print "SCREENXY %d/%d; EYETRACKER XY: %d/%d, DIST: %d" % (self._centerPos[0],self._centerPos[1],self.et.x,self.et.y,dist)
        # Check if current fixation is outside the accepted range 
        if dist > self.et_range:
            # Check if the off-fixation is beyond temporal limits
            if self.et.duration > self.et_range_time:
                if platform == 'win32':
                    winsound.PlaySound(self.soundfile, winsound.SND_ASYNC)
                return False
        return True

    def keyboard_input(self, event):

        if self.state_response:
            if event.key == pygame.K_RETURN:
                self.number = int(atof(self.number))
                self.presentation.set(quit = True)
                self.state_response = False
                #print self.number
                self.send_parallel(self.NUMBER_ADD + self.number)
                #self.state_verify = False
            elif event.key == pygame.K_BACKSPACE:
                if len(self.number)>0:
                    self.number = self.number[0:-1]
            elif event.key in (pygame.K_0,pygame.K_1,pygame.K_2,pygame.K_3,pygame.K_4,pygame.K_5,pygame.K_6,pygame.K_7,pygame.K_8,pygame.K_9):
                self.number = self.number + chr(event.key)
                #print "KEY ", event.key, type(event.key)

            self.ve_words.set(text="How many did you count?: %s" % self.number)
        
        
        ## CHECK
        elif event.key == pygame.K_q or event.type == pygame.QUIT:
            if self.state_response or self.state_verify:
                self.presentation.set(quit = True)
            self.on_stop()


if __name__ == "__main__":
    feedback = Symmetry2()
    feedback.on_init()
    feedback.on_play()
 

