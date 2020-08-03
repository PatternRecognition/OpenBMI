

import pygame
import os, time
import numpy as np
import random as r

from VisionEgg.Core import Screen
from VisionEgg.Core import Viewport
from VisionEgg.FlowControl import Presentation
from VisionEgg.MoreStimuli import FilledCircle
from VisionEgg.Textures import * #Texture
from VisionEgg.Text import Text

#from MainloopFeedback import MainloopFeedback 


from FeedbackBase.MainloopFeedback import MainloopFeedback


class Symmetry3(MainloopFeedback):
    
    # Trigger
    TARGET = 20
    NONTARGET = 19
    FIXATION = 18
    TRIAL_START, TRIAL_END = 248,249    
    RUN_START, RUN_END = 252, 253
    RESPONSE_1 , RESPONSE_2 = 16,17
    
    def init(self):


        # Experimental design        
        self.nTrials = 8  	# number of trials
        self.nCond = 4
        self.nImgSym = 10
        self.nImgRan = 10

        # Timing
        self.tStim = 1000 		# timing of stimulus (ms)
        self.tStim_seq = 200		# timing of stimulus in sequence
        self.tFixation = 500 		# timing of fixation cross
        self.tBeforeTrial = 500 	# blank screen at begin of each trial (before first fix cross)
        #self.tBlankAfterFix = 500 	# blank screen between fixation cross and stimulus
        #self.tBlankAfterStim = 1000 	# blank screen after stimulus
        #self.tITI = 2000  		# intertrial interval (when selfpaced = 0) (ms)
        self.tBetweenStim = 100

	# Stimuli
        self.basedir = '/home/lena/Desktop/Symmetrie/stimuli' 	# Path to directory with stimuli 	
        self.targetPrefix = 'sym'		
        self.nontargetPrefix = 'ran'
        self.postfix = '.jpg'

        # Screen Settings
        self.screenPos = [100, 100, 1440, 900]
        self.fullscreen = False
        self.bgColor = (0., 0., 0.)
        
        # Fixation Point Settings
        self.fixpointSize = 4.0
        self.fixpointColor = (1.0, 1.0, 1.0)

        ###############
        self.wordboxHeight = 60
        self.font_size_word = 65
        self.word_color = (0.2, 0.0, 1.0)

    def pre_mainloop(self):
        self.send_parallel(self.RUN_START)        

        self.seq = range(self.nCond)*(self.nTrials/self.nCond)
        self.sym_pic = range(1, self.nImgSym+1) #range(0,self.nImgSym)
        self.ran_pic = range(1, self.nImgRan+1)

        #self.state_stim = True
        self.state_response = False
        self.state_verify = False

        # get random sequence of trials
        # self.seq = self.sequence()
        self.currentTrial = 0

        # initialize screen elements
        self.__init_screen()

        # Set target triggers
        #self.TARGET = range(20,20+len(self.orientations))


    def __init_screen(self):
        screenWidth, screenHeight = self.screenPos[2:]
        # make screen:
        self.screen = Screen(size=(screenWidth, screenHeight),
                             fullscreen=self.fullscreen,
                             bgcolor=self.bgColor,
                             sync_swap=True)
        
        # make fixation point:
        self.ve_fixpoint = FilledCircle(radius=self.fixpointSize,
                                        position=(screenWidth/2., screenHeight/2.), 
                                        color=self.fixpointColor,
                                        on=False)
        
        # make stimuli:  
          
        #self.ve_pic = TextureStimulus(position=(screenWidth/2.,screenHeight/2.), on=False)
        self.ve_pic = TextureStimulus(position=(0,0), on=False)
        #self.ve_ran = TextureStimulus(position=(screenWidth/2.,screenHeight/2.), on=False) #screenWidth, -screenHeight

        # make Response
        self.ve_words = Text(position=(screenWidth/2., screenHeight/2.), #- self.wordboxHeight/2.),
                             text="Druecken Sie <ENTER> ",
                             font_size=self.font_size_word,
                             color=self.word_color,
                             anchor='center',
                             on=False)        


        # add elements to viewport:
        self.viewport_fixpoint = Viewport(screen=self.screen, stimuli=[self.ve_fixpoint])
        #self.viewport_blank = Viewport(screen=self.screen)
        self.viewport_pic = Viewport(screen=self.screen, stimuli=[self.ve_pic])
        #self.viewport_ran = Viewport(screen=self.screen, stimuli=[self.ve_ran])
        self.viewport_select = Viewport(screen=self.screen, stimuli=[self.ve_words])

        self.presentation = Presentation(viewports = [self.viewport_fixpoint,
                                                      self.viewport_pic,
                                                      #self.viewport_ran,
                                                      #self.viewport_blank,
                                                      self.viewport_select
                                                      ], 
                                                      handle_event_callbacks=[(pygame.KEYDOWN, self.keyboard_input)]
                                                      )


    def tick(self):
        pass            


    def trial(self, Prefix, List, Time):
        idx = r.randint(0,len(List)-1)
        self.pic = os.path.join(self.basedir, Prefix+str(List[idx])+self.postfix)
        List.remove(List[idx])
            
        self.texture_pic = Texture(self.pic)
        self.ve_pic.set(texture=self.texture_pic)
        self.ve_pic.set(on=True)      
        self.presentation.set(go_duration=(Time/1000., 'seconds'))
        self.presentation.go()
        self.ve_pic.set(on=False)
  
        return List


    def play_tick(self):

        self.send_parallel(self.TRIAL_START)

        ## Blank
        self.presentation.set(go_duration=(self.tBeforeTrial/1000., 'seconds'))
        self.presentation.go()

        ## Fixationpoint
        self.send_parallel(self.FIXATION)
        self.ve_fixpoint.set(on=True)
        self.presentation.set(go_duration=(self.tFixation/1000., 'seconds'))
        self.presentation.go()
        self.ve_fixpoint.set(on=False)   

        ## Stimulus
        cT = self.seq[self.currentTrial]

        if cT == 0:
            self.sym_pic = self.trial(self.targetPrefix, self.sym_pic, self.tStim_seq) 
            self.presentation.set(go_duration=(self.tBetweenStim/1000., 'seconds'))
            self.presentation.go()

            self.sym_pic = self.trial(self.targetPrefix, self.sym_pic, self.tStim_seq) 
            self.presentation.set(go_duration=(self.tBetweenStim/1000., 'seconds'))
            self.presentation.go()

            self.sym_pic = self.trial(self.targetPrefix, self.sym_pic, self.tStim_seq) 
            self.presentation.set(go_duration=(self.tBetweenStim/1000., 'seconds'))
            self.presentation.go()

            self.sym_pic = self.trial(self.targetPrefix, self.sym_pic, self.tStim_seq) 

        elif cT == 1:
            self.sym_pic = self.trial(self.targetPrefix, self.sym_pic, self.tStim) 

        elif cT == 2:
            self.ran_pic = self.trial(self.nontargetPrefix, self.ran_pic, self.tStim_seq) 
            self.presentation.set(go_duration=(self.tBetweenStim/1000., 'seconds'))
            self.presentation.go()

            self.ran_pic = self.trial(self.nontargetPrefix, self.ran_pic, self.tStim_seq) 
            self.presentation.set(go_duration=(self.tBetweenStim/1000., 'seconds'))
            self.presentation.go()

            self.ran_pic = self.trial(self.nontargetPrefix, self.ran_pic, self.tStim_seq) 
            self.presentation.set(go_duration=(self.tBetweenStim/1000., 'seconds'))
            self.presentation.go()

            self.ran_pic = self.trial(self.nontargetPrefix, self.ran_pic, self.tStim_seq)

        elif cT == 3:
            self.ran_pic = self.trial(self.nontargetPrefix, self.ran_pic, self.tStim) 
    

        ## Response
        self.state_response = True
        self.presentation.set(quit = False)
        self.presentation.run_forever()

        ## Verifying Response
        self.ve_words.set(on=True) 
        self.state_verify = True
        self.presentation.set(quit = False)
        self.presentation.run_forever()  
        self.ve_words.set(on=False)     

        self.send_parallel(self.TRIAL_END)

        self.currentTrial += 1
        if self.currentTrial == self.nTrials:
            self.on_stop()
            time.sleep(0.1)


    def post_mainloop(self):
       # pygame.quit()
        time.sleep(0.1)
        self.send_parallel(self.RUN_END)



    def keyboard_input(self, event):

        if (self.state_response or self.state_verify) and (event.key == pygame.K_1 or event.key == pygame.K_2):
            if event.key == pygame.K_1: 
                self.send_parallel(self.RESPONSE_1)
            else:
                self.send_parallel(self.RESPONSE_2)

            self.presentation.set(quit = True)
            self.state_response = False
            self.state_verify = False
        
        elif self.state_verify and (event.key == pygame.K_RETURN):
            self.presentation.set(quit = True)
            self.state_verify = False
        
        ## CHECK
        elif event.key == pygame.K_q or event.type == pygame.QUIT:
            if self.state_response or self.state_verify:
                self.presentation.set(quit = True)
            self.on_stop()


if __name__ == "__main__":
    feedback = Symmetry3()
    feedback.on_init()
    feedback.on_play()
 












