

import pygame
import os, time
import numpy as np
import random as r

from VisionEgg.Core import Screen
from VisionEgg.Core import Viewport
from VisionEgg.FlowControl import Presentation
#from VisionEgg.MoreStimuli import FilledCircle
from VisionEgg.Textures import * #Texture
#from VisionEgg.Text import Text

#from MainloopFeedback import MainloopFeedback 


from FeedbackBase.MainloopFeedback import MainloopFeedback


class Symmetry1(MainloopFeedback):
    
    # Trigger
    TARGET = 20
    NONTARGET = 19
    FIXATION = 18
    TRIAL_START, TRIAL_END = 248,249    
    RUN_START, RUN_END = 252, 253
    RESPONSE = 16
    
    def init(self):


        # Experimental design        
        self.nTrials = 6   	# number of trials
        self.nImgSym = 4 
        self.nImgRan = 6
        self.p = 0.2		# probability of symmertic pic

        # Timing
        self.tStim = 100 		# timing of stimulus (ms)
        #self.tFixation = 500 		# timing of fixation cross
        #self.tBeforeTrial = 500 	# blank screen at begin of each trial (before first fix cross)
        #self.tBlankAfterFix = 500 	# blank screen between fixation cross and stimulus
        #self.tBlankAfterStim = 1000 	# blank screen after stimulus
        #self.tITI = 2000  		# intertrial interval (when selfpaced = 0) (ms)
        self.tBetweenStim = 500

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
        #self.fixpointSize = 4.0
        #self.fixpointColor = (1.0, 1.0, 1.0)

        ###############
        #self.wordboxHeight = 60
        #self.font_size_word = 65
        #self.word_color = (0.2, 0.0, 1.0)

    def pre_mainloop(self):
        self.send_parallel(self.RUN_START)        

        self.sym_pic = range(1, self.nImgSym+1) #range(0,self.nImgSym)
        self.ran_pic = range(1, self.nImgRan+1)
        #self.state_stim = True
        #self.state_response = False
        #self.state_verify = False

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
        #self.ve_fixpoint = FilledCircle(radius=self.fixpointSize,
        #                                position=(screenWidth/2., screenHeight/2.), 
        #                                color=self.fixpointColor,
        #                                on=False)
        
        # make stimuli:  
          
        #self.ve_pic = TextureStimulus(position=(screenWidth/2.,screenHeight/2.), on=False)
        self.ve_pic = TextureStimulus(position=(0,0), on=False)
        #self.ve_ran = TextureStimulus(position=(screenWidth/2.,screenHeight/2.), on=False) #screenWidth, -screenHeight

        # make Response
        #self.ve_words = Text(position=(screenWidth/2., screenHeight/2.), #- self.wordboxHeight/2.),
        #                     text="Druecken Sie <ENTER> ",
        #                     font_size=self.font_size_word,
        #                     color=self.word_color,
        #                     anchor='center',
        #                     on=False)        


        # add elements to viewport:
        #self.viewport_fixpoint = Viewport(screen=self.screen, stimuli=[self.ve_fixpoint])
        #self.viewport_blank = Viewport(screen=self.screen)
        self.viewport_pic = Viewport(screen=self.screen, stimuli=[self.ve_pic])
        #self.viewport_ran = Viewport(screen=self.screen, stimuli=[self.ve_ran])
        #self.viewport_select = Viewport(screen=self.screen, stimuli=[self.ve_words])

        self.presentation = Presentation(viewports = [#self.viewport_fixpoint,
                                                      self.viewport_pic,
                                                      #self.viewport_ran,
                                                      #self.viewport_blank,
                                                      #self.viewport_select
                                                      ], 
                                                      handle_event_callbacks=[(pygame.KEYDOWN, self.keyboard_input)]
                                                      )


    def tick(self):
        pass            

    def play_tick(self):
        
        if self.p > r.random():
            trigger = self.TARGET  
            idx = r.randint(0,len(self.sym_pic)-1)
            self.pic = os.path.join(self.basedir, self.targetPrefix+str(self.sym_pic[idx])+self.postfix) 
            self.sym_pic.remove(self.sym_pic[idx])
        else:
            trigger = self.NONTARGET
            idx = r.randint(0,len(self.ran_pic)-1)
            self.pic = os.path.join(self.basedir, self.nontargetPrefix+str(self.ran_pic[idx])+self.postfix) 
            self.ran_pic.remove(self.ran_pic[idx])
        

        self.texture_pic = Texture(self.pic)
        self.ve_pic.set(texture=self.texture_pic) 

        self.presentation.set(go_duration=(self.tBetweenStim/1000., 'seconds'))
        self.presentation.go()

        self.send_parallel(trigger)  
        self.ve_pic.set(on=True)      
        self.presentation.set(go_duration=(self.tStim/1000., 'seconds'))
        self.presentation.go()
        self.ve_pic.set(on=False)

        ## Response
        #self.state_response = True
        #self.presentation.set(quit = False)
        #self.presentation.run_forever()

        ## Verifying Response
        #self.ve_words.set(on=True) 
        #self.state_verify = True
        #self.presentation.set(quit = False)
        #self.presentation.run_forever()  
        #self.ve_words.set(on=False)     

        #self.send_parallel(self.TRIAL_END)

        self.currentTrial += 1
        if self.currentTrial == self.nTrials:
            self.on_stop()
            time.sleep(0.1)


    def post_mainloop(self):
       # pygame.quit()
        time.sleep(0.1)
        self.send_parallel(self.RUN_END)



    def keyboard_input(self, event):
        
        if event.key == pygame.K_q or event.type == pygame.QUIT:
            self.presentation.set(quit = True)
            self.on_stop()
        else:
            self.send_parallel(self.RESPONSE)


if __name__ == "__main__":
    feedback = Symmetry1()
    feedback.on_init()
    feedback.on_play()
 












