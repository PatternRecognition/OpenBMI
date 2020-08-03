

import pygame
import os, time
import numpy as np

from VisionEgg.Core import Screen
from VisionEgg.Core import Viewport
from VisionEgg.FlowControl import Presentation
from VisionEgg.MoreStimuli import FilledCircle
from VisionEgg.Textures import * #Texture
from VisionEgg.Text import Text

from MainloopFeedback import MainloopFeedback 
from orthogonalDesign import orthogonalDesign

#from FeedbackBase.MainloopFeedback import MainloopFeedback
#from lib.ExperimentalDesign.OrthogonalDesign import orthogonalDesign

class TwoIFC(MainloopFeedback):
    
    # Trigger
    TARGET = range(20,38)
    NONTARGET = 19
    FIXATION = 18
    TRIAL_START, TRIAL_END = 248,249    
    RUN_START, RUN_END = 252, 253
    RESPONSE_1 , RESPONSE_2 = 16,17
    
    def init(self):
        # Experimental design
        #self.nConditions = 18	# number of conditions
        self.orientations = range(4) #range(18)
        self.targetIval = range(2)
        #self.selfpaced = 1 	# or 0 (tITI)
        self.nTrials = 72   	# number of trials
        self.nImgSym = 6 
        self.nImgRan = 11

        # Timing
        self.tStim = 250 		# timing of stimulus
        self.tFixation = 500 		# timing of fixation cross
        self.tBeforeTrial = 500 	# blank screen at begin of each trial (before first fix cross)
        self.tBlankAfterFix = 500 	# blank screen between fixation cross and stimulus
        self.tBlankAfterStim = 1000 	# blank screen after stimulus
        #self.tITI = 2000  		# intertrial interval (when selfpaced = 0) (ms)
        #self.tBetweenStim = 1000

	# Stimuli
        self.basedir = '/home/lena/Desktop/stimuli' 	# Path to directory with stimuli
        #self.subdirs = ['10' '20' ... '180']  	
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


        #self.state_stim = True
        self.state_response = False
        self.state_verify = False

        # get random sequence of trials
        self.seq = self.sequence()
        self.currentTrial = 0

        # initialize screen elements
        self.__init_screen()

        # Set target triggers
        self.TARGET = range(20,20+len(self.orientations))

    def sequence(self):
        # Specify design
        self.trials = orthogonalDesign([self.orientations,self.targetIval],self.nTrials)   
        self.trials = np.array(self.trials)
        r = np.random.permutation(self.nTrials)
        
        # random order
        self.order_sym = []
        for i in self.orientations:
            self.order_sym.append(np.random.permutation(self.nImgSym))

        self.order_ran = np.random.permutation(self.nImgRan)        

        return self.trials[r]

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
          
        self.ve_sym = TextureStimulus(position=(screenWidth/2.,screenHeight/2.), on=False)
        self.ve_ran = TextureStimulus(position=(screenWidth/2.,screenHeight/2.), on=False) #screenWidth, -screenHeight

        # make Response
        self.ve_words = Text(position=(screenWidth/2., screenHeight/2.), #- self.wordboxHeight/2.),
                             text="Druecken Sie <ENTER> ",
                             font_size=self.font_size_word,
                             color=self.word_color,
                             anchor='center',
                             on=False)        


        # add elements to viewport:
        self.viewport_fixpoint = Viewport(screen=self.screen, stimuli=[self.ve_fixpoint])
        self.viewport_blank = Viewport(screen=self.screen)
        self.viewport_sym = Viewport(screen=self.screen, stimuli=[self.ve_sym])
        self.viewport_ran = Viewport(screen=self.screen, stimuli=[self.ve_ran])
        self.viewport_select = Viewport(screen=self.screen, stimuli=[self.ve_words])

        self.presentation = Presentation(viewports = [self.viewport_fixpoint,
                                                      self.viewport_sym,
                                                      self.viewport_ran,
                                                      self.viewport_blank,
                                                      self.viewport_select], 
                                                      handle_event_callbacks=[(pygame.KEYDOWN, self.keyboard_input)])


    def tick(self):
        pass            

    def play_tick(self):
        
        # Get current orientation and target ival
        orientation,targetIval = self.seq[self.currentTrial]

        # load stimuli
        nr = self.order_sym[orientation][-1]
        self.order_sym[orientation] = self.order_sym[orientation][:-1]
        self.sym = os.path.join(self.basedir, str(orientation), self.targetPrefix+str(nr)+self.postfix)
        self.texture_sym = Texture(self.sym)
        self.ve_sym.set(texture=self.texture_sym)        

        nr = self.order_ran[-1]
        self.order_ran = self.order_ran[:-1]
        self.ran = os.path.join(self.basedir, self.nontargetPrefix, self.nontargetPrefix+str(nr)+self.postfix)
        self.texture_ran = Texture(self.ran)
        self.ve_ran.set(texture=self.texture_ran)        
   

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
        
        ## Blank
        self.presentation.set(go_duration=(self.tBlankAfterFix/1000., 'seconds'))
        self.presentation.go()

        ## Stimulus 1
        if targetIval:
            self.first = self.ve_sym
            self.second = self.ve_ran
            self.send_parallel(self.TARGET[orientation])
        else:
            self.first = self.ve_ran
            self.second = self.ve_sym 
            self.send_parallel(self.NONTARGET)

        self.first.set(on=True)      
        self.presentation.set(go_duration=(self.tStim/1000., 'seconds'))
        self.presentation.go()
        self.first.set(on=False)

        ## Blank
        self.presentation.set(go_duration=(self.tBlankAfterStim/1000., 'seconds'))
        self.presentation.go()

        ## Fixationpoint
        self.send_parallel(self.FIXATION)
        self.ve_fixpoint.set(on=True)
        self.presentation.set(go_duration=(self.tFixation/1000., 'seconds'))
        self.presentation.go()
        self.ve_fixpoint.set(on=False)
    
        ## Blank
        self.presentation.set(go_duration= (self.tBlankAfterFix/1000., 'seconds'))
        self.presentation.go()

        ## Stimulus 2
        if targetIval:
            self.send_parallel(self.NONTARGET)
        else:
            self.send_parallel(self.TARGET[orientation])

        self.second.set(on=True)        
        self.presentation.set(go_duration=(self.tStim/1000., 'seconds'))
        self.presentation.go()
        self.second.set(on=False)

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
    feedback = TwoIFC()
    feedback.on_init()
    feedback.on_play()

