'''Feedbacks.VisualSpeller.ChormaSpellerVE


Created on May 12, 2012

@author: "Lovisa Irpa Helgadottir"

Requires VisionEgg and Pygame.
'''
from VisualSpellerVE import VisualSpellerVE, animate_sigmoid, animate
from VEShapes import FilledTriangle, FilledHexagon,FilledHourglass,FilledCross
from VisionEgg.MoreStimuli import FilledCircle, Target2D
from VisionEgg.Text import *
from lib.P300Layout.CircularLayout import CircularLayout
from lib.P300Layout.MatrixLayout import MatrixLayout
from VisionEgg.FlowControl import FunctionController
from VisionEgg.Textures import *

import Image, ImageDraw

import pygame ,os
from lib import marker 
#import pygame
import numpy as NP
	  
import logging
#from logging.handlers import FileHandler
from sys import platform, maxint
if platform == 'win32':
    import winsound

class ChromaSpellerVE(VisualSpellerVE):
	'''
	classdocs
	'''
	
	def init(self):
	    '''
	    initialize parameters
	    '''
	    
	    VisualSpellerVE.init(self)

            self.log_filename = 'ChromaSpeller.log'
	    self.letter_set = [['A','B','C','D','E'], \
                           ['F','G','H','I','J'], \
                           ['K','L','M','N','O'], \
                           ['P','Q','R','S','T'], \
                           ['U','V','W','X','Y'], \
                           ['Z','_','.',',','<']] 

            self.letter_wav=[['a.wav','b.wav','c.wav','d.wav','e.wav','before.wav'], \
                           ['f.wav','g.wav','h.wav','i.wav','j.wav','before.wav'], \
                           ['k.wav','l.wav','m.wav','n.wav','o.wav','before.wav'], \
                           ['p.wav','q.wav','r.wav','s.wav','t.wav','before.wav'], \
                           ['u.wav','v.wav','w.wav','x.wav','y.wav','before.wav'], \
                           ['z.wav','Minus.wav','point.wav','bag.wav','empty.wav','before.wav']] 
            

	    ## sizes:
	    self.letter_radius = 150             
	    self.speller_radius = 250
	    self.font_size_level1 = 250         # letters in level 1
	    self.font_size_level2 = 700         # letters in level 2
            self.letter_level1_layout_opt=['circular','line']
            self.letter_level1_layout=self.letter_level1_layout_opt[1]
	    self.feedbackbox_size = 200.0
	    self.font_size_feedback_ErrP = 300
	    
	    
	    self.stimulus_duration = 0.083   # 5 frames @60 Hz = 83ms flash
	    self.interstimulus_duration = 0.0


	    ## feedback type:
	    self.feedback_show_shape = True
	    
	    self.backdoor_symbol = "^"

	    
	    ## colors:
	    self.shape_color = (1.0, 1.0, 1.0)

	    self.stimuli_colors = [[0.0,0.0,1.0],
	                           [0.0,0.53,0.006],
	                           [1.0,0.0,0.0],
	                           [1.0,1.0,0.0],
	                           [0.86,0.0,0.86],
	                           [0.95,0.95,0.95]]

	    self.letter_color = [[1.0,1.0,0.0],
	                          [0.86,0.0,0.86],
	                           [0.95,0.95,0.95],
	                           [0.0,0.0,1.0],
	                           [0.0,0.53,0.006],
	                           [1.0,0.0,0.0]]

	    self.feedback_color = (0.9, 0.9, 0.9)
	    self.feedback_ErrP_color = (0.7, 0.1, 0.1)
	   
	   


            self.countdown_images=['ChromaSpeller_MatrixDisplay.png']
            self.countdown_image_duration=1 # sec
            self.countdown_wav=['1.wav','2.wav','3.wav','4.wav','5.wav']
            # If False only shapes are shown in that level but not group symbols(Letters)
            self.level_1_symbols=False
            self.level_2_symbols=True
            self.letter_color_on= True
            self.letterbox_on = False # Letter box shown during sequence
            #initaialize audio 
            self.init_pygame()


	def prepare_mainloop(self):
	    '''
	    called in pre_mainloop of superclass.
	    '''
            
	    ## init containers for VE elements:

	    self._ve_letters = []
            self._ve_letters_bg=[]
            self._ve_countdown_image =[]
	    
            self._letter_sound=[]
	   
	    self._countdown_screen = False
            self.countdown_sound=[]

	    assert len(self.stimuli_colors)==self._nr_elements
    
            #Load auditory stimuli   
            for i in xrange(5):
                self.loadAuditoryStimuli(self.countdown_wav[i])
                self.countdown_sound.append(self.sound)


            for i in xrange(self._nr_elements):
                tmpsnd=[]
                for j in xrange(self._nr_elements):
                    self.loadAuditoryStimuli(self.letter_wav[i][j])
                    tmpsnd.append(self.sound)
                    
                self._letter_sound.append(tmpsnd)            
	    


	def pre_play_tick(self):
            
            if (self._state_countdown):
                self. pre_countdown_screen()
                self.audio_countdown()
   
	
	def init_screen_elements(self):
            '''
	    overwrite this function in subclass.
	    '''
	    ## create shapes:
	    if (self.letter_level1_layout == self.letter_level1_layout_opt[0]):        
	        self._letter_layout = CircularLayout(nr_elements=self._nr_elements,radius=self.letter_radius, start=NP.pi/6.*5)
                self._letter_layout.positions.reverse()
            else:
                self._letter_layout = MatrixLayout(size=(900,0))
     
	    for i in xrange(self._nr_elements):		    
	        for j in xrange(len(self.letter_set[i])): # warning: self.letter_set must be at least of length self._nr_elements!!!
	            # add letter:
		    self._ve_letters.append(Text(position=(self._letter_layout.positions[j][0]+self._centerPos[0],
		                                               self._letter_layout.positions[j][1]+self._centerPos[1]),
			                                       text=self.letter_set[i][j],
			                                       font_size=self.font_size_level1,
			                                       color=self.letter_color[j],
			                                       anchor='center',
			                                       on=False))
		    self._ve_letters_bg.append(Text(position=(self._letter_layout.positions[j][0]+self._centerPos[0],
		                                           self._letter_layout.positions[j][1]+self._centerPos[1]),
			                                   text=self.letter_set[i][j],
			                                   font_size=self.font_size_level1+10,
			                                   color=(0.0,0.0,0.0),
			                                   anchor='center',
			                                   on=False))
			
			
	     # add letters of level 2:
	    for i in xrange(self._nr_elements):
                               
                self._ve_letters.append(Text(position=(self._letter_layout.positions[i][0],
		                                         self._letter_layout.positions[i][1]),
			                                 text=" ",
			                                 font_size=self.font_size_level2,
			                                 color=(self.stimuli_colors[i]),
			                                 anchor='center',
			                                 on=False))

		self._ve_letters_bg.append(Text(position=(self._letter_layout.positions[i][0],
			                                       self._letter_layout.positions[i][1]),
			                                 text=" ",
			                                 font_size=self.font_size_level2+70,
			                                 color=(0,0,0),
			                                 anchor='center',
			                                 on=False))

			
	    ## add feedback letters:
	    self._ve_feedback_letters = []
                        
	    for i in xrange(self._nr_elements):
		self._ve_feedback_letters.append(Text(position=(self._letter_layout.positions[i][0]+self._centerPos[0],
			                                                    self._letter_layout.positions[i][1]+self._centerPos[1]),
			                                          color=self.letter_color[i],
			                                          font_size=self.font_size_level1,
			                                          text=" ",
			                                          on=False,
			                                          anchor="center"))
	    self._ve_feedback_letters.append(Text(position=(self._letter_layout.positions[i][0]+self._centerPos[0],
			                                                    self._letter_layout.positions[i][1]+self._centerPos[1]),
			                                      color=self.letter_color[i],
			                                      font_size=self.font_size_level2,
			                                      text=" ",
			                                      anchor='center',
			                                      on=False))
			
			
			
	    ## add feedback note (whether or not there was an ErrP detected):
	    self._ve_feedback_ErrP = Text(position=(self._letter_layout.positions[i][0]+self._centerPos[0],
			                                                    self._letter_layout.positions[i][1]+self._centerPos[1]),
			                              color=self.feedback_ErrP_color,
			                              text="X",
			                              font_size=self.font_size_feedback_ErrP,
			                              anchor='center',
			                              on=False)

	

            # Put countdown images in container
	    path = os.path.dirname( globals()["__file__"] ) 	
            for i in xrange(len(self.countdown_images)):
                texture=TextureFromFile(os.path.join(path,self.countdown_images[i]))
                self._ve_countdown_image.append(TextureStimulus(texture=texture,position=(0,0),size=(self.geometry[2],self.geometry[3]),on=False))	
	    

            # put letters in container:
            self._ve_elements.extend(self._ve_letters_bg)
	    self._ve_elements.extend(self._ve_letters)
	    self._ve_elements.extend(self._ve_feedback_letters)
	    self._ve_elements.append(self._ve_feedback_ErrP)
            self._ve_elements.extend(self._ve_countdown_image)
	

        def pre_countdown_screen(self):
            if self._current_level==1:
                self._ve_countdown_image[0].set(on=True)
                self._presentation.set(go_duration=(self.countdown_image_duration, 'seconds'))
	        self._presentation.go()
                self._ve_countdown_image[0].set(on=False)
            else: # pre countdown screen for level 2
                self.wordbox(True)
                self._screen.set(bgcolor=self.bg_color)
                idx_start = self._classified_element*(self._nr_elements-1)
	        idx_end = idx_start + self._nr_elements-1
	        j=0
	        for i in xrange(idx_start, idx_end):
	            self._ve_letters_bg[i].set(on=True)
	            self._ve_letters[i].set(on= self.letter_color_on,color=self.stimuli_colors[j])
	            j=j+1
                self._presentation.set(go_duration=(self.countdown_image_duration, 'seconds'))
	        self._presentation.go()
	        for i in xrange(idx_start, idx_end):
	            self._ve_letters_bg[i].set(on=False)
	            self._ve_letters[i].set(on=False)
	        self._presentation.go()
          


	def set_countdown_screen(self):
	    '''
	    set screen how it should look during countdown.
	    '''
	    self._countdown_screen = True
            self._screen.set(bgcolor=self.bg_color)
            self._presentation.set(go_duration=(0.5, 'seconds'))
            self._presentation.go()

	
	def set_standard_screen(self):
	    '''
	    set screen elements to standard state.
	    '''

	    self._countdown_screen = False
	    
	    for i in xrange(self._nr_elements):
	        self._ve_letters[self._nr_letters + i].set(position=self._centerPos, on=False)
                self._ve_letters_bg[self._nr_letters + i].set(position=(self._centerPos[0], self._centerPos[1]), on=False)
	        

	




        def audio_countdown(self):
                 
            if self._current_countdown == self.nCountdown:
               self.send_parallel(marker.COUNTDOWN_START)
               self.logger.info("[TRIGGER] %d" % marker.COUNTDOWN_START)
               self.set_countdown_screen()   
        
            while(self._state_countdown):
               
                self.countdown_sound[self._current_countdown-1].play()
                while(pygame.mixer.get_busy()>0):
                     pygame.time.wait(1000) # 1 sec
         
                self._current_countdown = (self._current_countdown-1) % self.nCountdown
        

                if (self._current_countdown == 0):
                    self._current_countdown = self.nCountdown
                    self.set_standard_screen()
                    pygame.time.wait(10)
                    self._state_countdown = False
                    self._state_trial = True



        def init_pygame(self):
            '''
            Initialize pygame mixer
            '''
            pygame.mixer.quit()
            pygame.mixer.init(frequency=8000, size=-16, channels=1, buffer=256)
            sample_rate, bit_rate, channels = pygame.mixer.get_init()
            self.logger.info('sampling rate = ' + str(sample_rate) )
            self.audioChannel = pygame.mixer.Channel(1) 
          

	def stimulus(self, i_element, on=True):
	    '''
	    turn on/off the stimulus elements and turn off/on the normal elements.
	    '''
	    self.wordbox(self.letterbox_on)
	     
	    if self._current_level==1:
                self._screen.set(bgcolor=self.stimuli_colors[i_element])
	        for i in xrange(len(self.letter_set[i_element])):
	            self._ve_letters_bg[(self._nr_elements-1) * i_element + i].set(on=on and self.level_1_symbols)
                    self._ve_letters[(self._nr_elements-1) * i_element + i].set(on=on and self.letter_color_on and self.level_1_symbols)
	    else:
                self._screen.set(bgcolor=self.stimuli_colors[i_element])
	        if i_element < len(self.letter_set[self._classified_element]):
	            self._ve_letters_bg[self._nr_letters + i_element].set(on=on and self.level_2_symbols, 
                                                                          text=(on and self.letter_set[self._classified_element][i_element] or " "))
                    self._ve_letters[self._nr_letters + i_element].set(on=on and self.level_2_symbols and self.letter_color_on, 
                                                                       text=(on and self.letter_set[self._classified_element][i_element] or " "))
	        else:
	            self._ve_letters_bg[self._nr_letters + self._nr_elements-1].set(on=on and self.level_2_symbols, text=(on and self.backdoor_symbol or " "))
	            self._ve_letters[self._nr_letters + self._nr_elements-1].set(on=on and self.level_2_symbols and self.letter_color_on, text=(on and self.backdoor_symbol or " "))	
	
	def feedback(self):
	    '''
	    Show classified element / letter(s). 
	    '''
	    
	    self.wordbox(True)

	    
	    self._show_feedback(True)
	        
	    ## present:
	    self._presentation.set(go_duration=(self.feedback_duration, 'seconds'))
	    self._presentation.go()
            if (self._current_level ==2):
	        self._letter_sound[self._classified_element][self._classified_letter].play()
                pygame.time.wait(1000)      # 1 sec.
	    self._show_feedback(False)
	
	
	def _show_feedback(self, on=True):
    
	    if self._current_level == 1:
	      
	        if self.feedback_show_shape:
	            if not on:
	                pos = self._centerPos

	            ## turn on/off selected element:
	            self._screen.set(bgcolor=self.stimuli_colors[self._classified_element])
	            ## turn on/off letters of selected element:
	            idx_start = self._classified_element*(self._nr_elements-1)
	            idx_end = idx_start + self._nr_elements-1
	            for i in xrange(idx_start, idx_end):
	                self._ve_letters_bg[i].set(on=on)
	                self._ve_letters[i].set(on=on and self.letter_color_on)

	    else: ### level 2:
	        ## check if backdoor classified:
	        if self._classified_letter >= len(self.letter_set[self._classified_element]):
	            text = self.backdoor_symbol
	        else:
	            text = self.letter_set[self._classified_element][self._classified_letter]
	            
	        ## turn on/off letter:
	        if self.offline :
	            self._ve_feedback_letters[-1].set(on=on,
	                                              text=text,
	                                              color=(self.stimuli_colors[self._classified_letter] or
	                                                     self.letter_color[-1]))
	        if self.feedback_show_shape:
  
	            ## turn on/off current element:
                    self._screen.set(bgcolor=self.stimuli_colors[self._classified_letter])
	            
	            ## turn on/off letter of current element:
	            idx = self._nr_letters + self._classified_letter
	            self._ve_letters_bg[idx].set(on=on, text=text, position=(self._centerPos))
	            self._ve_letters[idx].set(on=on and self.letter_color_on, text=text, position=(self._centerPos))

	            
	    
	def switch_level(self):
	        
	    ## turn on written and desired words:
	    self.wordbox(True)

	    
	    if self.use_ErrP_detection and self._ErrP_classifier:
	        self._ve_feedback_ErrP.set(on=True)
	        self._show_feedback(True)
	        self._presentation.set(go_duration=(self.feedback_ErrP_duration, 'seconds'))
	        self._presentation.go()
	        self._ve_feedback_ErrP.set(on=False)
	        self._show_feedback(False)
	        return
	    
	    if self._current_level==1:
	        '''level 1: move classified letters to circles '''

	        self.set_standard_screen()
	       
	            
	    else:
	        ''' level 2: move classified letter to wordbox '''
	        
	        ## check if backdoor classified:
	        if self._classified_letter >= len(self.letter_set[self._classified_element]):
	            text = self.backdoor_symbol
	        else:
	            text = self.letter_set[self._classified_element][self._classified_letter]
	        
	        ## animate letter, but not if backdoor classified:
	        if self._classified_letter < len(self.letter_set[self._classified_element]):
	            def update(t):
	                dt = t/self.animation_time
	                self._viewport.parameters.stimuli = self._viewport.parameters.stimuli[:-1]
	                pos = animate_sigmoid(self._centerPos, self._current_letter_position, dt)
	                color = (list(animate_sigmoid(self.stimuli_colors[self._classified_letter], self.current_letter_color, dt)) or
	                         list(animate_sigmoid(self.letter_color[i], self.current_letter_color, dt)))
	                font_size = int(round(animate(self.font_size_level2, self.font_size_current_letter, dt)))	
	                self._viewport.parameters.stimuli.append(Text(position=pos,
	                                                              color=color,
	                                                              font_size=font_size,
	                                                              text=text,
	                                                              anchor='center'))
	            # send to screen:
	            self._viewport.parameters.stimuli.append(None)
	            self._presentation.set(go_duration=(self.animation_time, 'seconds'))
	            self._presentation.add_controller(None,None,FunctionController(during_go_func=update))
	            self._presentation.go()
	            self._presentation.remove_controller(None,None,None)
	            self._viewport.parameters.stimuli = self._viewport.parameters.stimuli[:-1]
	        else:
	            self._presentation.set(go_duration=(self.animation_time, 'seconds'))
	            self._presentation.go()
	            
	    ## turn off feedback box:
	    self._ve_feedback_box.set(on=False)
	


        def loadAuditoryStimuli(self, sounds):
            path = os.path.dirname( globals()["__file__"] ) 
            try:
                self.sound = pygame.mixer.Sound(os.path.join(path,"data/", sounds)) 
            except pygame.error, message:
               print 'Cannot load image:', sounds
               raise SystemExit, message

        def loadImage(self,images):
            path = os.path.dirname( globals()["__file__"] ) 
            try:
                self.image = pygame.image.load(os.path.join(path,"data/",images))
            except pygame.error, message:
                print 'Cannot load image:', image
                raise SystemExit, message


        def wordbox(self,state):
       	    self._ve_letterbox.set(on=state)
	    self._ve_spelled_phrase.set(on=state) 
	    self._ve_current_letter.set(on=state)
	    self._ve_desired_letters.set(on=state)
	    self._ve_innerbox.set(on=state)


	def pre__classify(self):
	    self.wordbox(True) # show word box during sequencing?


if __name__ == '__main__':
	fb = ChromaSpellerVE()
	fb.on_init()
	fb.on_play()
