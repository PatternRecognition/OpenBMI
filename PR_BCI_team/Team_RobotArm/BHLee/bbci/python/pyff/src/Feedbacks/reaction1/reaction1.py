
import pygame
import sys
import random
import scipy
from FeedbackBase.MainloopFeedback import MainloopFeedback
import os

try:
     from pygame.locals import *
except ImportError:
    print("Error loading modules")  

#          for event in pygame.event.get():
#              if (event.type == pygame.KEYUP):
#                  _ = pygame.key.name(event.key)


class reaction1(MainloopFeedback):

    # TRIGGER VALUES FOR THE PARALLEL PORT (MARKERS)
    START_EXP, END_EXP = 252, 253
    STIMULUS_BEGIN = 1 #this is also end of ISI (inter-stimulus-interval)
    STIMULUS_END = 2 #this is also response-start
    RESPONSE_END = 3 # this is also ISI start
    RESPONSE_MARK = 4 # the response from the keyboard
    #MOTOR_BEGIN = 4 # beginning of self paced taps
    #MOTOR_END = 5 # end of self paced taps
    
     #initializing experimental parameters
    def init(self):
        """
        Initializes variables etc., but not pygame itself.
        """    
        self.num_blocks = 2
        self.block_trials = 50
        self.rest_time = 2* 60*1000
        #self.motor_task = 5*60*1000
        #rest_time = 2 * 60 * 1000 #2min in ms
        self.block_time = 10 * 60 * 1000 #10min in ms
        self.stim_minT = 30 #ms
        self.stim_maxT = 100 #ms
        self.responseT = 2 * 1000 #1s in ms
        self.isi_minT = 1 * 1000 #1s in ms
        self.isi_maxT = 10 * 1000 #3s in ms
        self.warnS = 10 * 1000 #warning signal--presentation of arrow
        #self.warnS = 10 * 60 * 10 * 1000 #warning signal--presentation of arrow
        self.x=370
        self.y=270
        self.width=60
        self.height=60
        self.thicknessR=0
        self.pointlistV=[(400,280),(400,320)]
        self.pointlistH=[(380,300),(420,300)]
        self.closed=False
        self.thicknessL=2
        self.grey = (60, 60, 60) #screen background
        self.white = (80,80,80) # white stimulus
        self.black = (79,79,79) # flicker black stimulus
        self.blackL = (0,0,0)


    def pre_mainloop(self):
        #self.logger.debug("on_play")
        self.init_pygame()
        #self.load_images()  # this is done in init_graphics
        #self.init_run()
    
    def init_pygame(self):
        """
        Set up pygame and the screen and the clock.
        """    
        #initializing pygame
        pygame.init()
        self.clock = pygame.time.Clock()
        self.fps = 120.1
        #os.environ['SDL_VIDEO_WINDOW_POS']="-800,1000"
        os.environ['SDL_VIDEO_WINDOW_POS']="0,0"
        self.screen=pygame.display.set_mode((800,600))
        #initializing the screen
        self.screen.fill(self.grey)
        self.send_parallel(reaction1.START_EXP)
        pygame.display.update()
        
    #def init_run(self):    
        #stimulus blocks
        for i in range(self.num_blocks):
            #self.stim_T = 1/120.1*1000*3*scipy.ones(150)  
            self.stim_T = 30+scipy.rand(self.block_trials)*(self.stim_maxT-self.stim_minT)
            self.isi_T = self.isi_minT + scipy.rand(self.block_trials)*(self.isi_maxT-self.isi_minT)
            print self.stim_T
            print self.isi_T
            #Display the arrow
            pygame.draw.lines(self.screen,self.blackL,self.closed,self.pointlistV,self.thicknessL)
            pygame.draw.lines(self.screen,self.blackL,self.closed,self.pointlistH,self.thicknessL)
            print 'start = ',self.clock.tick(self.fps)
            pygame.display.update()
            pygame.time.delay(self.warnS)
            print 'arrow = ', self.clock.tick(self.fps)
            
            
            #Display the white square initially
            pygame.draw.rect(self.screen,self.white,(self.x,self.y,self.width,self.height),self.thicknessR)
            pygame.display.update()
            
            for j in range(self.block_trials):
                #actual stimulus
                pygame.draw.rect(self.screen,self.black,(self.x,self.y,self.width,self.height),self.thicknessR)
                pygame.display.update()
                self.send_parallel(reaction1.STIMULUS_BEGIN)
                print 'white = ',self.clock.tick(self.fps)
                pygame.time.delay(int(self.stim_T[j])-10)
                
                #Inter-stimulus white square
                pygame.draw.rect(self.screen,self.white,(self.x,self.y,self.width,self.height),self.thicknessR)
                #print 'stimulus = ',clock.tick(fps)
                pygame.display.update()
                self.send_parallel(reaction1.STIMULUS_END)
                print 'stimulus = ',self.clock.tick(self.fps), 'delay =', int(self.stim_T[j])

                #for event in pygame.event.get():
                #        if (event.type == pygame.KEYUP):
                #            _ = pygame.key.name(event.key)
                #            self.send_parallel(reaction1.RESPONSE_MARK)
                #Response delay
                #t0 = pygame.time.get_ticks()
                #ta = 0
                #print 'response = ', t0
                #while(ta < self.responseT):
                #    for event in pygame.event.get():
                #        if (event.type == pygame.KEYUP):
                #            _ = pygame.key.name(event.key)
                #            self.send_parallel(reaction1.RESPONSE_MARK)
                #    ta = (pygame.time.get_ticks()-t0)
                pygame.time.delay(self.responseT)
                self.send_parallel(reaction1.RESPONSE_END)
                print 'response = ', self.clock.tick(self.fps)
        
                #Inter-trial delay
                pygame.time.delay(int(self.isi_T[j]))
        
            

##            # Motor Response Task
##            if i == 0:
##                #Display the arrow
##                
##                pygame.draw.lines(self.screen,self.blackL,self.closed,self.pointlistV,self.thicknessL)
##                pygame.draw.lines(self.screen,self.blackL,self.closed,self.pointlistH,self.thicknessL)
##                print ' motor start = ',self.clock.tick(self.fps)
##                pygame.display.update()
##                self.send_parallel(reaction1.MOTOR_BEGIN)
##                pygame.time.delay(self.motor_task)
##                print 'motor arrow end = ', self.clock.tick(self.fps)
##                
##              
##            if i == 2:
##                #Display the arrow
##                
##                pygame.draw.lines(self.screen,self.blackL,self.closed,self.pointlistV,self.thicknessL)
##                pygame.draw.lines(self.screen,self.blackL,self.closed,self.pointlistH,self.thicknessL)
##                print ' motor start = ',self.clock.tick(self.fps)
##                pygame.display.update()
##                self.send_parallel(reaction1.MOTOR_END)
##                pygame.time.delay(self.motor_task)
##                print 'motor arrow end = ', self.clock.tick(self.fps)
##                

            print 'trial end = ', self.clock.tick(self.fps)
            self.screen.fill(self.grey)
            pygame.display.update()
            pygame.time.delay(self.rest_time)
            print 'rest time = ', self.clock.tick(self.fps)
        self.send_parallel(reaction1.END_EXP)    
        pygame.quit()

if __name__ == '__main__':
    gk = reaction1()
    gk.on_init()
    gk.on_play()
