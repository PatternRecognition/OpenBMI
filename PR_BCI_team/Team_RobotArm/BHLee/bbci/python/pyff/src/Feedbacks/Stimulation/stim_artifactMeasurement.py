'''
Python version of stim_artifactMeasurement svn/bbci/acquisition/stimulation

'''
import os
import random
import numpy as np
import pygame
from pygame.locals import *
import string
from setup_relax import Setup

class stim_artifactMeasurement(Setup):
    
    INIT_FEEDBACK = 100
    SOUND_ON = 101
    SOUND_OFF = 102
    PAUSE_ON = 103
    PAUSE_OFF = 104
    INSTR_ON = 105
    INSTR_OFF = 106
    FIXATION_ON = 107
    FIXATION_OFF = 108
    COUNTDOWN_START = 109
    COUNTDOWN_STOP = 110
    EYES_OPEN = 111 # not currently used, consider usage
    EYES_CLOSED = 112 # not currently used, consider usage
    ANIM_ON = 113
    ANIM_OFF = 114
    GAME_OVER = 200
    
    def on_init(self):
        self._running = False
        self._paused = False
        self._inMainloop = False        
        self.logger.debug('on_init and Feedback Initiated')
        self.send_parallel(self.INIT_FEEDBACK)
        self.init_ToBeSetBeforeExec()
        #self.Triggers()
        self.init()        
            
    def init_ToBeSetBeforeExec(self):
        """ Default Variables to be defined/changed by experimenter """
        TOOLBOX_DIR = 'C:/svn/bbci/'
        self.BCI_DIR = TOOLBOX_DIR
        self.Lang = 'german'
        self.fname = 'season10_relax'
        self.seq = None #example: 'P2000 F21P2000 R[10] (F14P15000 F1P2000 F15A15000 F1P2000) F20P1000'
        self.dim = [1600,900]
        self.winPos = [0, 20]
        self.prefontSize = 60
        self.spacing = -7 # increase or decrease in steps of 1
        self.fineTune = 200 # increase or decrease in steps of 10 in order to adjust text to screen
        self.cap = 'Baseline Measurement'
        self.disp = None
        self.fullscreen = None
        self.lineByline = None
        self.frames = 50 # control speed to polygon animation
        self.FPS = self.frames 
        self.colorStep = np.pi/8 # color step from each frame of polygon animation
        self.aud = True # to play using pyaudiere
        self.countdown = False
        self.count = 10 # countdown no in seconds
        
    def init_audio(self):
        """ Initialize audio properties for pygame.mixer """
        self.sampR = None
        self.size = None
        self.channels = None
        self.buffer = 4096

    def init(self):
        """ Initializations """
        self.lang = string.lower(self.Lang)
        if self.lang == 'deutsch':
            self.lang = 'german'        
        self.SOUND_DIR = os.path.join(self.BCI_DIR,'acquisition','data','sound')
        self.SPEECH_DIR = os.path.join(self.SOUND_DIR,self.lang)
        self.TEXT_DIR = os.path.join(self.BCI_DIR,'acquisition','data','task_descriptions',self.fname+'.txt')        
#        self.resrceDir = os.path.join("C:/Python26/Lib/site-packages/pygame/fonts")
        self.resrceDir = os.path.join(os.path.dirname(__file__),'resources','font')      
        if self.seq == None:
            self.seq = 'P2000 F21P2000 R[10] (F14P15000 F1P2000 F15A15000 F1P2000) F20P1000'            
        self.init_pygame()
        self.init_graphics()
        self.init_glyph(self.prefontSize)
        self.init_audio()
        
    def Triggers(self):
        """ define Triggers"""
        pass        
    
    def pre_mainloop(self):
        """ pre mainloop """
        self.runSeq = {1: lambda x: self.playSound(os.path.join(self.SPEECH_DIR,'speech_'+self.auFiles[x-1])), \
                       2: lambda x: 'Method '+str(x)+': Sync Method Under Construction', \
                       3: lambda x: self.pauseFunct(x), \
                       4: lambda x: self.animation(dur=x) }
        #3: pygame.time.delay(x)
        self.makeFname()
        self.todo = self.decodeSeq()
        self.readText()
        self.lat2glyph()   
        if self.countdown:
            self.countDown(self.count, disp=None)     
        self.dispText()
        self.fixationCross(multi=None,disp=None) #makes crosses (single/multi), look into setup_relax
        
    def play_tick(self):
        """
        Called repeatedly in the mainloop if the Feedback is not paused.
        """
        for i in range(len(self.todo)/2):#
            for event in pygame.event.get():
                    if event.type == QUIT:
                        self.background.fill(self.grey)
                        self.screen.blit(self.background, (0, 0))
                        pygame.display.flip()
                        return
            
            print 'Step '+str(i+1)+' of '+str(len(self.todo)/2) 
            print self.todo[2*i]
            print self.todo[2*i+1]
            self.runSeq[self.todo[2*i]](self.todo[2*i+1])
        
        self._running = False       
        
    def post_mainloop(self):
        """Called after leaving the mainloop, e.g. after stop or quit."""
        self.background.fill(self.grey)
        self.screen.blit(self.background, (0, 0))
        pygame.display.flip()
        self.fixationCross()        
        self.logger.debug("on_quit")
        self.send_parallel(self.GAME_OVER)
        self.quit_pygame()
        
    def tick(self):
        """Process pygame events and advance time for 1/FPS seconds."""
        self.process_pygame_events()
        #self.elapsed = self.clock.tick(self.FPS)
     
    def init_pygame(self):
        """ Pygame initialization """
        self.step = 0

        pygame.init()        
        self.clock = pygame.time.Clock()
        
        self.screenWidth = self.dim[0]
        self.screenHeight = self.dim[1]
        self.black = [  0,  0,  0]
        self.dgrey = [12, 12, 12]
        self.white = [255,255,255]
        self.blue =  [  0,  0,255]
        self.green = [  0,255,  0]
        self.red =   [255,  0,  0]
        self.grey = [128, 128, 128]
        
        os.environ['SDL_VIDEO_WINDOW_POS'] = str(self.winPos[0])+","+str(self.winPos[1])
        if self.fullscreen:
            self.screen = pygame.display.set_mode((self.screenWidth, \
                                                   self.screenHeight), \
                                                   pygame.FULLSCREEN)
        else:
            self.screen = pygame.display.set_mode((self.screenWidth, \
                                                   self.screenHeight),\
                                                    pygame.RESIZABLE)
        pygame.display.set_caption(self.cap)  
        
        # Fill background
        self.background = pygame.Surface(self.screen.get_size())
        self.background = self.background.convert()
        self.background.fill(self.grey)   
        
    def init_graphics(self):
        """ Initializes animation graphics for eyes close and open condition """
        patchSize = min(self.dim)/3
        self.centerx = self.background.get_rect().centerx
        self.centery = self.background.get_rect().centery        
        self.x = range(self.centerx-patchSize/2,self.centerx+patchSize/2+1,1)
        self.y = range(self.centery-patchSize/2-10,self.centery+patchSize/2+1-10,1)
        self.posScreen = self.makePos() 
        self.posEnd = self.makePos()
        
    
    """ Main executable methods """    
    def animation(self, dur, disp=None):
        """ Creates the animation for eyes close and open condition """
        self.logger.debug('Start of Animation')
        self.send_parallel(self.ANIM_ON)
        self.updateColor()
        self.background.fill(self.grey)
        pygame.draw.polygon(self.background, self.color, self.posScreen, 0)
        pygame.draw.polygon(self.background, self.black, self.posScreen, 2)
        
        self.screen.blit(self.background, (0, 0))
        pygame.display.flip()
        totT = 0
        self.ms = pygame.time.Clock()
        
        while totT < dur:
            self.clock.tick(self.frames)
            
            self.updateColor()
            self.move()
            self.background.fill(self.grey)
            pygame.draw.polygon(self.background, self.color, self.posScreen, 0)
            pygame.draw.polygon(self.background, self.black, self.posScreen, 2)
            self.screen.blit(self.background, (0, 0))
            pygame.display.flip()            
            self.process_pygame_events()           
            totT = totT + self.ms.tick_busy_loop()
        
        self.process_pygame_events(disp)
        self.logger.debug('End of Animation')
        self.send_parallel(self.ANIM_OFF) 
        self.background.fill(self.grey)       
        self.screen.blit(self.background, (0, 0))
        pygame.display.flip()
        
    def move(self):
        """ To be put into stim_artifactMeasurement 
        Updates the position based on given and final positions and
        required frame rate """
        if (self.posScreen == self.posEnd).all():
            self.posEnd = self.makePos()
            self.dist = self.posEnd-self.posScreen
        
        self.dist = self.posEnd-self.posScreen
        self.posScreen = self.posScreen + (self.dist/np.abs(self.dist))       
            
    def makePos(self):
        """ stim_arti 
            Returns new position """
        tempx = random.sample(self.x,4)
        tempy = random.sample(self.y,4)
        position = np.array(([tempx[0],tempy[0]], \
                            [tempx[1],tempy[1]], \
                            [tempx[2],tempy[2]], \
                            [tempx[3],tempy[3]]))
        return position
    
    def updateColor(self):
        """ Creates smooth transitioning colour map """
        frames = self.frames
        rad =  self.colorStep * self.step/frames
        self.color = np.array([ 255*np.sin(rad) , 255*np.cos(rad) , 255*-1*np.cos(rad)])
        self.color = self.color.clip(min=0)
        self.color = self.color.round()
        self.color = 255 - self.color.round()
        self.step = self.step + 1       
        
if __name__ == '__main__':
    _ = stim_artifactMeasurement()
    _.on_init()
    _.on_play()
    