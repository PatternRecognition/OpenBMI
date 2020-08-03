'''
    Set up code for relax season. Available methods include:
    1. Initializes filenames based on the language input
    2. Reads the sound files (.wav) from acquisition/data/sounds
    3. Initializes the screen with the display message 
    4. Display Crosses
    5. Display a Countdown
    
'''
import sys,os
import pygame
from pygame.locals import *
<<<<<<< .mine
import PygameFeedback
import MainloopFeedback
=======
from FeedbackBase.PygameFeedback import PygameFeedback
>>>>>>> .r23813
import functions as myFunc
from glyph import Glyph, Macros
import audiere
import wave

class Setup(PygameFeedback):
    """ Sets up the environment for required EEG measurement """
        
    def init(self):
        """ Initialization """
        
 
    def makeFname(self):
        """ Creates the sound filename variables for auditory stimulus """
        if self.lang == 'german':                        
            self.auFiles =  ['stopp',            # 01                           
                         'anspannen',                      # 02
                         'links',                          # 03
                         'rechts',                         # 04
                         'fuss',                           # 05
                         'augen',                          # 06
                         'mitte',                          # 07
                         'links',                          # 08
                         'rechts',                         # 09
                         'oben',                           # 10
                         'unten',                          # 11
                         'blinzeln',                       # 12
                         'augen_fest_zu_druecken',         # 13
                         'augen_zu',                       # 14
                         'augen_auf',                      # 15
                         'schlucken',                      # 16
                         'zunge_gegen_gaumen_druecken',    # 17
                         'schultern_heben',                # 18
                         'zaehne_knirschen',               # 19
                         'vorbei',                         # 20
                         'entspannen' ]                        # 21]
        elif self.lang == 'english':
            self.auFiles = ['stop',                       # 01
                         'maximum_compression',         # 02
                         'left',                       # 03
                         'right',                      # 04
                         'foot',                       # 05
                         'look',                       # 06
                         'center',                     # 07
                         'left',                       # 08
                         'right',                      # 09
                         'up',                         # 10
                         'down',                       # 11
                         'blink',                      # 12
                         'press_your_eyelids_shut',    # 13
                         'eyes_closed',                # 14
                         'eyes_open',                  # 15
                         'swallow',                    # 16
                         'press_tongue_to_the_roof_of_your_mouth',
                         'lift_shoulders',             # 18
                         'clench_teeth',              # 19
                         'over',                       # 20
                         'relax']                       #21
        else :
            sys.exit('Sorry, english and german/deutsch are the only available \
                    languages')
        
    
    """ Executable section """
    def playSound(self, sound, multi=True):
        """ Plays the .wav file given by sound """
        self.logger.debug('Now Playing sound')
        self.send_parallel(self.SOUND_ON)
        if self.aud:
            d = audiere.open_device()
            l = d.open_file(sound+'.wav')
            l.play()
            while l.playing:
                pass        
        else :
            if not pygame.mixer.get_init():#44100, -16, 2, 2048
                f = wave.open(sound+'.wav', 'r')
                sampR = f.getframerate()
                sizeN = 16
                chan = f.getnchannels()
                pygame.mixer.init(sampR, sizeN, chan, self.buffer)
            snd = pygame.mixer.Sound(sound)
            #clock = pygame.time.Clock()
            snd.play()
            while pygame.mixer.get_busy():
                pass
            
        #winsound.PlaySound('%s.wav' % sound, winsound.SND_FILENAME)
        self.logger.debug('Sound Stopped')
        self.send_parallel(self.SOUND_OFF)
        if multi:
            pygame.mixer.quit()
        
    def readText(self):
        """ Reads text from a given file"""
        f = open(self.TEXT_DIR,'r')
        if self.lineByline:
            self.text = f.readlines()
        else :
            self.text = f.read()

    def pauseFunct(self, dur):
        """ holds the program until a fixed pause duration elapses
        pygame.time.delay causes freezing thereby flickering of win
        between transitions from pause to animation """
        self.logger.debug('Pause Begin')
        self.send_parallel(self.PAUSE_ON)
        t = 0
        m = pygame.time.Clock()
        t = t + m.tick_busy_loop()        
        while t < dur:
            for event in pygame.event.get():
                if event.type == QUIT:
                    self.quit_pygame()
                    
            t = t + m.tick_busy_loop()
        self.logger.debug('Pause Stop')
        self.send_parallel(self.PAUSE_OFF)
    
    def setFontSize(self,text,margin,tabLen):
        """ Sets the Font sizes """
        
        # or render one letter and check the geometry
        fontScale = (self.dim[1])/(len(text))
        font = pygame.font.Font(os.path.join(self.resrceDir, "FreeSans.ttf"), fontScale)
        lines=([len(text[i]) for i in range(len(text))])
        index,value=myFunc.find(lines,myFunc.equals,max(lines))
        lineSize = font.get_height()
        maxX=font.size(text[index])[0]
        maxY=font.size(text[index])[1]
        while (maxX > (margin[2]-margin[0])) or (maxY*len(text) > (margin[3]-margin[1]))  :
            fontScale = fontScale - 1
            font = pygame.font.Font(os.path.join(self.resrceDir, "FreeSans.ttf"), fontScale)
            lineSize = font.get_height()
            maxX=font.size(text[index])[0]
            maxY=font.size(text[index])[1]
            
        return fontScale
        
    def setFontSizeGlyph(self):
        """ auto sets font size for passing to Glyph """

        buff = self.gl._buff
        lineSize=max([buff[i].get_height() for i in range(len(buff))])
        """ improve to include the line spacing too"""
        totlines = len(self.gl._buff)
        txtSize = lineSize*totlines
        while txtSize > self.txtDim[3]-self.txtDim[1]+self.fineTune:
            fontsize = self.fontscale - 1
            self.init_glyph(fontsize)
            self.glyph_rect = self.gl.rect
            self.gl.input(self.text,justify = 'justified')
            totlines = len(self.gl._buff)
            
            buff = self.gl._buff
            lineSize=max([buff[i].get_height() for i in range(len(buff))])
            
            totlines = len(self.gl._buff)
            txtSize = lineSize*totlines
            
    def dispText(self, disp = True,intvl=0):
        """ To display the output of readText"""
        self.logger.debug('Instructions Start')
        self.send_parallel(self.INSTR_ON)
        if self.lineByline:
            self.fontScale = self.setFontSize(self.marginDim, self.tabLen)
            font = pygame.font.Font(None, self.fontScale)
            lineSize = font.get_height()
            for i in range(len(self.text)):
                
                txt = font.render(self.text[i].rstrip('\n'), 1, (10, 10, 10))
                textpos = txt.get_rect()
                textpos.centery = self.background.get_rect().centery-lineSize*len(self.text)/2
                textpos.move(0,i*lineSize)
                self.background.blit(txt, textpos.move(self.tabLen,i*lineSize))
                                
                # Blit everything to the screen
                self.screen.blit(self.background, (0, 0))
                pygame.display.flip()
        else :
            self.glyph_rect = self.gl.rect
            self.gl.input(self.text,justify = 'justified')

            self.setFontSizeGlyph()
            
            self.gl.update()
            self.screen.blit(self.background, (0, 0))
            self.screen.blit(self.gl.image, self.gl.rect)
            pygame.display.flip()

        self.clock.tick(intvl)
        #self.wait_for_pygame_event()
        self.process_pygame_events(disp)    
        self.logger.debug('Instructions End')
        self.send_parallel(self.INSTR_OFF)           
                
    def process_pygame_events(self,disp=None):
        """
        Process the pygame event queue.
        """          
        n = 1  
        while disp or n:
            for event in pygame.event.get():
                if event.type == QUIT:
                    self.background.fill(self.grey)
                    self.screen.blit(self.background, (0, 0))
                    pygame.display.flip()
                    self.quit_pygame()
                if event.type == pygame.KEYUP:
                    _ = pygame.key.name(event.key)
                    if (_ == 'return'):
                        self.background.fill(self.grey)
                        self.screen.blit(self.background, (0, 0))
                        pygame.display.flip()
                        return
            n = n-1
        pygame.display.flip()
            
    def init_glyph(self,fontsize):
        
        self.margin = self.dim[0]/40        
        self.tabLen = self.margin*3
        self.marginDim = [self.margin,self.margin,self.dim[0]-30-self.margin,self.dim[1]-30-self.margin]
        self.txtDim = [self.margin+60,self.margin+60,self.dim[0]-60-self.margin-60-30,self.dim[1]-60-self.margin-60-30]
        pygame.draw.rect(self.background,(0,0,0),(self.marginDim),5)
        
        self.fontscale = fontsize
        self.FONT = pygame.font.Font(os.path.join(self.resrceDir, "FreeSans.ttf"), self.fontscale)
        self.DEFAULT = {
            'bkg'       : self.grey,
            'color'     : self.dgrey,
            'font'      : self.FONT,
            'spacing'   : self.spacing, #FONT.get_linesize(),
            }
#        margin = 30
        self.rect = pygame.Rect(self.txtDim)
        self.gl = Glyph(self.rect, **self.DEFAULT)

        Macros['b'] = ('font', pygame.font.Font(os.path.join(self.resrceDir, "freesansbold.ttf"), self.fontscale))
        Macros['big'] = ('font', pygame.font.Font(os.path.join(self.resrceDir, "silkscreen.ttf"), self.fontscale+20))
        Macros['BIG'] = ('font', pygame.font.Font(os.path.join(self.resrceDir, "silkscreen_bold.ttf"), self.fontscale+20))
        Macros['red'] = ('color', (255, 0, 0))
        Macros['green'] = ('color', (0, 255, 0))
        Macros['bkg_blu'] = ('bkg', (0, 0, 255))
    
    def fixationCross(self,multi=None,rect=None,disp=True,intvl=0):
        """ Display one/multi(5) fixation crosses"""
        self.logger.debug('Fixation Cross Start')
        self.send_parallel(self.FIXATION_ON)
        offset = (min(self.dim)/15)
        xC = self.dim[0]/2
        yC = self.dim[1]/2
        pygame.draw.line(self.background, self.black, [xC, yC-offset], [xC, yC+offset], min(self.dim)/60)
        pygame.draw.line(self.background, self.black, [xC-offset, yC], [xC+offset, yC], min(self.dim)/60)
        if multi:
            xCi = [xC-self.dim[1]/4, self.dim[0]/2, xC+self.dim[1]/4, self.dim[0]/2] 
            yCi = [self.dim[1]/2, self.dim[1]/4, self.dim[1]/2, 3*self.dim[1]/4]
            if rect:
                xCi = [self.dim[0]/4, self.dim[0]/2, 3*self.dim[0]/4, self.dim[0]/2] 
                yCi = [self.dim[1]/2, self.dim[1]/4, self.dim[1]/2, 3*self.dim[1]/4]
            for i in range(4):
                pygame.draw.line(self.background, self.black, [xCi[i], yCi[i]-offset/10*8], [xCi[i], yCi[i]+offset/10*8], min(self.dim)/80)
                pygame.draw.line(self.background, self.black, [xCi[i]-offset/10*8, yCi[i]], [xCi[i]+offset/10*8, yCi[i]], min(self.dim)/80)
            
        self.screen.blit(self.background, (0, 0))
        pygame.display.flip()
        # use pygame.wait instead ???
        self.clock.tick(intvl)
        self.process_pygame_events(disp)
        self.logger.debug('Fixation Cross Period Over')
        self.send_parallel(self.FIXATION_OFF)
                
    def countDown(self,intvl=10,disp=True):
        """ To implement a countdown Screen """
        self.logger.debug('Countdown Start')
        self.send_parallel(self.COUNTDOWN_START)
        font = pygame.font.Font(None, min(self.dim)/5)
        for i in range(intvl+1):
            self.process_pygame_events()
            self.clock.tick(1)
            self.background.fill(self.grey)
            txt = font.render(str(intvl-i), 1, (10, 10, 10))
            txtpos = txt.get_rect()
            txtpos.centerx = self.background.get_rect().centerx
            txtpos.centery = self.background.get_rect().centery
            self.background.blit(txt,txtpos)
            self.screen.blit(self.background, (0, 0))
            pygame.display.flip()
            
        self.clock.tick(1)
        self.background.fill(self.grey)
        pygame.display.flip()
        self.process_pygame_events(disp)
        self.logger.debug('Countdown finished')
        self.send_parallel(self.COUNTDOWN_STOP)
                

    def decodeSeq(self):
        """ Makes the stimulus sequence from the coded seq """
        S = "".join(i for i in self.seq.split(' '))
        ls = []
        for i in range(len(S)):
            try :
                int(S[i])
            except :
                ls.append(i)
        m=[]
        for i in range(len(ls)):
            m.append(S[ls[i]])
            t=None
            try :
                t=S[ls[i]+1:ls[i+1]]
            except :
                t=S[ls[i]+1:len(S)]
            if t.isalnum():
                if t.isdigit():
                    t=int(t)
                m.append(t)
        todo = self.parse(m, 0)
        todoN = todo[0]

        return todoN
        
    def parse(self,seq,rek):
        """ Decodes the self.seq string 
        
        DECODING THE 'seq' STRING SEQUENCE:
             1.P stands for index 3 (For Pausing)
                 followed by duration (numerical time) in ms
             2.F stands for index 1 (see switch case in stim_artifactMeasurement)
                 (For Playing Sound)
                 followed by another index which refers to which sound to play (see cel)
             3.A stands for 4 (for Animation)
                 followed by duration of animation in ms
             4.R is for repeat and is
                 followed by number of times of the repeat
                 followed by () indicating the sequence to repeat R times
             
         INDICES: 
                 1 : play with asyn
                 2 : play with syn ??
                 3 : pause
                 4 : animation for given time
                 5 : play the stimutil_countdown(functionality??) for given time
        """
        
        a=[]
        if not seq:
            if rek > 0:
                raise('parsing error')
            a = []
            rest = ''
            return a, rest
        
        if isinstance(seq[0],int):
            raise('parsing error')
            a = []
            rest = ''
            return a, rest
        
        if seq[0] == 'F':
            [aN, rest] = self.parse(seq[2:len(seq)], rek)
            a.extend([1,seq[1]])
            if aN:
                a.extend(aN)
            return a, rest
        elif seq[0] == 'f':
            [aN, rest] = self.parse(seq[2:len(seq)], rek)
            a.extend([2,seq[1]])
            if aN:
                a.extend(aN)
            return a, rest
        elif seq[0] == 'P':
            [aN, rest] = self.parse(seq[2:len(seq)], rek)
            a.extend([3,seq[1]])
            if aN:
                a.extend(aN)
            return a, rest
        elif seq[0] == 'A':
            [aN, rest] = self.parse(seq[2:len(seq)], rek)
            a.extend([4,seq[1]])
            if aN:
                a.extend(aN)
            return a, rest
        elif seq[0] == 'C':
            [aN, rest] = self.parse(seq[2:len(seq)], rek)
            a.extend([5,seq[1]])
            if aN:
                a.extend(aN)
            return a, rest
        elif seq[0] == ')':
            if rek == 0:
                raise('parsing error')
                a = []
                rest = ''
                return a, rest
            a = None
            rest = seq[1:len(seq)]
            return a, rest
        elif seq[0] == 'R':
            if (not seq[1] == '[') or (not isinstance(seq[2],int)) or (not seq[3] == ']') or (not seq[4] == '('):
                raise('parsing error')
                a = []
                rest = ''
                return a, rest
            [aN, rest] = self.parse(seq[5:len(seq)], rek+1)
            b = []
            for _ in range(seq[2]):
                b.extend(aN)
            a.extend(b)
            [aN,rest] = self.parse(rest, rek)
            a.extend(aN)
            if not rest=='':
                a.extend([rest])
            return a, rest
        else:
            raise('parsing error')
            a = []
            rest = ''
            return a, rest
    
    def lat2glyph(self):
        """ To convert bold/italic from latex to Glyph text rendering formats """
        target = ['\\bf{','\\rm']
        missiles = ['{b;','']
        self.text = myFunc.misc(self.text, target, missiles)
        self.text = myFunc.spaces(self.text)

