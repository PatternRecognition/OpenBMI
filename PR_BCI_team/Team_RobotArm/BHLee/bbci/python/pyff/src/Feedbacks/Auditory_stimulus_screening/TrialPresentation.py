# -*- coding: utf-8 -*-
import time, pygame, os, sys, logging, ocempgui
from numpy import arange, shape, array
from ocempgui.widgets import Renderer, ScrolledList, Entry
from ocempgui.widgets.Constants import *


class TrialPresentation: 

    def __init__(self, screenPos, sounds, templateTexts):
        self.logger = logging.getLogger("FeedbackController")        
        
        self.init_pygame(screenPos)
        self.init_graphics()
        self.loadAuditoryStimuli(sounds)
        self.draw_initial()
        self.templateTexts = templateTexts
        pass

    
    def init_pygame(self, screenPos):
        """
        Set up pygame and the screen and the clock.
        """

        # Due to some sound-problems on windows-platform, we have to initialize 
        # the components of pygame seperately!
        #pygame.init()

        ## SOUND
        #init sound:
        pygame.mixer.quit()
        #pygame.mixer.init(frequency=15000, size=-16, channels=2, buffer=512)
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512) # negative bit-size for signed values
        
        # The reduced buffer of 1024bits (512) is very important. With the standard 
        # value the latency is varying dramatically

        sample_rate, bit_rate, channels = pygame.mixer.get_init()
        self.logger.info('sampling rate = ' + str(sample_rate) )

        #self.audioChannel = pygame.mixer.find_channel() ## TODO
        self.audioChannel = pygame.mixer.Channel(1) 
    
        ## GRAPHICS
        # graphics parameters
        self.fullscreen = False
        self.visualStimuliEnabled = False
        self.screenPos = screenPos;
        self.screenWidth = self.screenPos[2]
        self.screenHeight = self.screenPos[3]
        
        # init graphics-component
        os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (self.screenPos[0],self.screenPos[1])        
        pygame.display.init()
        pygame.font.init()
        #pygame.time.init()
        if self.fullscreen:
            self.screen = pygame.display.set_mode((self.screenPos[2], self.screenPos[3]), pygame.FULLSCREEN)
        else:
            self.screen = pygame.display.set_mode((self.screenPos[2], self.screenPos[3]), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        # visual domain has to be initialized also for non-visual stimuli
        # --> Instructions, gazepoint etc.
        pygame.display.set_caption('PASS2D - Predictive Auditory Spatial ERP Speller with 2D stimuli')
        self.clock = pygame.time.Clock()
        
        self.renderer = Renderer()
        scrollSurface = pygame.Surface((self.screenWidth, int(self.screenHeight / 5)))
        self.renderer.screen = scrollSurface
        
        self.logger.info("pygame was initialized")
        
        
    def loadAuditoryStimuli(self, sounds):
        
        path = os.path.dirname( globals()["__file__"] ) 
     #dummy
        sound0 = pygame.mixer.Sound(os.path.join(path, sounds[0])) 

     ##high pitch
        sound1 = pygame.mixer.Sound(os.path.join(path, sounds[0])) 
        sound2 = pygame.mixer.Sound(os.path.join(path, sounds[1])) 
        sound3 = pygame.mixer.Sound(os.path.join(path, sounds[2])) 
     #medium pitch
        sound4 = pygame.mixer.Sound(os.path.join(path, sounds[3])) 
        sound5 = pygame.mixer.Sound(os.path.join(path, sounds[4])) 
        sound6 = pygame.mixer.Sound(os.path.join(path, sounds[5])) 
        
     #low pitch
        sound7 = pygame.mixer.Sound(os.path.join(path, sounds[6]))#kick_l.wav")) 
        sound8 = pygame.mixer.Sound(os.path.join(path, sounds[7]))#kick_m.wav")) 
        sound9 = pygame.mixer.Sound(os.path.join(path, sounds[8])) 

        self.soundlist = [sound0, sound1, sound2, sound3, sound4, sound5, sound6, sound7, sound8, sound9]


    def init_graphics(self):
        """
        Initialize the surfaces and fonts depending on the screen size.
        """
        # some standard parameters
        self.backgroundColor = (0,0,0)
        self.stdColor = (100, 100, 100)
        self.presentationColor = (0,255,0)
        self.deviantColor = (255,0,0)
        self.renderer.color = 100, 100, 100
        
        #self.stdAlphabetLegend = ["DUMP",u'\u2190', "it","ti","ta","at","ta","to","ot","to"]
        self.stdAlphabetLegend = ["DUMP","ti","it","ti",u"tä",u"ät",u"tä","to","ot","to"]
        #self.stdAlphabetLegend = ["DUMP",".\n-","it","ti",u"tä",u"ät",r"-\n.","to",u"-\n.","-\n."]
        self.alphabetLegend = self.stdAlphabetLegend
        
        self.screen = pygame.display.get_surface()
        self.size = min(self.screen.get_height(), self.screen.get_width())
        self.fontSizeBig = int(self.size/14)
        self.fontSizeSmall = int(self.size/18)
        self.background = pygame.Surface((self.screenWidth, self.screenHeight))
        self.background = self.background.convert()
        self.backgroundRect = self.background.get_rect(center=self.screen.get_rect().center)
        self.background.fill(self.backgroundColor)
        
        # self.headerPos = [(0.5*self.screenWidth,0.15*self.screenHeight/4)] #header!!
        self.headerPos = [(0.5*self.screenWidth,0.15*self.screenHeight/4)] #header!!
        self.footer1Pos = [(0.3*self.screenWidth,0.84*self.screenHeight)] #header!!
        self.footer2Pos = (0.6*self.screenWidth,0.82*self.screenHeight) #header!!

        self.centerButton = (int(0.5*self.screenWidth), int(0.5*self.screenHeight))
        
#        if self.visualStimulus or self.spellerMode: #spellerMode OR visualStim
        self.buttonWidth = int(self.size*0.15)
        self.buttonHeight = int(self.size*0.15)
        buttonSize = (self.buttonWidth, self.buttonHeight)
        self.buttonList = ['N/A']  #first element -> [0] is Dump
        self.buttenRectList = ['N/A']
        self.rectSizes = (0,0,self.buttonWidth,self.buttonWidth)
  
     
        for i in arange(1,10):
            button = pygame.Surface(buttonSize)
            tmp = i%3 #indicator in which column we are
            #find corresponding horizontal position!
            if tmp==1:
                hpos = 1*self.screenWidth/4 #left column
                hpos = self.centerButton[0] - 1.5*buttonSize[0]
            elif tmp == 2:
                hpos = 2*self.screenWidth/4 #middle column
                hpos = self.centerButton[0]
            elif tmp == 0:
                hpos = 3*self.screenWidth/4 #right column
                hpos = self.centerButton[0] + 1.5*buttonSize[0]
      
      
            #find corresponding vertical position!
            if i<=3: #first row
                vpos = 1*self.screenHeight/4
                vpos = self.centerButton[1] - 1.5*buttonSize[0]
            elif i<=6: #2. row
                vpos = 2*self.screenHeight/4
                vpos = self.centerButton[1]
            elif i<=9: #3. row
                vpos = 3*self.screenHeight/4
                vpos = self.centerButton[1] + 1.5*buttonSize[0]
            
            centerPosition = (hpos, vpos)
            self.headerPos.append(centerPosition)
            
            buttonRect = button.get_rect(center=centerPosition, size=(button.get_width(), button.get_height()))
            button.fill(self.backgroundColor)    
            button = button.convert()
            
            self.buttonList.append(button)
            self.buttenRectList.append(buttonRect)
            pygame.draw.rect(button, self.stdColor, self.rectSizes, 3) 
            #end if self.visual
           
        
        #establish a header-box, where we later on give the instructions
        #this is needed also for non-visual stimuli!
        self.headerbox = pygame.Surface((self.screenWidth, int(self.screenHeight/4))).convert()
        self.headerboxRect = self.headerbox.get_rect(center=(self.headerPos[0][0],0))
        self.headerbox.fill(self.backgroundColor)
        
        self.footerbox = pygame.Surface((int(0.6*self.screenWidth), int(self.screenHeight/4))).convert()
        self.footerboxRect = self.footerbox.get_rect(topleft=(0, int(self.screenHeight * 0.8)))
        self.footerbox.fill(self.backgroundColor)
        
        # Some widgets.
        self.scrolledList= ocempgui.widgets.ScrolledList (int(0.4*self.screenWidth), int(self.screenHeight/6))
        self.scrolledList.scrolling = SCROLL_ALWAYS
        
        self.scrolledList.items.append(ocempgui.widgets.components.TextListItem("Start:" + time.strftime('%x %X')))
#            item = ocempgui.widgets.components.TextListItem ("Item with a different style")
#            item.editable = True
#            item.get_style ()["fgcolor"][STATE_NORMAL] = (100, 200, 100)
#            item.style["font"]["size"] = 20
#            self.scrolledList.items.append(item)
        self.renderer.add_widget (self.scrolledList)
        
        # Blit the Renderer's contents at the desired position.
        self.renderer.topleft = self.footer2Pos
        self.screen.blit (self.renderer.screen, self.renderer.topleft)
        pygame.time.set_timer (SIG_TICK, 10)
        
        
    def draw_initial(self):
        """
         draw buttons on the screen: 3x3 grid in std color
         """
        self.screen.blit(self.background, self.backgroundRect)
        for i in arange(1,10):
            button = self.buttonList[i]
            buttonRect = self.buttenRectList[i]
            self.screen.blit(button, buttonRect)
            
            #write the number (1..9) in the buttons 
            
            #self.do_print(str(i), self.stdColor, self.fontSizeBig, tuple(array(self.headerPos[i]) - [0,0]), True)
            self.do_print(self.alphabetLegend[i], self.stdColor, self.fontSizeBig, tuple(array(self.headerPos[i]) - [0,0]), True)
            #write specific letters ... space --> 'OPEN BOX' (U+2423)           
            #self.do_print(self.alphabetLegend[i], self.stdColor, self.fontSizeSmall, tuple(array(self.headerPos[i]) + [0,self.fontSizeSmall]), True)
        
        self.screen.blit (self.renderer.screen, self.renderer.topleft)    

        pygame.display.flip()  #draw all rects


    def printTextInHeader(self, text):
        """
        print a string into the header
        """
        self.do_print(text, (255,255,255), int(self.fontSizeBig*0.7), self.headerPos[0] , True)
        pygame.display.update(self.headerboxRect)
        #print "Header: ", text

    
    def printTextInFooter(self, text1):
        """
        print 2 strings into the footer. text1 is shown above of text2!
        """
        self.do_print(text1, (255,255,255), int(self.fontSizeSmall), self.footer1Pos[0] , True)
        pygame.display.update(self.footerboxRect)
        #print "Footer: ", text1, ":", text2

        
    def deleteTextInFooter(self): 
        """
        clear text in footer
        """
        self.screen.blit(self.footerbox, self.footerboxRect)
        pygame.display.update(self.footerboxRect)
       
      
    def deleteTextInHeader(self):
        """
        clear text in header
        """
        self.screen.blit(self.headerbox, self.headerboxRect)
        pygame.display.update(self.headerboxRect)

       
    def do_print(self, text, color, size=None, center=None, superimpose=False):
        """
        Print the given text in the given color and size on the screen.
        --> copied from GoalKeeper!!
        """
        if not color:
            color = self.fontColor
        if not size:
            size = self.size/10
        if not center:
            center = self.screen.get_rect().center

        font = pygame.font.Font(pygame.font.match_font('tahoma', False, False), size)
        #font = pygame.font.Font(None, size)
        if not superimpose:
            self.screen.blit(self.background, self.backgroundRect)
        surface = font.render(text, 1, color)
        self.screen.blit(surface, surface.get_rect(center=center))
        if not superimpose:
            pygame.display.update()

    
    def start_substim(self, istim):
        if self.visualStimuliEnabled:
            self.start_visual_cue(istim)
        self.start_auditory_cue(istim)

    
    def stop_substim(self, istim):
        """
        end of a subtrial, changing color of the 
        button back to normal if visual stimulus!
        """
        #not needed anymore if no visual input
        if self.visualStimuliEnabled:
            self.stop_visual_cue(istim)

    
    def start_visual_cue(self, iStimulus):
        self.changeButton(iStimulus,self.deviantColor)


    def stop_visual_cue(self, iStimulus):
        self.changeButton(iStimulus,self.stdColor)

      
    def changeButton(self, i, color):
        """
        set button (i.e. visual representation of one class) to a specified colour
        """
        button = self.buttonList[i]
        buttonRect = self.buttenRectList[i]
        pygame.draw.rect(button, color, self.rectSizes, 3)
        self.screen.blit(button, buttonRect)
        self.do_print(self.alphabetLegend[i], self.stdColor, self.fontSizeBig, tuple(array(self.headerPos[i]) - [0,0]), True)

        #self.do_print(str(i), color, self.fontSizeBig, tuple(array(self.headerPos[i]) - [0,0]), True)
        #self.do_print(self.alphabetLegend[i], color, self.fontSizeSmall, tuple(array(self.headerPos[i]) + [0,self.fontSizeSmall]), True)

        pygame.display.update(buttonRect)
        #pygame.display.flip()


    def show_mode1(self):
        """
        changes the button labels to the spelling (standard) mode.
        """
        self.alphabetLegend = self.stdAlphabetLegend
        self.draw_initial()
        #print "SHOW MODE1"


    def show_mode2(self, suggestions=[]):
        """
        changes the button labels to the selection mode.
        suggestions has to be a list of strings (not more than 5!)
        """
        if len(suggestions) > 5:
            self.logger.critical(time.strftime('%x %X') + " too many (>4) word-suggestions")
        else: #fill up the missing keys
            suggestions = suggestions + [""]*(5-len(suggestions)) 
            
        self.alphabetLegend = ["DUMP","ENTER","BACK", "DEL"] + suggestions + ["ENTWURF"]
        self.draw_initial()
        #print "SHOW MODE2"

        
    def show_mode3(self):
        """
        changes the button labels to the template mode.
        """
        
        self.alphabetLegend = ["DUMP"] + self.templateTexts + ["MENU"]
        self.draw_initial()
        #print "SHOW MODE2"

    
    def start_auditory_cue(self, iStimulus):
        """
        performs a single trial -> one auditory stimulus
        """     
        
        sound = self.soundlist[iStimulus]
        
        #self.audioChannel.play(sound) 
        sound.play()
        pygame.event.pump()
    
    
    def playSoundObject(self, sound):
        """
        play the sound with the given filename or mixer.Sund object with the class-intern channel 
        """
        if os.path.isfile(sound): #conversion if filename was given
            s = pygame.mixer.Sound(sound)
            print "sound loaded from path"
        else:
            s = sound
        print 'PLAY'
        self.audioChannel.play(s)
        pygame.event.pump()
        
    
    def manageEvents(self): 
        events = pygame.event.get()
        if len(events) > 0:
            self.renderer.distribute_events (*events)
            # Blit the renderer, too
            self.screen.blit(self.renderer.screen, self.renderer.topleft)
            pygame.display.flip ()
            #print events

    
    def newParagraph(self, txt):
        """
        writes the last paragraph into ScrolledList and starts a new paragraph!
        """
        item = ocempgui.widgets.components.TextListItem (txt)
        item.get_style ()["fgcolor"][STATE_NORMAL] = (100, 200, 100)
        item.style["font"]["size"] = self.fontSizeSmall
        item.editable = True
        self.scrolledList.items.append(item)


    def printTemplate(self, txt):
        """
        
        """
        item = ocempgui.widgets.components.TextListItem (txt)
        item.editable = True
        item.get_style ()["fgcolor"][STATE_NORMAL] = (100, 200, 100)
        item.style["font"]["size"] = self.fontSizeSmall
        self.scrolledList.items.append(item)

        
    def wait(self, t):
        pygame.time.wait(t)

        

if __name__ == "__main__":
    import logging
    from numpy import random
    logging.basicConfig(level=logging.DEBUG)
    
    sounds = ["sounds/set8-2/1.wav", "sounds/set8-2/2.wav", "sounds/set8-2/3.wav", \
          "sounds/set8-2/4.wav", "sounds/set8-2/5.wav", "sounds/set8-2/6.wav", \
          "sounds/set8-2/7.wav", "sounds/set8-2/8.wav", "sounds/set8-2/9.wav"]
    
    templateTexts = ["absaugen", "umlagern", "CUFF", "ALARM", "Kopf nach links", "Kopf nach rechts", "Danke", "System Pause 3min"]
    screenPos = [100, 100, 800, 600]
    tp = TrialPresentation(screenPos, sounds, templateTexts)
    randSeq = range(10)[1:] #first seq 1:9  !!
    inMode2 = False
    
    while 1:
        i = randSeq[0]
        tp.start_auditory_cue(i)
        tp.start_visual_cue(i)
        
        pygame.time.wait(800)
        tp.stop_visual_cue(i)
        if len(randSeq) > 1:
            randSeq = randSeq[1:]
        else:
            randSeq = (random.permutation(9)+1).tolist()
            #randSeq = range(10)[1:]
        pygame.time.wait(200)
    
    pygame.quit()
    
