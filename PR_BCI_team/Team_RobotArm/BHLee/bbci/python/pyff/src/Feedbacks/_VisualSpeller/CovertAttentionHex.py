'''
Created on Nov 18, 2009

@author: nico schmidt

'''
import colorsys
import pygame
import numpy as np
import sys
import os

from FeedbackBase.MainloopFeedback import MainloopFeedback
from lib.P300Layout.CircularLayout import CircularLayout 
from lib.P300VisualElement.Circle import Circle

class CovertAttentionHex(MainloopFeedback):
    '''
    classdocs
    '''
    
    # Triggers:
    SHORTPAUSE_START, SHORTPAUSE_END = 249, 250
    RUN_START, RUN_END = 252,253
    COUNTDOWN_START = 240
    FIXATION_START = 241
    DIRECTINGATTENTION_START = 242
    MASKER_START = 243
    RESPONSE_START = 244
    REST_START = 245
    CUE_START = 10 # 10-15
    TARGET_START = 30 # 30-35 for '+' and 36-41 for 'x'
    KEY_1, KEY_2 = 100,101
    
    def init(self):
        
        # sizes:
        self.geometry = [0, 0, 1680, 1050]
        self.canvas_width, self.canvas_height = 900, 900
        self.fullscreen = False
        self.fixationpoint_radius = 5
        self.cue_size = 100
        self.cue_bordersize = 3
        self.display_radius = 400
        self.nr_elements = 6
        self.circle_radius = 80
        self.circle_textsize = 70
        self.circle_textwidth = 9
        
        
        # trials:
        self.nTrials = 100 # number of trials
        self.frac_longest = 0.5 # fraction of trials with longest duration (see directattention_duration)
        self.frac_shortest = 0.2 # fraction of trials with shortest duration (see directattention_duration)
        self.frac_correct = 0.8 # fraction of correct trials (cue=target)
        
        # response options
        self.rsp_keys = ['m', 'x'] # which keys to press as response to '+' and 'x'
        
        # times (all in milliseconds):
        self.FPS = 50 # frame rate
        self.countdown_from = 4000
        self.fixation_duration = 1000
        self.cue_duration = 100
        self.directingattention_duration = [500, 2000]
        self.target_duration = 200
        self.masker_duration = 140
        self.responsetime_duration = 700
        self.rest_duration = 1000
        
        # colors:
        self.bgcolor = (0, 0, 0)
        self.countdown_color = (200, 80, 118)
        self.font_color = (0, 150, 150)
        self.circle_color = (255, 255, 255)
        self.circle_textcolor = (0, 0, 0)
        self.fixationpoint_color = (255, 255, 255)
        self.cue_colors = [(0.0, 1.0, 0.8), \
                           (0.2, 0.0, 0.8), \
                           (0.4, 1.0, 0.8), \
                           (0.2, 0.0, 0.8), \
                           (0.7, 1.0, 0.8), \
                           (0.2, 0.0, 0.8)]
        self.cue_bordercolor = (0.0, 0.0, 1.0)
        self.user_cuecolor = 4 # the index of the color, which is used as cue            
    
    
    def pre_mainloop(self):
        """
        initializes variables, triallist, graphics and pygame.
        """        
        self.accept_response = False
        self.elements = []
        
        # Feedback state booleans
        self.state_pause = False
        self.state_countdown = True  
        self.state_fixation = False   
        self.state_cue = False
        self.state_directingattention = False
        self.state_target = False
        self.state_masker = False
        self.state_response = False
        self.state_rest = False
        
        self.state_time_elapsed = 0
        
        self.currentTrial = 0
        
        self.build_trials()
#          
        # calc needed time:
        self.stat_time_needed = sum([t[3] for t in self.trialList]) + self.nTrials * (self.fixation_duration +
                                                                                      self.cue_duration +
                                                                                      self.responsetime_duration + 290 +
                                                                                      self.rest_duration) + self.countdown_from
        
        print "time needed for experiment: approx. %d:%d" % (np.floor(self.stat_time_needed/1000/60.), np.mod(self.stat_time_needed/1000, 60)), "(mm:ss)"
        
#        self.save_triallist_stats()
        
        self.cue_color = [tuple([255*i for i in colorsys.hsv_to_rgb(*self.cue_colors[j])]) for j in xrange(self.nr_elements)]
        self.cue_color.append(tuple([255*i for i in colorsys.hsv_to_rgb(*self.cue_bordercolor)]))
        
        self.send_parallel(self.RUN_START)
        self.init_pygame()
        self.init_graphics()    
    
    
    def post_mainloop(self):
        """
        Sends end marker to parallel port and quits pygame.
        """
        self.send_parallel(self.RUN_END)
        pygame.quit()        
    
    
    def tick(self):
        """
        called every frame.
        """
        self.process_pygame_events()
        pygame.time.wait(10)
        self.elapsed = self.clock.tick(self.FPS)
    
    
    def pause_tick(self):
        """
        called every frame, if in pause mode.
        """
        if not self.state_pause:
            self.state_pause = True
            self.do_print("Pause", self.font_color, self.size / 6)
            self.send_parallel(self.SHORTPAUSE_START)
            
    
    
    def play_tick(self):
        """
        called every frame, if in play mode.
        """
        if self.state_pause:
            self.state_pause = False
            self.send_parallel(self.SHORTPAUSE_END)
            self.draw_screen(self.state_cue)
        elif self.state_countdown:
            self.countdown_tick()
        elif self.state_fixation:
            self.fixation_tick()
        elif self.state_cue:
            self.cue_tick()
        elif self.state_directingattention:
            self.directingattention_tick()
        elif self.state_target:
            self.target_tick()
        elif self.state_masker:
            self.masker_tick()
        elif self.state_response:
            self.responsetime_tick()
        elif self.state_rest:
            self.rest_tick()
        else:
            self.on_stop()
    
    
    def countdown_tick(self):
        """
        One tick of the countdown loop.
        """        
        # start countdown:
        if self.state_time_elapsed == 0:
            self.send_parallel(self.COUNTDOWN_START)
        
        # draw countdown on screen:
        t = ((self.countdown_from + 1000) - self.state_time_elapsed) / 1000
        self.do_print(str(t), self.countdown_color, self.size / 4)
                
        # stop countdown:
        if self.state_time_elapsed >= self.countdown_from:
            self.state_countdown = False
            self.state_fixation = True
            self.state_time_elapsed = 0
            return
        else:
            self.state_time_elapsed += self.elapsed
    
    
    def fixation_tick(self):        
        """
        One tick of the fixation loop.
        """        
        # start fixation:
        if self.state_time_elapsed == 0:
            self.send_parallel(self.FIXATION_START)
            self.draw_screen()
        
        # stop fixation:
        if self.state_time_elapsed >= self.fixation_duration:
            self.state_fixation = False
            self.state_cue = True
            self.state_time_elapsed = 0
        else:
            self.state_time_elapsed += self.elapsed
    
    
    def cue_tick(self):             
        """
        One tick of the cue loop.
        """        
        # start cue:
        if self.state_time_elapsed == 0:
            self.send_parallel(self.CUE_START+self.trialList[self.currentTrial][1])
            self.draw_screen(True)
        
        # stop cue:
        if self.state_time_elapsed >= self.cue_duration:
            self.state_cue = False
            self.state_directingattention = True      
            self.state_time_elapsed = 0
        else:
            self.state_time_elapsed += self.elapsed 
    
    
    def directingattention_tick(self):           
        """
        One tick of the directingattention loop.
        """        
        # start directingattention:
        if self.state_time_elapsed == 0:
            self.send_parallel(self.DIRECTINGATTENTION_START)
            self.draw_screen()
                       
        # stop directingattention:
        if self.state_time_elapsed >= self.trialList[self.currentTrial][3]:
            self.state_directingattention = False
            self.state_target = True
            self.state_time_elapsed = 0
        else:
            self.state_time_elapsed += self.elapsed 
    
    
    def target_tick(self):         
        """
        One tick of the target loop.
        """
        # start target:
        if self.state_time_elapsed == 0:
            self.send_parallel(self.TARGET_START + self.trialList[self.currentTrial][0] + 6*(self.trialList[self.currentTrial][2]-1))
            direction = self.trialList[self.currentTrial][0]
            symbol = self.trialList[self.currentTrial][2]
            self.elements[direction].update(symbol)
            self.draw_screen()
        
        # stop target:
        if self.state_time_elapsed >= self.target_duration:
            self.state_target = False
            self.state_masker = True
            self.state_time_elapsed = 0
        else:
            self.state_time_elapsed += self.elapsed
    
    
    def masker_tick(self):         
        """
        One tick of the masker loop.
        """
        # start masker:
        if self.state_time_elapsed == 0:
            self.send_parallel(self.MASKER_START)
            direction = self.trialList[self.currentTrial][0]
            self.elements[direction].update(3)
            self.draw_screen()
            
        # stop masker:
        if self.state_time_elapsed >= self.masker_duration:
            self.state_masker = False
            self.state_response = True
            self.state_time_elapsed = 0
        else:
            self.state_time_elapsed += self.elapsed
            
    
    def responsetime_tick(self):     
        """
        One tick of the responsetime loop.
        """
        # start responsetime:
        if self.state_time_elapsed == 0:            
            self.send_parallel(self.RESPONSE_START)
            direction = self.trialList[self.currentTrial][0]
            self.elements[direction].update(0)
            self.draw_screen()
            self.accept_response = True
        
        # stop responsetime:
        if self.state_time_elapsed >= self.responsetime_duration:
            self.accept_response = False
            self.state_response = False
            self.state_rest = True
            self.state_time_elapsed = 0
        else:
            self.state_time_elapsed += self.elapsed
    
    
    def rest_tick(self):
        """
        One tick of the rest loop.
        """        
        # start rest:
        if self.state_time_elapsed == 0:
            self.send_parallel(self.REST_START)
        
        # stop rest:
        if self.state_time_elapsed >= self.rest_duration:
            self.state_rest = False
            self.state_time_elapsed = 0
            if self.currentTrial < self.nTrials-1:
                self.state_fixation = True
            self.currentTrial += 1
        else:
            self.state_time_elapsed += self.elapsed 
    
    
    def draw_screen(self, drawcue=False):
        """
        update the graphics.
        """        
        self.screen.blit(self.background,self.background_rect)
        self.all_elements_group.draw(self.screen)
        if drawcue:
            direction = self.trialList[self.currentTrial][1]
            self.screen.blit(self.cue[direction], self.cue_rect)  
        else:
            self.screen.blit(self.point, self.pointRect)
        pygame.display.flip()
    
    
    def do_print(self, text, color, size=None, center=None, superimpose=True):
        """
        Print the given text in the given color and size on the screen.
        """
        if not color:
            color = self.font_color
        if not size:
            size = self.size / 10
        if not center:
            center = self.screen.get_rect().center
        
        font = pygame.font.Font(None, size)
        if not superimpose:
            self.screen.blit(self.background, self.background_rect)            
        surface = font.render(text, 1, color,self.bgcolor)    
        self.screen.blit(surface, surface.get_rect(center=center))
        pygame.display.flip()
    
    
    def process_pygame_events(self):
        """
        Process the pygame event queue.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            elif  event.type == pygame.KEYDOWN:
                if event.unicode == unicode("q"):
                    sys.exit()
                elif self.accept_response:
                    if event.unicode == unicode(self.rsp_keys[0]): # Plus (+)-key was pressed
                        self.send_parallel(self.KEY_1)
                        self.accept_response = False
                        if self.trialList[self.currentTrial][2] == 1:
                            print "correct"
                        else:
                            print "wrong"
                    elif event.unicode == unicode(self.rsp_keys[1]): # (x)-key was pressed
                        self.send_parallel(self.KEY_2)
                        self.accept_response = False
                        if self.trialList[self.currentTrial][2] == 2:
                            print "correct"
                        else:
                            print "wrong"
    
    
    def init_graphics(self):
        """
        Initialize the surfaces and fonts.
        """
        self.size = min(self.screen.get_height(), self.screen.get_width())
        
        # init background
        self.background = pygame.Surface([self.canvas_width, self.canvas_height]) 
        self.background.fill(self.bgcolor)
        self.background_rect = self.background.get_rect(center=[self.geometry[2]/2, self.geometry[3]/2])
        # Background for whole screen (needs lots of time to paint, use self.background in most cases) 
        self.all_background = pygame.Surface([self.geometry[2], self.geometry[3]]) 
        self.all_background.fill(self.bgcolor)
        self.all_background_rect = self.all_background.get_rect(center=[self.geometry[2]/2, self.geometry[3]/2])
        self.screen.blit(self.all_background, self.all_background_rect)
        
        # init 6 circles:
        layout = CircularLayout(nr_elements=self.nr_elements, radius=self.display_radius)
        a = self.circle_textsize
        b = self.circle_textsize/2
        c = int(round(np.sin(np.pi/4.) * float(b)))
        p1, p2, p3, p4 = [b,0], [b,a], [0,b], [a,b]
        p5, p6, p7, p8 = [b-c, b-c], [b+c, b+c], [b-c, b+c], [b+c, b-c]
        img_plus = pygame.Surface([a, a]) # target image '+'
        img_plus.fill(self.circle_color)
        pygame.draw.line(img_plus, self.circle_textcolor, p1, p2, self.circle_textwidth)
        pygame.draw.line(img_plus, self.circle_textcolor, p3, p4, self.circle_textwidth)
        
        img_x = pygame.Surface([a,a]) # target image 'x'
        img_x.fill(self.circle_color)
        pygame.draw.line(img_x, self.circle_textcolor, p5, p6, int(round(self.circle_textwidth*1.5)))
        pygame.draw.line(img_x, self.circle_textcolor, p7, p8, int(round(self.circle_textwidth*1.5)))
        
        img_plus_x = pygame.Surface([a,a]) # masker image with '+' and 'x'
        img_plus_x.fill(self.circle_color)
        pygame.draw.line(img_plus_x, self.circle_textcolor, p1, p2, self.circle_textwidth)
        pygame.draw.line(img_plus_x, self.circle_textcolor, p3, p4, self.circle_textwidth)
        pygame.draw.line(img_plus_x, self.circle_textcolor, p5, p6, int(round(self.circle_textwidth*1.5)))
        pygame.draw.line(img_plus_x, self.circle_textcolor, p7, p8, int(round(self.circle_textwidth*1.5)))
        
        for i in range(self.nr_elements): # create circle elements            
            e = CircleWithImage(nr_states=4, textcolor=self.circle_textcolor, textsize=self.circle_textsize, color=self.circle_color, radius=self.circle_radius, colorkey=(0, 0, 0), circular_layout=False, circular_offset= - np.pi / 2)
            e.set_states(0, {"image":None}) # blank circle
            e.set_states(1, {"image":img_plus}) # + as target
            e.set_states(2, {"image":img_x}) # x as target
            e.set_states(3, {"image":img_plus_x}) # masker state
            
            # Position element so that it is centered on the screen 
            (x,y) = layout.positions[len(self.elements)]
            e.pos = (x+self.geometry[2]/2,y+self.geometry[3]/2)
            self.elements.append(e)
            e.refresh()
            e.update(0)
        
        self.all_elements_group = pygame.sprite.RenderUpdates(self.elements)
        self.all_elements_group.update(0)
        
        # init fixation point
        self.pointSize = (self.fixationpoint_radius*2, self.fixationpoint_radius*2)  
        self.point = pygame.Surface(self.pointSize)  
        self.point.fill(self.bgcolor)    
        pygame.draw.circle(self.point, self.fixationpoint_color, [self.fixationpoint_radius, self.fixationpoint_radius], self.fixationpoint_radius )
        self.pointRect = self.point.get_rect(center=[self.geometry[2]/2, self.geometry[3]/2])
        
        # init cues:
        self.cue = [None]*self.nr_elements
        a = float(self.cue_size)
        b = np.ceil(np.sqrt(a**2 - a**2/4))
        p0 = [int(round(a/2.)), int(round(b/2))]
        p1 = [int(round(a/4.)), 0]
        p2 = [int(round(3.*a/4.)), 0]
        p3 = [int(round(a)), int(round(b/2.))]
        p4 = [int(round(3.*a/4.)), int(round(b))]
        p5 = [int(round(a/4.)), int(round(b))]
        p6 = [0, int(round(b/2.))]
        for i in xrange(self.nr_elements):
            self.cue[i] = pygame.Surface([a,b])
            self.cue[i].fill(self.bgcolor)
            # triangles:
            pygame.draw.polygon(self.cue[i], self.cue_color[np.mod(0-i+self.user_cuecolor,self.nr_elements)], [p0, p1, p2])
            pygame.draw.polygon(self.cue[i], self.cue_color[np.mod(1-i+self.user_cuecolor,self.nr_elements)], [p0, p2, p3])
            pygame.draw.polygon(self.cue[i], self.cue_color[np.mod(2-i+self.user_cuecolor,self.nr_elements)], [p0, p3, p4])
            pygame.draw.polygon(self.cue[i], self.cue_color[np.mod(3-i+self.user_cuecolor,self.nr_elements)], [p0, p4, p5])
            pygame.draw.polygon(self.cue[i], self.cue_color[np.mod(4-i+self.user_cuecolor,self.nr_elements)], [p0, p5, p6])
            pygame.draw.polygon(self.cue[i], self.cue_color[np.mod(5-i+self.user_cuecolor,self.nr_elements)], [p0, p6, p1])
            # border:
            pygame.draw.polygon(self.cue[i], self.cue_color[-1], [p1, p2, p3, p4, p5, p6], self.cue_bordersize)
#            pygame.draw.lines(self.cue[i], self.cue_color[12], True, [p1a, p2a, p3a, p4a, p5a, p6a], self.cue_bordersize)
            # fixationpoint:
            pygame.draw.circle(self.cue[i], self.fixationpoint_color, p0, self.fixationpoint_radius )
        
        self.cue_rect = self.cue[0].get_rect(center=[self.geometry[2]/2,self.geometry[3]/2], size=[a,b])
        
        pygame.display.flip()
    
    
    def init_pygame(self):
        """
        initialize sreen, start pygame.
        """
        # Initialize pygame, open screen and fill screen with background color
        os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (self.geometry[0], self.geometry[1])   
        pygame.init()
        pygame.display.set_caption('Covert Attention Hex')
        if self.fullscreen: 
            #use opts = pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.FULLSCREEN to use doublebuffer and vertical sync
            opts = pygame.FULLSCREEN
            self.screen = pygame.display.set_mode((self.geometry[2],self.geometry[3]),opts)
        else: 
            self.screen = pygame.display.set_mode((self.geometry[2],self.geometry[3]))
        self.clock = pygame.time.Clock()
    
    
    def build_trials(self):  
        """
        build the list of trials as [[target_direction, cue_direction, symbol, trial_duration], ...]
        build the trial numbers deterministic for the different durations and for the validity,
        but probabilistic for the target location and target symbol.
        """      
        # build list of fractions of trial conditions:
        fractions = [self.frac_longest                          *    self.frac_correct,     # valid trials with longest duration
                     self.frac_longest                          * (1-self.frac_correct),    # invalid trials with longest duration
                     self.frac_shortest                         *    self.frac_correct,     # valid trials with shortest duration
                     self.frac_shortest                         * (1-self.frac_correct),    # invalid trials with shortest duration
                     (1-self.frac_longest-self.frac_shortest)   *    self.frac_correct,     # valid trials with random duration
                     (1-self.frac_longest-self.frac_shortest)   * (1-self.frac_correct)]    # invalid trials with random duration
        
        # calc quantity of Trials for each condition:
        fraction_quantities = np.zeros(len(fractions))
        nTrials = 0
        while nTrials < self.nTrials:
            differences = [fractions[i] - fraction_quantities[i]/nTrials for i in xrange(len(fractions))]
            index_max = np.argmax(differences) # get index of max difference
            fraction_quantities[index_max] += 1 # increase state with max distance
            nTrials += 1
#            diff = 100*sum(np.abs(differences))
#            print index_max, fraction_quantities
#            print "distance to optimal fractions:", diff, "%"
#            print
        
        fractions_actual = [fraction_quantities[i]/self.nTrials for i in xrange(len(fractions))]
        self.stat_differences = [fractions[i] - fractions_actual[i] for i in xrange(len(fractions))]
#        for i in xrange(len(fraction_quantities)):
#            print str(round(fractions[i],5)), str(round(fractions_actual[i],5)), self.stat_differences[i]
#        print fraction_quantities
        print "distance to optimal fractions:", 100*sum(np.abs(self.stat_differences)), "%"
        
        # fill trial list using the calculated quantities:
        self.trialList = []
        a = np.array(range(6))
        nLocations = [0, 0, 0, 0, 0, 0]
        for _ in xrange(int(fraction_quantities[0])): # valid cue, maximal duration
            cue = np.argmin(nLocations)
            target = cue
            symbol = np.random.randint(1,3)
            nLocations[cue] += 1
            self.trialList.append([target, cue, symbol, self.directingattention_duration[1]])
            
        for _ in xrange(int(fraction_quantities[1])): # invalid cue, maximal duration
            cue = np.argmin(nLocations)
            target = a[a!=cue][np.random.randint(5)]
            symbol = np.random.randint(1,3)
            nLocations[cue] += 1
            self.trialList.append([target, cue, symbol, self.directingattention_duration[1]])
            
        for _ in xrange(int(fraction_quantities[2])): # valid cue, minimal duration
            cue = np.argmin(nLocations)
            target = cue
            symbol = np.random.randint(1,3)
            nLocations[cue] += 1
            self.trialList.append([target, cue, symbol, self.directingattention_duration[0]])
            
        for _ in xrange(int(fraction_quantities[3])): # invalid cue, minimal duration
            cue = np.argmin(nLocations)
            target = a[a!=cue][np.random.randint(5)]
            symbol = np.random.randint(1,3)
            nLocations[cue] += 1
            self.trialList.append([target, cue, symbol, self.directingattention_duration[0]])
            
        for _ in xrange(int(fraction_quantities[4])): # valid cue, random duration
            cue = np.argmin(nLocations)
            target = cue
            symbol = np.random.randint(1,3)
            nLocations[cue] += 1
            self.trialList.append([target, cue, symbol, np.random.randint(self.directingattention_duration[0], self.directingattention_duration[1])])
            
        for _ in xrange(int(fraction_quantities[5])): # invalid cue, random duration
            cue = np.argmin(nLocations)
            target = a[a!=cue][np.random.randint(5)]
            symbol = np.random.randint(1,3)
            nLocations[cue] += 1
            self.trialList.append([target, cue, symbol, np.random.randint(self.directingattention_duration[0], self.directingattention_duration[1])])
        
#        symbols = ["+", "x"]
#        for t in self.trialList:
#            print "target:", t[0], "cue:", t[1], "symbol:", symbols[t[2]-1], "time:", t[3]
        tl = np.array(self.trialList)
        symbols = tl[:,2]
        print 
        print "symbol = x:",len(symbols[symbols==2]), "(%.1f%%)" % (len(symbols[symbols==2])/float(self.nTrials)*100)
        print "symbol = +:",len(symbols[symbols==1]), "(%.1f%%)" % (len(symbols[symbols==1])/float(self.nTrials)*100)
        targets = tl[:,0]
        for i in xrange(6): print "target = %d:" % i, len(targets[targets==i]), "(%.1f%%)" % (len(targets[targets==i])/float(self.nTrials)*100)
        
        # shuffle trial list:
        np.random.shuffle(self.trialList)
    
    def evaluate_triallist(self):
        self.stat_symbol_locations = np.zeros([2,6])
        self.stat_symbol_locations[self.trialList[0][2]-1, self.trialList[0][0]] += 1
        self.stat_followers_target = np.zeros([6,6])
        self.stat_followers_cue = np.zeros([6,6])
        this_target = self.trialList[0][0]
        this_cue = self.trialList[0][1]
        next_target = -1
        next_cue = -1
        for i in xrange(1,self.nTrials):
            next_target = self.trialList[i][0]
            next_cue = self.trialList[i][1]
            self.stat_followers_target[this_target, next_target] += 1
            self.stat_followers_cue[this_cue, next_cue] += 1
            self.stat_symbol_locations[self.trialList[i][2]-1, next_target] += 1
            this_target = next_target
            this_cue = next_cue
        
    def save_triallist_stats(self):
        from datetime import datetime
        
        self.evaluate_triallist()
        
        f = open("triallist_stats.dat", "w")
        tl = np.array(self.trialList)
        symbols = tl[:,2]
        targets = tl[:,0]
        cues = tl[:,1]
        f.write('CovertAttentionHex experiment on %s:\n' % str(datetime.now()))
        f.write('time needed for experiment: approx. %d:%d (mm:ss)\n' % (np.floor(self.stat_time_needed/1000/60.), np.mod(self.stat_time_needed/1000, 60)))
        f.write('distance to optimal fractions: %s%%\n' % str(100*sum(np.abs(self.stat_differences))))
        f.write('symbol = x: %d, (%.1f%%)\n' % (len(symbols[symbols==2]), len(symbols[symbols==2])/float(self.nTrials)*100))
        f.write('symbol = +: %d, (%.1f%%)\n' % (len(symbols[symbols==1]), len(symbols[symbols==1])/float(self.nTrials)*100))
        for i in xrange(6):
            f.write('cue = %d: %d (%.1f%%)\n' % (i, len(cues[cues==i]), len(cues[cues==i])/float(self.nTrials)*100))
        for i in xrange(6):
            f.write('target = %d: %d (%.1f%%)\n' % (i, len(targets[targets==i]), len(targets[targets==i])/float(self.nTrials)*100))
        f.write('\nfollowers_target:\n')
        for [a,b,c,d,e,f_] in self.stat_followers_target:
            f.write('[%d, %d, %d, %d, %d, %d]\n' % (a,b,c,d,e,f_))
        f.write('\nfollowers_cue:\n')
        for [a,b,c,d,e,f_] in self.stat_followers_cue:
            f.write('[%d, %d, %d, %d, %d, %d]\n' % (a,b,c,d,e,f_))
        f.write('\nsymbol locations:\n')
        for [a,b,c,d,e,f_] in self.stat_symbol_locations:
            f.write('[%d, %d, %d, %d, %d, %d]\n' % (a,b,c,d,e,f_))
        f.write('\ntriallist:\n')
        for [t, c, s, d] in self.trialList:
            f.write('[%d, %d, %d, %d]\n' % (t,c,s,d))
        f.write('\n\n\n')
        f.close()


class CircleWithImage(Circle):

    DEFAULT_TEXT = None
    DEFAULT_TEXTCOLOR = 200, 200, 200
    DEFAULT_TEXTSIZE = 20
    DEFAULT_COLOR = 255, 255, 0
    DEFAULT_RADIUS = 30
    DEFAULT_ANTIALIAS = True
    DEFAULT_COLORKEY = None
    DEFAULT_CIRCULAR_LAYOUT = False   
    DEFAULT_CIRCULAR_OFFSET = 0
    DEFAULT_BUNTER_KREIS = False
    DEFAULT_IMAGE = None
    
    def __init__(self, nr_states=2, pos=(0, 0), text=DEFAULT_TEXT, textcolor=DEFAULT_TEXTCOLOR, textsize=DEFAULT_TEXTSIZE, color=DEFAULT_COLOR, radius=DEFAULT_RADIUS, antialias=DEFAULT_ANTIALIAS, colorkey=DEFAULT_COLORKEY, circular_layout=DEFAULT_CIRCULAR_LAYOUT, circular_offset=DEFAULT_CIRCULAR_OFFSET, bunter_kreis=DEFAULT_BUNTER_KREIS, image=DEFAULT_IMAGE):
        
        Circle.__init__(self, nr_states, pos, text, textcolor, textsize, color, radius, antialias, colorkey, circular_layout, circular_offset, bunter_kreis)
        self.img = image
    
    
    def refresh(self):
        Circle.refresh(self)
        # For each state, add img to image:
        for i in range(self.nr_states):
            if self.states[i].has_key("image"): img = self.states[i]["image"]
            else: img = self.img
            if self.states[i].has_key("radius"):    radius = self.states[i]["radius"]
            else: radius = self.radius
            if img:
                img_rect = img.get_rect()
                w2, h2 = img_rect.width / 2, img_rect.height / 2
                
                image = self.images[i]
                image.blit(img, (radius - w2, radius - h2))
                
                image = image.convert()
                self.images[i] = image 
                self.rects[i] = self.images[i].get_rect(center=self.pos)



if __name__ == "__main__":
    fb = CovertAttentionHex()
    fb.on_init()
    fb.on_play()
