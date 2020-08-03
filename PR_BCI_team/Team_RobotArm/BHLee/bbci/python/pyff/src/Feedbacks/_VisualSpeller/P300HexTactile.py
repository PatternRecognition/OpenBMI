# HexTactileTrainer.py -
# Copyright (C) 2009  Chris Haeusler
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

"""
HexTactileTrainer can be used to test the accuracy with users can recognise tactile/auditory 
stimulus at vary frequencies, durations and repetition times. The results can then be used to fine tune
P300 experiments using tactile or auditory stimulus
The user is displayed 6 hexagons on the screen, each of which corresponds to one of the 
tactile/auditory stimulation positions. THe tactile/auditory stimulators are the activated randomly and when
the stimulator is active at the same time as  its corresponding hexagon on screen, the user should press the
space bar.
A report of the number of hits and misses is saved to a log file at the end.
"""

import sys, os, math
import numpy as np
import pygame
import logging as L

from FeedbackBase.VisualP300 import VisualP300
from lib.P300Layout.CircularLayout import CircularLayout 
from lib.P300VisualElement.Circle import Circle
from lib.P300VisualElement.Hexagon import Hexagon
from lib.P300VisualElement.Textrow import Textrow
from lib.P300Aux.P300Functions import *



class P300HexTactile(VisualP300):
    
    DEFAULT_DISPLAY_RADIUS = 220        
    DEFAULT_NR_ELEMENTS = 6
    
    # Pre codes
    PRE_WORD = 0
    PRE_COUNTDOWN = 1
    PRE_WAIT = 2
    PREP_TRIAL = 3
    
    # Trigger
    START_TRIAL_LV1 = 100         # Start trial level 1
    START_TRIAL_LV2 = 101         # Start trial level 2
    INVALID = 66               # invalid trial
    HEX_CENTRAL_FIX = 82# central fixation condition
    HEX_TARGET_FIX = 83    # target fixation condition
    
    def init(self):
        VisualP300.init(self)
        
        # setup some logging stuff
        self.logger = L.getLogger("p300HexTactile_Logger")
        self.logger.setLevel(L.INFO)
        log_handler = L.StreamHandler() #self.logger.name + ".log")
        log_format = L.Formatter("%(levelname)s %(asctime)s %(funcName)s "\
                                                "%(lineno)d %(message)s") 
        log_handler.setFormatter(log_format)
        log_handler.setLevel(L.DEBUG)
        
        self.logger.addHandler(log_handler)       
        self.tactile_stimulus = True
        self.visual_stimulus = False
        
        self.display_radius = self.DEFAULT_DISPLAY_RADIUS
        self.nr_elements = self.DEFAULT_NR_ELEMENTS   
        # Graphical settings
        #self.speller_fade_time = 1500   # How long speller is faded in/out
        self.pygame_info = False
        self.bgcolor = 0, 0, 0
        self.screenWidth, self.screenHeight = 800, 800
        self.canvasWidth, self.canvasHeight = 660, 800
        self.fullscreen = False        
        # Hex speller settings
        # Hex level can be 0 (group level) or 1 (element level)
        self.hex_time = 100     # Time for each circle takes to enter the stage
        self.hex_letters = ("CDE AB", "GHIJ F", "KLMNO ", " PQRST", "Y UVWX", ".< Z,_")
        self.words = ["WINKT", "FJORD", "LUXUS", "SPHINX", "QUARZ", "VODKA", "YACHT", "GEBOT", "MEMME"] 
        " Overwritten members of VisualP300 "
        
        self.word_time = 50            # How long word is shown (# fps)
        self.word_time_ms = 2500        # word time in ms
        self.min_dist = 2           # Min number of intermediate flashes bef. a flash is repeated twice 
        self.flash_duration = 160   # 5 frames @60 Hz = 83ms flash
        self.soa = 5
        self.hex_countdown = [1, 0]  # How many seconds countdown before level 1 and level 2
        self.nr_sequences = 1   # how many visual sequences 
        self.nr_tactile_sequences = 10 # number of tactile sequences
        self.trial_nr = 1 
        self.fps = 30
        self.fixation_dot = True
        
        # Timing
        self.animation_time = 20        # Length of animation in #frames
        """ Einige Programm teile mit 30fps,
        mit EEG und Refresh rate synchronisierte teile mit 60 fps
        
        LOGIK:
        Wenn die refresh rate (60Hz) den fps (60) entspricht, und 
        2 flip commandos aufgerufen werden, dann blockt der zweite
        bis der erste flip gezeichnet wurde
         
        """
        # Triggers for Level 1 (letter groups) and Level 2 (individual letters)  
        self.hex_group_triggers = [ [11, 12, 13, 14, 15, 16] , [21, 22, 23, 24, 25, 26] ]
        # If a certain group is a target, this value will be added to the trigger  
        self.trigger_target_add = 20
                
        # For data logging (-> the data file is opened in pre_mainloop)
        self.datafilename = "c:/temp/datafile_hex.txt"
        self.datafile = None
        self.pre_mode = self.PREP_TRIAL
              
      

    def before_mainloop(self):
        """
        Get a matrix layout, add circle elements and add groups according
        to rows and columns. 
        """
        
        # There are 7 hex displays, one for the group level and six for the subgroups
        self.hex_displays = [None] * 7
        self.hex_groups = [None] * 7
        
        # Get layout & elements
        self.layout = CircularLayout(nr_elements=self.nr_elements, radius=self.display_radius)
        colors = ((255, 255, 255), (255, 100, 255), (255, 255, 100), (255, 100, 100), (100, 100, 255), (100, 255, 255))
        textcolors = ((255, 0, 0) , (0, 255, 0) , (0, 0, 255) , (0, 255, 255) , (255, 155, 0), (0, 20, 160), (0, 255, 0) , (0, 0, 255) , (0, 255, 255) , (255, 155, 0), (0, 100, 200))
        self.hex_textcolor = (255, 0, 0)
        color , radius = (255, 255, 255), 60
        
        # Create the top-level display
        for i in range(self.nr_elements):
            e = Circle(nr_states=3, color=color, radius=radius, text=self.hex_letters[i], textcolor=self.hex_textcolor, colorkey=(0, 0, 0), circular_layout=True, circular_offset= - math.pi / 2)
            e.set_states(0, {"textsize":45, "radius":74})
            e.set_states(1, {"textsize":70, "radius":100})
            # Also add a blank version (for animation)
            e.set_states(2, {"textsize":25, "radius":74, "text":"" , "circular_layout":False})
              
            self.add_element(e)
            e.refresh()
            e.update(0)
        
        # Get groups and add them
        for i in range(self.nr_elements):
            self.add_group(i)
        
        # Add deco to deco group
        if len(self.deco) > 0:
            self.deco_group = pygame.sprite.RenderUpdates(self.deco)
        
        # Add text row
        self.textrow = Textrow(text="", textsize=42, color=(255, 255, 255), size=(450, 42), edgecolor=(55, 100, 255), antialias=True, colorkey=(0, 0, 0), highlight=[1], highlight_color=(255, 0, 0), highlight_size=62)
        self.textrow.pos = (self.screenWidth / 2, (self.screenHeight - self.canvasHeight) / 2 + 22)
        self.textrow.refresh()
        self.textrow.update()
        self.deco.append(self.textrow)
        # Add count row (where count is entered by participant)
        self.countrow = Textrow(text="", textsize=60, color=(150, 150, 255), size=(100, 60), edgecolor=(255, 255, 255), antialias=True, colorkey=(0, 0, 0))
        self.countrow.pos = (self.screenWidth / 2, self.screenHeight / 2)
        self.countrow.refresh()
        self.countrow.update()
        
        # Add deco to deco group
        if len(self.deco) > 0:
            self.deco_group = pygame.sprite.RenderUpdates(self.deco)
            
        # Save group display as first display
        self.hex_displays[0] = self.elements
        self.hex_groups[0] = self.groups
         
        for j in range(6):
            # Empty elements bin and start again
            self.elements, self.groups = [], []
            # Create the element-level displays
            lettergroup = self.hex_letters[j]      # Add empty space element
            for i in range(self.nr_elements):
                e = Circle(color=color, radius=radius, text=lettergroup[i], textcolor=self.hex_textcolor, textsize=50, colorkey=(0, 0, 0))
                e.set_states(0, {"textsize":110, "radius":74, "color":(255, 255, 255) })
                e.set_states(1, {"textsize":160, "radius":100, "color":(255, 255, 255) })
                self.add_element(e)
                e.refresh()
                e.update(0)
    
            # Get groups and add them
            for i in range(self.nr_elements):
                self.add_group(i)
                       
            # Save element display and groups
            self.hex_displays[j + 1] = self.elements
            self.hex_groups[j + 1] = self.groups
            
            
                      
        # Groups for entering and leaving
        self.enter_group = pygame.sprite.RenderUpdates()
        self.leave_group = pygame.sprite.RenderUpdates()
        
        # Sounds
        self.sound_new_word = pygame.mixer.Sound("Feedbacks\P300\windaVinciSysStart.wav")
        self.sound_countdown = pygame.mixer.Sound("Feedbacks\P300\winSpaceDefault.wav")
        self.sound_invalid = pygame.mixer.Sound("Feedbacks\P300\winSpaceCritStop.wav")
        
        # Open file for logging data
        if self.datafilename != "":
            try: 
                self.datafile = open(self.datafilename, 'a')
            except IOError:
                print "Cannot open datafile"
                self.datafile = None
                self.on_quit()
        
        # Init other variables
        self.group_trigger = None           # Set trigger before each trial
        self.hex_level = 0
        self.current_word = 0           # Index of current word
        self.current_letter = 0         # Index of current letter
        self.pre_mode = self.PRE_WORD
        self.current_tick = 0
        self.invalid_trial = 0          # Set wheth
       
        
        
    
    def pre_trial(self):
        # Countdown,prepare 
        if self.pre_mode == self.PRE_WORD: self.new_word()
        elif self.pre_mode == self.PREP_TRIAL: self.prep_trial()
        elif self.pre_mode == self.PRE_COUNTDOWN: self.pre_countdown()
        else: self.wait()

    def new_word(self):
        # If we just started a new word: present it
        if self.hex_level == 0 and self.current_letter == 0 and self.word_time > 0:
          
            self.sound_new_word.play()
            self.current_tick += 1
            word = self.words[self.current_word]
            font = pygame.font.Font(None, self.textsize)
            next_word_image = font.render("Next word: " + word, True, self.textcolor);
            next_word_rect = next_word_image.get_rect(center=(self.screenWidth / 2, self.screenHeight / 2))
            # Paint it
            self.screen.blit(self.all_background, self.all_background_rect)
            pygame.display.flip()
            self.screen.blit(self.all_background, self.all_background_rect)
            self.screen.blit(next_word_image, next_word_rect)
            pygame.display.flip()
            pygame.time.wait(self.word_time_ms)

            self.pre_mode = self.PREP_TRIAL
            self.current_tick = 0
            self.current_countdown = 0
            
        else:
            self.pre_mode = self.PREP_TRIAL
            self.current_tick = 0
            self.current_countdown = 0
    
    
    
    def pre_countdown(self):
        """
        Some countdown sounds - initialise display
        """    
        # only enter if we need a countdown for this hex level 
        if self.hex_countdown[self.hex_level] > 0:     
            if self.current_countdown == 0:
                self.screen.blit(self.all_background, self.all_background_rect)
                self.all_elements_group.draw(self.screen)
                if len(self.deco) > 0: self.deco_group.draw(self.screen)
                pygame.display.flip()       
            self.current_countdown += 1
            if self.current_countdown == self.hex_countdown[self.hex_level]:
                self.pre_mode = self.PRE_WAIT
            self.sound_countdown.play()
            pygame.time.wait(1000)
        else: self.pre_mode = self.PRE_WAIT
    
    
    
                
                    
    def wait(self):
        
        self.screen.blit(self.all_background, self.all_background_rect)
        self.all_elements_group.draw(self.screen)
        if len(self.deco) > 0: self.deco_group.draw(self.screen)
        pygame.display.flip()
        if self.hex_level == 0:
            self.send_parallel(self.START_TRIAL_LV1)
        else: self.send_parallel(self.START_TRIAL_LV2)
        pygame.time.wait(1000)
        self.state_finished = True


    def prep_trial(self):
        
        # reset variables
        self.invalid_trial = 0
        self.current_countdown = 0
                
        word = self.words[self.current_word]
        if self.hex_level == 0:
            self.elements = self.hex_displays[0] 
            self.groups = self.hex_groups[0] 
        else:
            # Second level: search the right hex and prepare it
            # Check to which level-2 group current letter belongs 
            hex_nr = None
            letter = word[self.current_letter]
            for i in range(len(self.hex_letters)):
                if letter in self.hex_letters[i]:
                    hex_nr = i
                    break
            self.elements = self.hex_displays[hex_nr + 1]  
            self.groups = self.hex_groups[hex_nr + 1]   

        self.textrow.text = word # Set new word
        self.textrow.highlight = [self.current_letter]  # highlight current letter
        self.textrow.refresh()
        # Delete old flash sequence & make new random sequence 
        self.tactile_sequence = []
        
        # Experimental sequences
        if self.tactile_stimulus:
            for i in range(self.nr_tactile_sequences):
                random_tactile_sequence(self, min_dist=self.min_dist, repetition=True)
        # Determine target group  set triggers & get a copy of that list
        self.group_trigger = self.hex_group_triggers[self.hex_level][:]
        # Determine target hex
        letter = self.words[self.current_word][self.current_letter]
        hex_nr, target = 0, 0
        target = np.random.randint(1,6)
        
        while letter not in self.hex_letters[hex_nr]: hex_nr += 1
        if self.hex_level == 0:  
            target = hex_nr 
        else:   # hex level 1
            target = self.hex_letters[hex_nr].index(letter)
            
        # Modify trigget of target by adding a value
        self.group_trigger[target] += self.trigger_target_add   
        # Step to next pre_mode
        self.pre_mode = self.PRE_COUNTDOWN
        # For logfile , if new trial is started
        if self.hex_level == 0: self.datalines = []
        # Flash count
        self.flashcount = 0
        
        

    def pre_stimulus(self):
        # Control eye tracker
        
        if self.invalid_trial == 0 and (self.stim_state == self.STIM_START_FLASH) and self.group_trigger is not None:
            #print "GROUP TRIGGER"
            self.send_parallel(self.group_trigger[self.tactile_sequence[self.current_tactile]])
            self.log_data()


   
   
    def log_data(self):
        """
        Structure of logfile
        Word Letter TrialNr Speller Fix_condition Trigger Time targetx targety currentx currenty Duration Invalid(-> added at flush)
        """
        #print self.et.x, self.et.y, self.et.duration
        if self.state == self.STIM_START_FLASH:
            word = self.words[self.current_word]
            items = []
            items.append(word)
            items.append(word[self.current_letter])
            items.append(str(self.trial_nr))
            items.append("hex")
            items.append("target")
            items.append(str(self.tactile_sequence[self.current_tactile]))
            items.append(str(pygame.time.get_ticks()))
            
            line = "\t".join(items)
            self.datalines.append(line) 
    
    def flush_data(self):
        # Writes the data into the data logfile
        for line in self.datalines:
            line2 = line + "\t" + str(self.flashcount) + "\t" + str(self.invalid_trial) + "\n"
            if self.datafile is not None:
                try: self.datafile.write(line2)
                except IOError:
                    self.logger.warn("Could not write to datafile")
            #print line2
        
    
        
    def post_trial(self):
      
        # Give 1 second time
        
        pygame.time.delay(1000)
        self.state_finished = True
        self.log_data()
        #flush_data()
        self.on_stop()




    
            
    def after_mainloop(self):
        
        self.color = None
        self.text = None
        self.start_pos = None
        self.end_pos = None
        self.et_targetxy = None
        self.datalines = None
        self.enter_group = None
        self.leave_group = None
        self.sound_new_word = None
        self.sound_countdown = None
        self.sound_invalid = None
        self.hex_displays = None
        self.hex_groups = None
        self.textrow = None
        self.countrow = None
        
        
 

        
    

        
    def hex_enter(self):
        # Erase and draw on both fore and back screen using flip
        self.screen.blit(self.background, self.background_rect)
        self.deco_group.draw(self.screen)
        pygame.display.flip()
        self.screen.blit(self.background, self.background_rect)
        self.deco_group.draw(self.screen)
        # Makes the hex elements enter the screen one for one
        self.enter_group.empty()
        for i in range(self.nr_elements):
            self.enter_group.add(self.elements[i])
            self.enter_group.update(0)
            self.enter_group.draw(self.screen)
            pygame.display.flip()
            pygame.time.wait(self.hex_time)
        # Draw latest version also onto back screen
        self.deco_group.draw(self.screen)
        self.enter_group.draw(self.screen)
        
    def hex_leave(self, reverse=False):
        """
        Makes the hex elements leave the screen one for one
        If reverse=True then they leave in reverse order 
        """
        self.leave_group.empty()
        self.leave_group.add(self.elements)
        for i in range(self.nr_elements):
            if reverse: ind = self.nr_elements - i - 1
            else: ind = i
            self.leave_group.remove(self.elements[ind])
            self.leave_group.update(0)
            self.screen.blit(self.background, self.background_rect)
            self.leave_group.draw(self.screen)
            self.deco_group.draw(self.screen)
            pygame.display.flip()
            pygame.time.wait(self.hex_time)

    def animate_letters(self, t, reverse=False):
        " Moves the letters from the chosen hex to their destinations "
        " Set elements to 'empty' states and get letter positions "
        " 'reverse' performs the reverse motion pattern "
        " Get current position "
        p = t / float(self.animation_time)     # Position parameters
        font = pygame.font.Font(None, int(round(self.end_size * p + self.start_size * (1 - p))))
        # paint everything 
        self.screen.blit(self.background, self.background_rect)
        self.all_elements_group.draw(self.screen)
        if len(self.deco) > 0: self.deco_group.draw(self.screen)
        for i in range(len(self.text)):
            xs, ys = self.start_pos[i]
            xe, ye = self.end_pos[i]
            x, y = xe * p + xs * (1 - p), ye * p + ys * (1 - p)
            surf = font.render(self.text[i], False, self.color)
            rect = surf.get_rect(center=(x, y))
            self.screen.blit(surf, rect)
        pygame.display.flip()
   
        
   


        
    
if __name__ == "__main__":
    a = P300HexTactile()
    a.on_init()
    a.on_play()
    a.on_stop()
    
    a.on_quit()
    sys.exit()
