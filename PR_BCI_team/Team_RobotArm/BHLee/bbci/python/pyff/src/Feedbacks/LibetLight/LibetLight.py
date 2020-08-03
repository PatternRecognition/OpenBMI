# LibetLight.py -
# Copyright (C) 2010-2011  Rafael Schultze-Kraft
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

import random, sys, pygame
sys.path.append('/home/raf/CNS/bbci/pyff/src') 
from FeedbackBase.PygameFeedback import PygameFeedback

class LibetLight(PygameFeedback):

    def init(self):
        # initialize variables
        PygameFeedback.init(self)
        self.screenSize = [1280, 1024]
        self.caption = "LibetLight"
        self.winning_score = 50                 # score needed to win
        self.bulb_size = 80                     # size of 'light bulb'
        self.light_on_color = [220, 220, 0]     # color of light when on
        self.light_off_color = [0, 0, 0]        # color of light when off
        self.player_color = [0, 0, 160]         # color of player label 
        self.comp_color = [160, 0, 0]           # color of computer label
        self.score_color = [0, 0, 0]            # color of scores
        self.background_color = [127, 127, 127]
        self.player_score = 0                   # starting scores
        self.comp_score = 0
        self.pause_after_switch = 3000          # pausing time after switching on/off light
        self.player_up_color = [124, 252, 0, 80]
        self.comp_up_color = [250, 0, 0, 80]
        self.both_up_color = [0, 0, 250, 80]
        self.who_scored = None
        self.player_label = 'YOU'
        self.comp_label = 'COMP'
        self.label_fontsize = 64
        self.score_fontsize = 128
        self.player_label_pos = [50, 200]
        self.comp_label_pos = [self.screenSize[1]-15, 200]
        self.player_score_pos = [50, 250]
        self.comp_score_pos = (self.screenSize[1], 250)
        self.classifier_active = False
        self.classifier_active_time = 1000
        self.begin_classifier_time = 0
       
        # three possible events        
        self.event1_marker = 21                 # 21 = keypress
        self.event2_marker = 22                 # 22 = classifier
        self.event3_marker = 23                 # 23 = classifier + keypress (at the same time)
                
    def pre_mainloop(self):
        PygameFeedback.pre_mainloop(self)
        self.light_on = False                   # start with light off
        self.clock.tick()
        self.present_stimulus()
        pygame.time.wait(2000)

    def post_mainloop(self):
        PygameFeedback.post_mainloop(self)
        
    def tick(self):
        # mainloop
        self.eval_input()                            # evaluate classifier output
        if self.player_score == self.winning_score \
        or self.comp_score == self.winning_score:
            self.on_stop()                           # stop if winning score reached
        else:
            self.present_stimulus()

    def present_stimulus(self):
        self.screen.fill(self.background_color)
        self.draw_light()
        self.draw_labels()
        self.draw_scores()
        pygame.display.flip()

    def draw_light(self):
        if self.light_on == True:
            pygame.draw.circle(self.screen, self.light_on_color, self.screen.get_rect().center, self.bulb_size, 0)
	else:
            pygame.draw.circle(self.screen, self.light_off_color, self.screen.get_rect().center, self.bulb_size, 50)

    def draw_labels(self):
        font = pygame.font.Font(None, self.label_fontsize)
        subj_text = font.render(self.player_label, True, self.player_color)
        comp_text = font.render(self.comp_label, True, self.comp_color)
        self.screen.blit(subj_text, self.player_label_pos)
        self.screen.blit(comp_text, self.comp_label_pos)

    def draw_scores(self):
        font = pygame.font.Font(None, self.score_fontsize)
        subj_score_text = font.render("%02d" % self.player_score, True, self.score_color)
        comp_score_text = font.render("%02d" % self.comp_score, True, self.score_color)
        self.screen.blit(subj_score_text, self.player_score_pos)
        self.screen.blit(comp_score_text, self.comp_score_pos)
	
    def post_event(self):
        self.screen.fill(self.background_color)
        self.draw_labels()
        
        if self.who_scored == 'player scored':
            alpha_surface = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA, 16)
            alpha_surface.fill(self.player_up_color)
            self.screen.blit(alpha_surface, (0,0))
            self.draw_light()
            font = pygame.font.Font(None, 180)
            score_text = font.render("%02d" % self.player_score, True, self.score_color)
            self.screen.blit(score_text, self.player_score_pos)
            font = pygame.font.Font(None, self.score_fontsize)
            comp_score_text = font.render("%02d" % self.comp_score, True, self.score_color)
            self.screen.blit(comp_score_text, self.comp_score_pos)

        elif self.who_scored == 'comp scored':
            alpha_surface = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA, 16)
            alpha_surface.fill(self.comp_up_color)
            self.screen.blit(alpha_surface, (0,0))
            self.draw_light()
            font = pygame.font.Font(None, 180)
            score_text = font.render("%02d" % self.comp_score, True, self.score_color)
            self.screen.blit(score_text, self.comp_score_pos)
            font = pygame.font.Font(None, self.score_fontsize)
            player_score_text = font.render("%02d" % self.player_score, True, self.score_color)
            self.screen.blit(player_score_text, self.player_score_pos)

        elif self.who_scored == 'both':
            alpha_surface = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA, 16)
            alpha_surface.fill(self.both_up_color)
            self.screen.blit(alpha_surface, (0,0))
            self.draw_light()
            font = pygame.font.Font(None, 180)
            player_score_text = font.render("%02d" % self.player_score, True, self.score_color)
            self.screen.blit(player_score_text, self.player_score_pos)
            comp_score_text = font.render("%02d" % self.comp_score, True, self.score_color)
            self.screen.blit(comp_score_text, self.comp_score_pos)
            
        else:
            pass

        pygame.display.flip()
        self.on_pause()
        pygame.time.wait(self.pause_after_switch)
        self.on_play()        

    def eval_input(self):
        # 0 = no event
        # 1 = keypress
        # 2 = classifier
        # 3 = classifier + keypress

        if not self.classifier_active:
            
            if self._data['cl_output'] == 0:
            # no event
                pass

            elif self._data['cl_output'] == 1:
            # keypress    
                self.send_parallel(self.event1_marker)
                self.light_on = True
                self.player_score += 1
                self.who_scored = 'player scored'
                self.post_event()  
        
            elif self._data['cl_output'] == 2:
            # classifier
                self.send_parallel(self.event2_marker)
                self.begin_classifier_time = pygame.time.get_ticks()
                self.classifier_active = True
                self.light_on = True
                self.present_stimulus()
                           
            elif self._data['cl_output'] == 3:
            # classifier + keypress
                self.send_parallel(self.event3_marker)
                self.player_score += 1
                self.comp_score += 1
                self.who_scored = 'both'
                self.post_event()

            else:
                pass
        
        if self.classifier_active:
            
            if self._data['cl_output'] == 0:
            # no event
                pass

            elif self._data['cl_output'] == 1:
            # keypress    
                self.send_parallel(self.event1_marker)
                self.classifier_active = False
                self.light_on = False
                self.comp_score += 1
                self.who_scored = 'comp scored'
                self.post_event()  
        
            elif self._data['cl_output'] == 2:
            # classifier
                # self.send_parallel(self.event2_marker)
                self.begin_classifier_time = pygame.time.get_ticks()
                                           
            elif self._data['cl_output'] == 3:
            # classifier + keypress
                self.send_parallel(self.event3_marker)
                self.classifier_active = False
                self.light_on = False
                self.comp_score += 1
                self.who_scored = 'comp scored'
                self.post_event()  

            else:
                pass

            if pygame.time.get_ticks() - self.begin_classifier_time > self.classifier_active_time:
                self.classifier_active = False
                self.light_on = False
                self.present_stimulus()
            else:
                pass
        
if __name__ == "__main__":
   fb = LibetLight()
   fb.on_init()
   fb.on_play()
