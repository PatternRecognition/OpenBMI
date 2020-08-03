#!/usr/bin/env python

# SMR_NeuroFeedback.py -
# Copyright (C) 2010  Marton Danoczy
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


import math

import pygame

from FeedbackBase.PygameFeedback import PygameFeedback


STATE_RUNNING = 1
STATE_TESTMODE = 2

TRIGGER_START_EXP = 100
TRIGGER_END_EXP = 101


class SMR_NeuroFeedback(PygameFeedback):
    
    def init(self):
        PygameFeedback.init(self)
        # Raw values as sent by the BCI
        self.values = [0.2, 0.25]
        # What to paint, polygons, lines, ellipses and circles, can be freely 
        # mixed
        #
        self.backgroundColor = [0x40, 0x40, 0x40]
        self.rectangleColor  = [0x00, 0x80, 0x00]
        self.triangleColor   = [0x00, 0xc0, 0x00]
        self.bar_width_fraction = 0.1
        self.fontsize = 100
        self.caption = "SMR Neurofeedback training"
        # Normal trialmode vs endless trial
        self.testmode = False
        self.state = STATE_TESTMODE
    
    def init_graphics(self):
        PygameFeedback.init_graphics(self)
    
    def pre_mainloop(self):
        PygameFeedback.pre_mainloop(self)
        self.send_parallel(TRIGGER_START_EXP)
        if not self.testmode:
            self.state = STATE_RUNNING
        else:
            self.state = STATE_TESTMODE
        self.clock.tick()

        
    def post_mainloop(self):
        PygameFeedback.post_mainloop(self)
        self.send_parallel(TRIGGER_END_EXP)
        
    
    def tick(self):
        PygameFeedback.tick(self)
       
    
    def play_tick(self):
        PygameFeedback.play_tick(self)
        
        # Blank the screen
        self.screen.fill(self.backgroundColor)

        w, h = self.screen.get_width(), self.screen.get_height()
        hrec, htri = self.values
        
        mid = 0.5*w
        l = mid*(1.0 - self.bar_width_fraction)
        r = mid*(1.0 + self.bar_width_fraction)
        yrec = h*(1.0-hrec)

        rectangle = pygame.Rect(l, yrec, w*self.bar_width_fraction, h-yrec)
        pygame.draw.rect(self.screen, self.rectangleColor, rectangle)

        if hrec>0.5:
            ytri = yrec - yrec*htri
        else:
            ytri = yrec - (h*hrec)*htri

        triangle = [ [l,yrec], [r,yrec], [mid,ytri] ]
        pygame.draw.polygon(self.screen, self.triangleColor, triangle)

        
        # Update the screen
        pygame.display.flip()
        


    def pause_tick(self):
        PygameFeedback.pause_tick(self)
        
    
    def on_control_event(self, data):
        self.values = data.get("cl_output", [0,0])
        
   
    
    def draw_background(self):
        """Draw the Y and the empty targets."""
        # Paint the background
        #pygame.draw.aaline(self.screen, self.color2, self.triangle[0], self.triangle[1])
        #pygame.draw.aaline(self.screen, self.color2, self.triangle[1], self.triangle[2])
        #pygame.draw.aaline(self.screen, self.color2, self.triangle[2], self.triangle[0])
        #for i in self.triangle:
        #    pygame.draw.aaline(self.screen, self.color2, self.triangle_center, i)
        #    pygame.draw.circle(self.screen, self.color2, i, self.target_size)


    

    def print_center(self, srf, text, color, size):
        """Print the string on the surface."""
        font = pygame.font.Font(None, size)
        surface = font.render(text, 1, color)
        srf.blit(surface, surface.get_rect(center=srf.get_rect().center))
        
    

if __name__ == "__main__":
    fb = SMR_NeuroFeedback()
    fb.on_init()
    fb.on_play()
