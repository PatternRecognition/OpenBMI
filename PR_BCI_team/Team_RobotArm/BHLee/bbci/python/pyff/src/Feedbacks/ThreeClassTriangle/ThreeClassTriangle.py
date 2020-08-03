#!/usr/bin/env python

# ThreeClassTriangle.py -
# Copyright (C) 2009  Bastian Venthur
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
import random

import pygame

from FeedbackBase.PygameFeedback import PygameFeedback


STATE_COUNTDOWN = 1
STATE_CUE = 2
STATE_TRIAL = 3
STATE_AFTER_TRIAL = 4
STATE_GAME_OVER = 5
STATE_TESTMODE = 6
STATE_SHORTBREAK = 7

TRIGGER_TARGET = [1, 2, 3]
TRIGGER_CORRECT = [11, 12, 13]
TRIGGER_INCORRECT = [21, 22, 23]
TRIGGER_START_EXP = 100
TRIGGER_END_EXP = 101


class ThreeClassTriangle(PygameFeedback):
    
    def init(self):
        PygameFeedback.init(self)
        # Raw values as sent by the BCI
        self.raw_values = [0.0, 0.5, 1.0]
        # Values after calculations (bias, gain, etc.)
        self.values = [0.0, 0.0, 0.0]
        # bias and gain for each class (only for absolute control)
        self.bias = [0.0, 0.0, 0.0]
        self.gain = [1.0, 1.0, 1.0]
        # What to paint, polygons, lines, ellipses and circles, can be freely 
        # mixed
        self.polygon = True
        self.lines = False
        self.ellipses = True
        self.circles = False
        #
        self.backgroundColor = [0x00, 0x00, 0x21]
        self.color = [0x00, 0x00, 0x75]
        self.color2 = [0x00, 0x00, 0xcc]
        self.fontsize = 100
        self.caption = "3 Class Triangle"
        # Size of the target
        self.target_size = 50
        self.target_color = [0xcc, 0x00, 0x66]
        # Index of current target
        self.current_target = 0
        # Normal trialmode vs endless trial
        self.testmode = False
        # In this mode we only show targets nothing else
        self.calibrationmode = False
        self.state = STATE_TESTMODE
        # Trial configuration
        self.nr_of_trials = 30
        self.trial_duration = 5
        self.after_trial_duration = 2
        self.cue_duration = 1
        self.countdown_duration = 4
        # short break configuration
        self.short_break_after = 25
        self.short_break_duration = 15
        # Relative or Absolute control
        self.relative_control = True
        # seconds to reach the target if a class always wins.
        self.relative_speed = 1
        # Time elapsed in the current trial
        self.current_state_duration = 0
        self.current_trial = 0
        # After how many trials to show the points
        self.show_stats_after_trials = 10
        # Random seed to make random stuff reproducible
        self.random_seed = 1234
        # wins vs loses
        self.win_lose = [0, 0]
        # Positions of the values on the screen
        self.positions = [[0,0], [0,0], [0,0]]
        # Corners of the triangle
        self.triangle = [[0,0], [0,0], [0,0]]
        self.triangle_center = [0,0]
        # Angles of the ellipsis
        self.e_angles = [150, 30, -90]
        # Height of the ellipsis
        self.e_height = 50
        # Linesize 0 = fill
        self.size = 0

    
    def init_graphics(self):
        PygameFeedback.init_graphics(self)
        self.calculate_triangle_corners()

    
    def pre_mainloop(self):
        PygameFeedback.pre_mainloop(self)
        self.send_parallel(TRIGGER_START_EXP)
        if not self.testmode:
            self.state = STATE_COUNTDOWN
        else:
            self.state = STATE_TESTMODE
        random.seed(self.random_seed)
        self.current_trial = 0
        self.values = [0,0,0]
        self.clock.tick()

        
    def post_mainloop(self):
        PygameFeedback.post_mainloop(self)
        self.send_parallel(TRIGGER_END_EXP)
        
    
    def tick(self):
        PygameFeedback.tick(self)
        self.current_state_duration += self.elapsed / 1000.
        
    
    def play_tick(self):
        PygameFeedback.play_tick(self)
        # Trial about to end:
        if self.state == STATE_TRIAL and self.current_state_duration >= self.trial_duration:
            # Check if subject hit the target
            index = self.positions.index(max(self.positions))
            if index == self.current_target:
                self.win_lose[0] += 1
                self.send_parallel(TRIGGER_CORRECT[self.current_target])
            else:
                self.win_lose[1] += 1
                self.send_parallel(TRIGGER_INCORRECT[index])
            self.logger.debug(str(self.win_lose))
            if self.current_trial >= self.nr_of_trials:
                self.state = STATE_GAME_OVER
            else:
                self.state = STATE_AFTER_TRIAL
            self.current_state_duration = 0
        # After Trial about to end:
        elif self.state == STATE_AFTER_TRIAL and self.current_state_duration >= self.after_trial_duration:
            if self.current_trial % self.short_break_after == 0:
                self.state = STATE_SHORTBREAK
            else:
                self.state = STATE_CUE
                # Prepare next trial
                self.current_trial += 1
                self.current_target = random.choice(range(3))
                self.send_parallel(TRIGGER_TARGET[self.current_target])
            self.current_state_duration = 0
        # Shortbreak about to end
        elif self.state == STATE_SHORTBREAK and self.current_state_duration >= self.short_break_duration:
            self.state = STATE_CUE
            self.current_state_duration = 0
            # Prepare next trial
            self.current_trial += 1
            self.current_target = random.choice(range(3))
            self.send_parallel(TRIGGER_TARGET[self.current_target])
        # Cue about to end:
        elif self.state == STATE_CUE and self.current_state_duration >= self.cue_duration:
            self.state = STATE_TRIAL
            self.current_state_duration = 0
            self.values = [0,0,0]
        # Countdown about to end:
        elif self.state == STATE_COUNTDOWN and self.current_state_duration >= self.countdown_duration:
            self.state = STATE_CUE
            self.current_state_duration = 0
            # Prepare next trial
            self.current_trial += 1
            self.current_target = random.choice(range(3))
            self.send_parallel(TRIGGER_TARGET[self.current_target])
        
        # Blank the screen
        self.screen.fill(self.backgroundColor)
        if self.state == STATE_COUNTDOWN:
            self.countdown_tick()
        elif self.state == STATE_CUE:
            self.cue_tick()
        elif self.state == STATE_TRIAL:
            self.calculate_values()
            self.trial_tick()
        elif self.state == STATE_SHORTBREAK:
            self.shortbreak_tick()
        elif self.state == STATE_AFTER_TRIAL:
            self.after_trial_tick()
        elif self.state == STATE_GAME_OVER:
            self.gameover_tick()
        elif self.state == STATE_TESTMODE:
            self.calculate_values()
            self.testmode_tick()
        # Update the screen
        pygame.display.flip()
        


    def pause_tick(self):
        PygameFeedback.pause_tick(self)
        
    
    def on_control_event(self, data):
        self.raw_values = data.get("cl_output", [0,0,0])
        
    
    def calculate_values(self):
        """Add bias and gain to the real values."""
        if self.relative_control:
            # index of the maximum value
            max_i = self.raw_values.index(max(self.raw_values))
            for i in range(3):
                speed = self.relative_speed if max_i == i else -self.relative_speed
                speed = 1./speed
                self.values[i] = self.values[i] + speed * self.elapsed / 1000.
        else:
            # absolute control
            for i in range(3):
                self.values[i] = self.raw_values[i] * self.gain[i] + self.bias[i]
        # cap the values
        for i in range(3):
            if self.values[i] > 1: self.values[i] = 1
            if self.values[i] < 0: self.values[i] = 0
            
    
    def trial_tick(self):
        """Paint a tick of a trial."""
        self.calculate_positions()
        if not self.calibrationmode:
            self.draw_data()
        self.draw_background()
        self.draw_target()
    
    
    def testmode_tick(self):
        self.trial_tick()
    
    
    def cue_tick(self):
        """Paint background and target."""
        self.draw_background()
        self.draw_target()


    def after_trial_tick(self):
        """Paint background and target."""
        self.draw_background()
        if not self.calibrationmode and self.current_trial % self.show_stats_after_trials == 0:
            self.print_center(self.screen, "Correct: %i / Incorrect: %i" % (self.win_lose[0], self.win_lose[1]), self.target_color, self.fontsize) 
        
    
    def countdown_tick(self):
        t = int(math.ceil(self.countdown_duration - self.current_state_duration))
        self.print_center(self.screen, str(t), self.target_color, self.fontsize)
        
    
    def shortbreak_tick(self):
        t = int(math.ceil(self.short_break_duration - self.current_state_duration))
        self.print_center(self.screen, str(t), self.target_color, self.fontsize)


    def gameover_tick(self):
        self.print_center(self.screen, "Game Over", self.target_color, self.fontsize)
    

    def draw_background(self):
        """Draw the Y and the empty targets."""
        # Paint the background
        #pygame.draw.aaline(self.screen, self.color2, self.triangle[0], self.triangle[1])
        #pygame.draw.aaline(self.screen, self.color2, self.triangle[1], self.triangle[2])
        #pygame.draw.aaline(self.screen, self.color2, self.triangle[2], self.triangle[0])
        for i in self.triangle:
            pygame.draw.aaline(self.screen, self.color2, self.triangle_center, i)
            pygame.draw.circle(self.screen, self.color2, i, self.target_size)


    def draw_target(self):
        """Highlight active target."""
        pygame.draw.circle(self.screen, 
                           self.target_color, 
                           self.triangle[self.current_target], 
                           self.target_size)


    def draw_data(self):
        """Draw the EEG values."""
        # Plot the data
        if self.polygon:
            pygame.draw.polygon(self.screen, self.color, self.positions, self.size)
        for i in range(3):
            if self.lines:
                self.roundline(self.screen, 
                               self.color2, 
                               self.triangle_center, 
                               self.positions[i], 
                               self.size/2)
            if self.ellipses:
                self.ellipse(self.screen, 
                             self.color2,
                             self.middle(self.triangle_center, self.positions[i]), 
                             self.distance(self.triangle_center, self.positions[i]), 
                             self.e_height, 
                             self.e_angles[i], 
                             self.size)
            if self.circles:
                pygame.draw.circle(self.screen, self.color2, self.positions[i], self.size/2, 0)

    
    def calculate_triangle_corners(self):
        """Calculate the positions of the corners according to the current 
        screen size.
        """
        # w, h of the bounding box of the triangle
        w, h = self.screen.get_width()-self.target_size*2, self.screen.get_height()-self.target_size*2
        # Make the bounding box fit into the screen
        h_1 = math.sqrt(1**2 - (1/2.0)**2)
        w = h/h_1 if w*h_1 > h else w
        h = w*h_1 if h/h_1 > w else h
        # Put the triangle in the bounding box
        self.triangle = [[0, 0], [w, 0], [w/2, h]]
        # Move the bounding Box to the center if necessary
        self.triangle = [[x+self.screen.get_width()/2 - w/2, y+self.screen.get_height()/2 - h/2] for x, y in self.triangle]
        # distance: side to center
        center = (w / 6.0) * math.sqrt(3)
        self.triangle_center = [self.triangle[0][0] + w/2, self.triangle[0][1] + center]
        
    
    def calculate_positions(self):
        """Calculate the screen positions of the EEG data."""
        for i in range(3):
            dx = abs(self.triangle[i][0] - self.triangle_center[0])
            dy = abs(self.triangle[i][1] - self.triangle_center[1])
            # distance from center
            posx = dx * self.values[i]
            posy = dy * self.values[i]
            # absolute position
            if i == 0:
                self.positions[i][0] = round(self.triangle_center[0] - posx)
                self.positions[i][1] = round(self.triangle_center[1] - posy)
            elif i == 1:
                self.positions[i][0] = self.triangle_center[0] + posx
                self.positions[i][1] = self.triangle_center[1] - posy
            else:
                self.positions[i][0] = self.triangle_center[0]
                self.positions[i][1] = self.triangle_center[1] + posy
        
    
    def roundline(self, srf, color, start, end, radius=1):
        """Paints a line on the screen."""
        dx = end[0]-start[0]
        dy = end[1]-start[1]
        distance = max(abs(dx), abs(dy))
        for i in range(distance):
            x = int( start[0]+float(i)/distance*dx)
            y = int( start[1]+float(i)/distance*dy)
            pygame.draw.circle(srf, color, (x, y), radius)
            

    def ellipse(self, srf, color, pos, width, height, angle, size):
        """Draws a elipsis fitting in the box (widhth, height), roatate it and 
        center the result at the given position.
        """
        if width < 1 or height < 1:
            return
        surface = pygame.Surface((width, height), pygame.SRCALPHA)
        try:
            pygame.draw.ellipse(surface, color, [0,0,width,height], size)
            surface = pygame.transform.rotate(surface, angle)
            srf.blit(surface, [pos[0]-surface.get_rect().width/2,pos[1]-surface.get_rect().height/2])
        except:
            pass
        
    
    def print_center(self, srf, text, color, size):
        """Print the string on the surface."""
        font = pygame.font.Font(None, size)
        surface = font.render(text, 1, color)
        srf.blit(surface, surface.get_rect(center=srf.get_rect().center))
        
    
    def distance(self, p1, p2):
        """Return the euklidian distance between two points."""
        dx = abs(p1[0] - p2[0])
        dy = abs(p1[1] - p2[1])
        return math.sqrt(dx**2 + dy**2)
    
    
    def middle(self, p1, p2):
        """Return the middle of the given two points."""
        p = [0, 0]
        for i in range(2):
            p[i] = p1[i] + abs(p1[i]-p2[i])/2 if p1[i] < p2[i] else p2[i] + abs(p1[i]-p2[i])/2
        return p


if __name__ == "__main__":
    fb = ThreeClassTriangle()
    fb.on_init()
    fb.on_play()
