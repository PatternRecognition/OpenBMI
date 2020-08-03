#!/usr/bin/env python
# -*- coding: utf-8 -*-

# HexoNavi/HexoNavi.py -
# Copyright (C) 2009  Marton Danoczy
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

"""Hex-o-Navi BCI Feedback."""

import random
import sys
import os
import math
import time
import struct
from socket import socket, AF_INET, SOCK_DGRAM
from lib.P300VisualElement.Textrow import Textrow

import pygame
from stuff import HexagonalGrid, Avatar, col_trans, generate_random_permutations

from FeedbackBase.MainloopFeedback import MainloopFeedback

    
class states:
    initializing = 0
    stimulating = 1
    pause_before_move = 2
    moving = 3
    waiting = 4
    
class markers:
    INIT_FEEDBACK = 200
    GAME_STATUS_PLAY = 210
    GAME_STATUS_PAUSE = 211
    GAME_OVER = 254


class HexoNavi(MainloopFeedback):
    #Some code stolen from FeedbackCursorArrow.py, thanks Basti & Simon!

    def init(self):
        """
        Initializes the variables and stuff, but not pygame itself.
        """
        self.logger.debug("HexoNavi init")
        self.send_parallel(markers.INIT_FEEDBACK)
        
        self.FPS = 25
        
        self.vest_host = "localhost"
        self.vest_port = 55555

        # For data logging (-> the data file is opened in pre_mainloop)
        self.datafilename = "c:\data\bbciRaw\responses"
        self.datafile = None

        
        self.fullscreen = False
        self.screenPos = [-1279, 0, 1280, 1024]        
        #self.screenPos = [100, 100, 800, 600]        
        self.avatar_color = (0,0,128)
        
        self.move_duration = 75
        
        self.level_file = "level_balanced2.txt" 
        
        self.num_stimuli = 6       #how many sides does a hexagon have? ha!
        self.num_rounds = 10       #how many times 6 stimuli are presented
        self.num_draws_with_repetitions_pre = 6
        self.num_draws_with_repetitions_post = 6
        self.stimulus_onset_asynchrony_in_frames = 8 #duration of stimulus + pause
        self.stimulus_duration_in_frames = 5 #duration of one stimulus in frames = 200 msec
        self.pause_after_trial_in_frames = 0 #125
        self.show_coordinates = False
 
        self.use_horizontal_tactors = True
        self._horizontal_tactors = [(45,49),(43,31),(41,29),(38,26),(37,25),(47,51)] #horizontal
        self._vertical_tactors   = [(58,46),(59,35),(34,32),(27,14),( 1,13),(11,12)] #vertical
        
        self.do_visual_stimulation = True
        self.do_tactile_stimulation = True
        self.return_to_start = True
        
        
    def pre_mainloop(self):
        self.logger.debug("pre_mainloop")
        self.state = states.initializing
        
        self.grid = HexagonalGrid(self.level_file)
        self.avatar_position = self.grid.start
        if self.return_to_start:
            self.going_back = False

        os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (self.screenPos[0], self.screenPos[1])        
        pygame.init()
        pygame.display.set_caption('Hex-o-Navi Feedback')
        if self.fullscreen:
            self.screen = pygame.display.set_mode((self.screenPos[2], self.screenPos[3]), pygame.FULLSCREEN)
        else:
            self.screen = pygame.display.set_mode((self.screenPos[2], self.screenPos[3]), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()

        #create sprites group
        self.sprites = pygame.sprite.RenderUpdates()
        self.sprites.add(Avatar(self.avatar_color))
        
        self.udp_socket = socket(AF_INET,SOCK_DGRAM)
        
        self.init_graphics()
        self.state = states.waiting

    def post_mainloop(self):
        self.logger.debug("post_mainloop")
        self.send_parallel(markers.GAME_OVER)
        pygame.quit()


    def tick(self):
        self.process_pygame_events()
        pygame.time.wait(10)
        self.elapsed = self.clock.tick(self.FPS)


    def pause_before_move_tick(self):
        self._pausing_since = self._pausing_since + 1
        if self._pausing_since >= self.pause_after_trial_in_frames:
            del self._pausing_since
            self.command_move()
        

    def play_tick(self):
        """
        Decides in which state the feedback currently is and calls the appropriate
        tick method.
        """
        if self.state == states.waiting:
            pass
        elif self.state == states.stimulating:
            self.stimulating_tick()
        elif self.state == states.moving:
            self.moving_tick()
        elif self.state == states.pause_before_move:
            self.pause_before_move_tick()
        else:
            raise "problem!"

    def moving_tick(self):
        self._moving_since = self._moving_since + 1
        
        if self._moving_since < self.move_duration:
            a = float(self._moving_since)/float(self.move_duration)
            self.move_avatar(a*self._moving_to[0] + (1-a)*self._moving_from[0],
                             a*self._moving_to[1] + (1-a)*self._moving_from[1])
        else:
            self.state = states.waiting
            self.move_avatar(*self._moving_to)      #to avoid rounding issues
            del self._moving_to                     #clean up a bit
            del self._moving_from
            del self._moving_since
            if self.return_to_start:
                if self.going_back and self.avatar_position == self.grid.start:
                    self.state = states.waiting
                else:
                    if not self.going_back and self.avatar_position == self.grid.goal:
                        self.last_position = None
                        self.going_back = True
                    self.command_stimulate()
            else:
                if self.avatar_position == self.grid.goal:
                    self.state = states.waiting
                else:
                    self.command_stimulate()

    def stimulating_tick(self):
        if self._stim_since == 0:
            stim = self._stimuli[self._stim_index]
            self.send_stimulus(stim)
            self.send_direction_marker(stim)
        self._stim_since = self._stim_since + 1
        if self._stim_since == self.stimulus_duration_in_frames:
            self.stop_stimuli()
        if self._stim_since == self.stimulus_onset_asynchrony_in_frames:
            self.stop_stimuli(even_visual=True)
            self._stim_since = 0
            self._stim_index = self._stim_index + 1
            if self._stim_index == len(self._stimuli):
                del self._stim_index
                del self._stim_since
                self.command_pause_before_move()
                
    def command_move(self, destination=None):
        if destination is None:   #find which is the only way to go
            walkable = self.grid.get_walkable_hexagons(*self.avatar_position)
            last_set = set([self.last_position])
            destination = walkable - last_set # we don't wanna go back
            if len(destination) != 1:
                raise RuntimeError("More than one way to go!")
            destination = destination.pop()
        self.state = states.moving
        self.last_position = int(self.avatar_position[0]), int(self.avatar_position[1]) 
        self._moving_to    = destination
        self._moving_from  = self.avatar_position
        self._moving_since = 0
    
    def command_pause_before_move(self):
        self._pausing_since = 0
        self.state = states.pause_before_move
        self.prompt_count()
        self.flush_data()
        self.init_graphics()
    
    def command_stimulate(self):
        self.state = states.stimulating
        self._stimuli = generate_random_permutations(range(6), self.num_rounds, 
                                                    self.num_draws_with_repetitions_pre,
                                                    self.num_draws_with_repetitions_post)
        self._stim_index = 0
        self._stim_since = 0
            
    def send_stimulus(self,stim):
        """Move some little arrow to one side of the current hexagon"""
                
        if self.do_tactile_stimulation:
            num_tactors = 64
            if self.use_horizontal_tactors:
                tact = self._horizontal_tactors[stim]
            else:
                tact = self._vertical_tactors[stim]
            msg = [0] * num_tactors
            try:
                for t in tact:
                    msg[t] = -1
            except TypeError: #tact is not iterable
                msg[tact] = -1
                    
            packet = struct.pack("l" * num_tactors, *msg)    
            self.udp_socket.sendto(packet, (self.vest_host,self.vest_port))
        
        if self.do_visual_stimulation:
            pos = self.grid.hexagonal2carthesian(*self.avatar_position)
            self.stim_sprites.clear(self.screen, self.grid.surface)
            self.stim_sprites.empty()
            self.stim_sprites.add(self.grid.stim_sprites[stim])
            self.stim_sprites.update(pos)
            rectlist = self.stim_sprites.draw(self.screen)
            rectlist = rectlist + self.sprites.draw(self.screen)
            pygame.display.update(rectlist)
            
    def send_direction_marker(self, stim):
        accessible = self.grid.get_walkable_hexagons(*self.avatar_position)
        last_set = set([self.last_position])
        destination = accessible - last_set # we don't wanna go back
        dir = self.grid.get_accessible_hexagons(*self.avatar_position)[stim]

        if len(destination) == 1 and destination.pop() == dir:
            marker = 21 + stim
        else:
            marker = 1 + stim
        self.send_parallel(marker)
        
    def stop_stimuli(self, even_visual=False): 
        
        if self.do_tactile_stimulation:
            str_zero = '\x00\x00\x00\x00'
            msg = str_zero * 64
            self.udp_socket.sendto(msg, (self.vest_host,self.vest_port))

        if even_visual and self.do_visual_stimulation:
            pos = self.grid.hexagonal2carthesian(*self.avatar_position)
            rectlist = self.stim_sprites.draw(self.screen)
            self.stim_sprites.clear(self.screen, self.grid.surface)
            rectlist = rectlist + self.sprites.draw(self.screen)
            pygame.display.update(rectlist)
            
    def move_avatar(self,a,b):
        self.avatar_position = a,b
        self.sprites.clear(self.screen, self.grid.surface)
        self.sprites.update(self.grid.hexagonal2carthesian(*self.avatar_position))
        rectlist = self.sprites.draw(self.screen)
        pygame.display.update(rectlist)

    def on_control_event(self, data):
        #self.logger.debug("on_control_event: %s" % str(data))
        self.f = data["data"][ - 1]    
    
    def prompt_count(self):
        background = pygame.Surface( (200,133) ) 
        background.fill((0,0,0))
        background_rect = background.get_rect(center = (self.screenPos[2]/2,self.screenPos[3]/2) )
        
        # Add count row (where count is entered by participant)
        countrow = Textrow(text="", textsize=60, color=(150, 150, 255), size=(100, 60), edgecolor=(255, 255, 255), antialias=True, colorkey=(0, 0, 0))
        countrow.pos = (self.screenPos[2] / 2, self.screenPos[3] / 2)
        countrow.refresh()
        countrow.update()

        pygame.event.clear()
        text, ready = "", False
        while not ready:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    k = event.key
                    if k == pygame.K_BACKSPACE:
                        if len(text) > 0: text = text[0: - 1]   # Delete last number
                    elif len(text) < 2:
                        if k == pygame.K_0: text = text + "0"
                        elif k == pygame.K_1: text = text + "1"
                        elif k == pygame.K_2: text = text + "2"
                        elif k == pygame.K_3: text = text + "3"
                        elif k == pygame.K_4: text = text + "4"
                        elif k == pygame.K_5: text = text + "5"
                        elif k == pygame.K_6: text = text + "6"
                        elif k == pygame.K_7: text = text + "7"
                        elif k == pygame.K_8: text = text + "8"
                        elif k == pygame.K_9: text = text + "9"
                    elif k == pygame.K_RETURN: ready = True
            countrow.text = text
            countrow.refresh()
            countrow.update(0)
            self.screen.blit(background, background_rect)
            self.screen.blit(countrow.image, countrow.rect)
            pygame.display.flip()
            pygame.time.wait(100)
        self.flashcount = int(text)
        self.send_parallel(100 + self.flashcount)

    def flush_data(self):
        # Writes the data into the data logfile
        line = str(self.flashcount) + "\n"
        if self.datafile is not None:
            try: self.datafile.write(line)
            except IOError:
                self.logger.warn("Could not write to datafile")
 
    def init_graphics(self):
        """
        Initialize the surfaces and fonts depending on the screen size.
        """

        self.screen = pygame.display.get_surface()
        self.size = min(self.screen.get_height(), self.screen.get_width())
        
        #where to put the level
        box = self.screen.get_rect().inflate(-20,-150)   #could be smaller later
        self.grid.scale(self.screen, box, self.show_coordinates)
        
        #sprite group for the stimuli
        self.stim_sprites = pygame.sprite.RenderUpdates()
        for s in self.grid.stim_sprites:
            self.stim_sprites.add(s)
            
        self.screen.blit(self.grid.surface,self.screen.get_rect())
        self.sprites.draw(self.screen)
        pygame.display.update()
        #if self.state == states.initializing:
        #    self.move_avatar(*self.grid.carthesian2hexagonal(*self.avatar_spawn_location))
        #else:
        self.move_avatar(*self.avatar_position) #don't move it, simply update
            
    def process_pygame_events(self):
        """
        Process the the pygame event queue and react on VIDEORESIZE.
        """
        for event in pygame.event.get():
            if event.type == pygame.VIDEORESIZE:
                self.screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE) 
                self.init_graphics()
            elif event.type == pygame.QUIT:
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.unicode == u" ":
                    self.command_move(self.grid.start)
                    self.last_position = None
                    self.send_parallel(markers.GAME_STATUS_PLAY)
                elif event.unicode == u"1": self.send_stimulus(0)
                elif event.unicode == u"2": self.send_stimulus(1)
                elif event.unicode == u"3": self.send_stimulus(2)
                elif event.unicode == u"4": self.send_stimulus(3)
                elif event.unicode == u"5": self.send_stimulus(4)
                elif event.unicode == u"6": self.send_stimulus(5)
                elif event.unicode == u"d" : step = 0.1
                time.sleep(.2)
                self.stop_stimuli(even_visual=True)

if __name__ == '__main__':
    print "Starting in standalone mode"
    fb = HexoNavi(None)
    fb.on_init()
    fb.on_play()

