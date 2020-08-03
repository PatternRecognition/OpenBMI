#!/usr/bin/env python

# HandWorkStation2.py - HandWorkStation Game with Online Feedback
# Copyright (C) 2011  Matthias Schultze-Kraft
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


import random
import os
import pygame
import time
from numpy import savetxt, linspace
from FeedbackBase.PygameFeedback import PygameFeedback




class HandWorkStation(PygameFeedback):
    """HandWorkStation Game"""


    #SCREEN_SIZE = (1920, 1080)
    SCREEN_SIZE = (1300, 900)
    X_MARGIN = SCREEN_SIZE[0]/14
    Y_MARGIN = SCREEN_SIZE[1]/20
    WORK_RANGE = (X_MARGIN, SCREEN_SIZE[0]-X_MARGIN)
    WORK_HEIGHT = SCREEN_SIZE[1] - Y_MARGIN
    COLOR_KEYS = ('b', 'g', 'r', 'y')

    TRIG_STOP = 255

    TRIG_START_NORMAL = 10
    TRIG_END_NORMAL = 11
    TRIG_START_STRESS = 20
    TRIG_END_STRESS = 21

    #TRIG_SCREW_NORMAL = 31
    #TRIG_SCREW_STRESS = 32
    #TRIG_MOVE_BARREL = 21
    #TRIG_SELECT_COLOR = 22
    #TRIG_RESET_ONE = 23
    #TRIG_RESET_ALL = 24

    TRIG_CORRECT = 111
    TRIG_WRONG = 112
    TRIG_MISSED = 113



    def init(self):

        # initializations
        pygame.mixer.pre_init(frequency=22050, size=-16, channels=8, buffer=512)    # must be called before pygame.init() in order to ensure correct (on-time) playback of sounds
        PygameFeedback.init(self)
        self.random_seed = 1234

        # screen settings
        self.FPS = 250
        self.caption = "HandWorkStation"
        self.screenSize = self.SCREEN_SIZE
        self.background_color = [191, 191, 191]
        self.work_color = [63, 63, 63]
        self.screenPos = [0, 0, self.SCREEN_SIZE[0], self.SCREEN_SIZE[1]]
        self.fullscreen = False

        # secondary experiment parameters
        self.MIN_SCREW_DISTANCE = self.SCREEN_SIZE[0]/16
        self.MAX_SCREW_DISTANCE = self.SCREEN_SIZE[0]/4
        self.NR_PXLS_TO_SHIFT = 1

        # MODE
        self.MODE = 3

        # NEW PARAMETERS IN VERSION 2.0
        self.BLOCK_SPEED = .99
        self.DIST_NORMAL = .5
        self.DIST_STRESS_H_LIST = [.535, .49, .445, .4, .355, .31, .265]
        self.DIST_STRESS_L_LIST = [.52, .5, .48, .46, .44, .42, .4]


    def pre_mainloop(self):

        if self.MODE == 1:
            # OFFLINE
            #self.NR_BLOCKS = 12
            #self.BLOCK_DURATION = [120.0]*self.NR_BLOCKS
            self.NR_BLOCKS = 4
            self.BLOCK_DURATION = [30.0]*self.NR_BLOCKS
            self.BLOCK_SPEED = [self.BLOCK_SPEED]*self.NR_BLOCKS
            self.BLOCK_STRESS = [False, True]*int(self.NR_BLOCKS/2)
            self.DIST_STRESS_H = [self.DIST_STRESS_H_LIST[5]]*self.NR_BLOCKS
            self.DIST_STRESS_L = [self.DIST_STRESS_L_LIST[5]]*self.NR_BLOCKS
            self.save_log = False
        elif self.MODE == 2:
            # SEMI-ONLINE
            #self.NR_BLOCKS = 12
            #self.BLOCK_DURATION = [120, 120, 90, 140, 160, 90, 90, 160, 140, 90, 120, 120]
            self.NR_BLOCKS = 4
            self.BLOCK_DURATION = [30.0]*self.NR_BLOCKS
            self.BLOCK_SPEED = [self.BLOCK_SPEED]*self.NR_BLOCKS
            self.BLOCK_STRESS = [False, True]*int(self.NR_BLOCKS/2)
            self.DIST_STRESS_H = [self.DIST_STRESS_H_LIST[5]]*self.NR_BLOCKS
            self.DIST_STRESS_L = [self.DIST_STRESS_L_LIST[5]]*self.NR_BLOCKS
            self.save_log = False
        elif self.MODE == 3:
            # ONLINE
            self.NR_BLOCKS = 1
            self.BLOCK_DURATION = [120]
            self.BLOCK_SPEED = [self.BLOCK_SPEED]
            self.BLOCK_STRESS = [True]
            self.DIST_STRESS_H = [self.DIST_STRESS_H_LIST[3]]
            self.DIST_STRESS_L = [self.DIST_STRESS_H_LIST[3]]
            self.save_log = False

        self.screws = list()
        self.log = []

        PygameFeedback.pre_mainloop(self)
        self.clock.tick()
        self.load_sounds()  # load sounds

        # initialize counters
        self.tick_count = -self.FPS*2   # initial tick count
        self.missed = 0
        self.correct = 0
        self.wrong = 0
        self.block_nr = 1

        # screw settings
        self.update_speed()
        self.next_interval = self.NORMAL_ISI
        self.stress_screw_countdown = 3
        self.next_short = False
        self.last_screw_pos = self.SCREEN_SIZE[0]/2

        # flags
        self.barrel_pressed = False
        self.color_pressed = False

        # experiment start trigger
        self.timer = Timer()
        self.send_parallel(self.TRIG_START_NORMAL)
        #self.log.append(self.TRIG_START_NORMAL)


    def post_mainloop(self):

        PygameFeedback.post_mainloop(self)

        self.send_parallel(self.TRIG_STOP)

        if self.save_log:
            savetxt(self.log_fname, self.log, delimiter="\n")



    def init_graphics(self):

        PygameFeedback.init_graphics(self)

        self.load_images()
        self.screen = pygame.display.get_surface()

        self.background = pygame.Surface((self.SCREEN_SIZE[0], self.SCREEN_SIZE[1])).convert()
        self.background_rect = self.background.get_rect()
        self.background.fill(self.background_color)

        self.work_control_left = pygame.Surface((self.WORK_RANGE[0], self.SCREEN_SIZE[1])).convert()
        self.work_control_left_rect = pygame.Rect((0, 0), (self.work_control_left.get_size()))
        self.work_control_left.fill(self.work_color)

        self.work_control_right = pygame.Surface((self.WORK_RANGE[0], self.SCREEN_SIZE[1])).convert()
        self.work_control_right_rect = pygame.Rect((self.WORK_RANGE[1], 0), (self.work_control_left.get_size()))
        self.work_control_right.fill(self.work_color)

        self.arrow_rect = [None, None, None, None, None, None]
        self.arrow_rect[0] = pygame.Rect((5, self.arrow_size[1]), self.arrow_size)
        self.work_control_left.blit(self.arrows[0], self.arrow_rect[0])
        self.arrow_rect[1] = pygame.Rect((5, self.arrow_size[1]*3), self.arrow_size)
        self.work_control_left.blit(self.arrows[1], self.arrow_rect[1])
        self.arrow_rect[2] = pygame.Rect((5, self.arrow_size[1]), self.arrow_size)
        self.work_control_right.blit(self.arrows[2], self.arrow_rect[2])
        self.arrow_rect[3] = pygame.Rect((5, self.arrow_size[1]*3), self.arrow_size)
        self.work_control_right.blit(self.arrows[3], self.arrow_rect[3])

        self.arrow_rect[4] = pygame.Rect((5, self.arrow_size[1]*5), self.arrow_size)
        self.work_control_left.blit(self.arrows[4], self.arrow_rect[4])
        self.arrow_rect[5] = pygame.Rect((5, self.arrow_size[1]*5), self.arrow_size)
        self.work_control_right.blit(self.arrows[5], self.arrow_rect[5])

        self.arrow_rect[2] = pygame.Rect((self.WORK_RANGE[1]+5, self.arrow_size[1]), self.arrow_size)
        self.arrow_rect[3] = pygame.Rect((self.WORK_RANGE[1]+5, self.arrow_size[1]*3), self.arrow_size)
        self.arrow_rect[5] = pygame.Rect((self.WORK_RANGE[1]+5, self.arrow_size[1]*5), self.arrow_size)

        self.background.blit(self.work_control_left, self.work_control_left_rect)
        self.background.blit(self.work_control_right, self.work_control_right_rect)

        self.work_ground = pygame.Surface((self.SCREEN_SIZE[0]-self.X_MARGIN*2, self.Y_MARGIN)).convert()
        self.work_ground_rect = pygame.Rect((self.X_MARGIN, self.WORK_HEIGHT), self.work_ground.get_size())
        self.work_ground.fill(self.work_color)
        self.background.blit(self.work_ground, self.work_ground_rect)

        self.barrel = Barrel(self)

        self.draw_all()



    def draw_all(self):

        self.screen.blit(self.background, self.background_rect)

        for ii in range(len(self.screws)):
                self.screen.blit(self.screws[ii].image[0], self.screws[ii].rect_top)
                self.screen.blit(self.screws[ii].image[1], self.screws[ii].rect_mid)
                self.screen.blit(self.screws[ii].image[2], self.screws[ii].rect_bottom)

        self.screen.blit(self.barrel_image, self.barrel.rect)
        for p in range(3):
            if not self.barrel.color[p]==None:
                self.screen.blit(self.barrel_labels[self.barrel.color[p]][p], self.barrel.rect)

        pygame.display.update()



    def load_images(self):

        path = os.path.dirname(globals()["__file__"])
        colors4 = ('blue', 'green', 'red', 'yellow')
        colors6 = ('blue', 'green', 'red', 'yellow', 'black_1', 'black_3')
        label_positions = ('bottom', 'mid', 'top')

        #self.barrel_size = (self.SCREEN_SIZE[1]/12, self.SCREEN_SIZE[1]/10)
        #self.screw_size = (self.SCREEN_SIZE[1]/15, self.SCREEN_SIZE[1]/15)
        self.barrel_size = (self.SCREEN_SIZE[1]/8, self.SCREEN_SIZE[1]/7)
        self.screw_size = (self.SCREEN_SIZE[1]/10, self.SCREEN_SIZE[1]/10)
        self.arrow_size = (self.X_MARGIN-10, self.SCREEN_SIZE[1]/7)

        self.barrel_image = pygame.image.load(os.path.join(path, 'Barrel.gif')).convert()
        self.barrel_image = pygame.transform.scale(self.barrel_image, self.barrel_size)
        self.barrel_image.set_colorkey(self.barrel_image.get_at((0, 0)))

        self.screw_images = [None, None, None, None]
        for c, color in enumerate(colors4):
            self.screw_images[c] = pygame.image.load(os.path.join(path, 'screw_' + color + '.png')).convert()
            self.screw_images[c] = pygame.transform.scale(self.screw_images[c], self.screw_size)
            self.screw_images[c].set_colorkey(self.screw_images[c].get_at((0, 0)))

        self.barrel_labels = [[None, None, None], [None, None, None], [None, None, None], [None, None, None]]
        for c, color in enumerate(colors4):
            for p, pos in enumerate(label_positions):
                self.barrel_labels[c][p] = pygame.image.load(os.path.join(path, 'barrel_' + pos + '_' + color + '.gif')).convert()
                self.barrel_labels[c][p] = pygame.transform.scale(self.barrel_labels[c][p], self.barrel_size)
                self.barrel_labels[c][p].set_colorkey(self.barrel_labels[c][p].get_at((0, 0)))

        self.arrows= [None, None, None, None, None, None]
        for c, color in enumerate(colors6):
            self.arrows[c] = pygame.image.load(os.path.join(path, 'arrow_' + color + '.png')).convert()
            self.arrows[c] = pygame.transform.scale(self.arrows[c], self.arrow_size)
            self.arrows[c].set_colorkey(self.arrows[c].get_at((0, 0)))



    def load_sounds(self):
        path = os.path.dirname(globals()["__file__"])
        self.sound_click   = pygame.mixer.Sound(os.path.join(path, 'click.wav'))



    def process_pygame_event(self, event):

        PygameFeedback.process_pygame_event(self,event)

        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self.on_stop()



    def on_control_event(self, data):
        if data.has_key("speed"):
            fb = int(data["speed"])
            self.DIST_STRESS_H[0] = self.DIST_STRESS_H_LIST[fb-1]
            self.DIST_STRESS_L[0] = self.DIST_STRESS_L_LIST[fb-1]
            self.update_speed()



    def play_tick(self):

        self.tick_count += 1

        # set screw distances via online feedback
        #if self.MODE==3:
        #    self.DIST_STRESS_H = self.DIST_STRESS_H_LIST[self.feedback-1]
        #    self.DIST_STRESS_L = self.DIST_STRESS_L_LIST[self.feedback-1]

        # switch block or end game
        if self.timer.elapsed > self.BLOCK_DURATION[self.block_nr-1]:
            # send stop trigger
            if self.BLOCK_STRESS[self.block_nr-1]:
                self.send_parallel(self.TRIG_END_STRESS)
                #self.log.append(self.TRIG_START_STRESS)
            else:
                self.send_parallel(self.TRIG_END_NORMAL)
                #self.log.append(self.TRIG_START_NORMAL)
            # print calibration results
#            print "BLOCK: " + str(self.block_nr) + "/" + str(self.NR_BLOCKS)
#            if self.MODE == 2:
#                self.log.append(self.BLOCK_SPEED[self.block_nr-1])
#                print "SPEED: " + str(self.BLOCK_SPEED[self.block_nr-1])
#            elif self.MODE == 3:
#                self.log.append(self.DIST_STRESS_H[self.block_nr-1])
#                self.log.append(self.DIST_STRESS_L[self.block_nr-1])
#                print "DISTANCES: " + str(self.DIST_STRESS_H[self.block_nr-1]) + ", " + str(self.DIST_STRESS_L[self.block_nr-1])
#            print "ERROR RATE: " + str((self.wrong+self.missed*1.0)/(self.correct+self.wrong+self.missed))
            self.block_nr += 1
            # end of game?
            if self.block_nr > self.NR_BLOCKS:
                self.on_stop()
                return
            # reset timer
            self.timer.reset()
            # update speed
            self.update_speed()
            # log and reset counter
            self.log.append((self.wrong+self.missed*1.0)/(self.correct+self.wrong+self.missed))
            self.missed = 0
            self.correct = 0
            self.wrong = 0
            # send start trigger
            if self.BLOCK_STRESS[self.block_nr-1]:
                self.send_parallel(self.TRIG_START_STRESS)
                #self.log.append(self.TRIG_START_STRESS)
            else:
                self.send_parallel(self.TRIG_START_NORMAL)
                #self.log.append(self.TRIG_START_NORMAL)

        # alternative color selection via keyboard
        if self.keypressed:
            key = self.lastkey_unicode
            self.keypressed = False
            if key in self.COLOR_KEYS:
                self.barrel_add_color(self.color_keys.index(key))

        # get events from touch screen
        if not pygame.mouse.get_pressed()[0]:
            self.barrel_pressed = False
            self.color_pressed = False
        if self.barrel_pressed:
            mouse_pos = pygame.mouse.get_pos()
            self.barrel.rect.centerx = mouse_pos[0]
        elif pygame.mouse.get_pressed()[0]:
            mouse_pos = pygame.mouse.get_pos()
            if self.barrel.rect.collidepoint(mouse_pos):
                self.barrel_pressed = True
                self.barrel.rect.centerx = mouse_pos[0]
                #self.send_parallel(self.TRIG_MOVE_BARREL)
            elif not self.color_pressed:
                if self.arrow_rect[0].collidepoint(mouse_pos):
                    self.sound_click.play()
                    self.barrel.add_color(0)
                    self.color_pressed = True
                    #self.send_parallel(self.TRIG_SELECT_COLOR)
                elif self.arrow_rect[1].collidepoint(mouse_pos):
                    self.sound_click.play()
                    self.barrel.add_color(1)
                    self.color_pressed = True
                    #self.send_parallel(self.TRIG_SELECT_COLOR)
                elif self.arrow_rect[2].collidepoint(mouse_pos):
                    self.sound_click.play()
                    self.barrel.add_color(2)
                    self.color_pressed = True
                    #self.send_parallel(self.TRIG_SELECT_COLOR)
                elif self.arrow_rect[3].collidepoint(mouse_pos):
                    self.sound_click.play()
                    self.barrel.add_color(3)
                    self.color_pressed = True
                    #self.send_parallel(self.TRIG_SELECT_COLOR)
                elif self.arrow_rect[4].collidepoint(mouse_pos):
                    self.sound_click.play()
                    self.barrel.remove_color()
                    self.color_pressed = True
                    #self.send_parallel(self.TRIG_RESET_ONE)
                elif self.arrow_rect[5].collidepoint(mouse_pos):
                    self.sound_click.play()
                    self.barrel.reset()
                    self.color_pressed = True
                    #self.send_parallel(self.TRIG_RESET_ALL)

        # constraint position of barrel
        if self.barrel.rect.left < self.WORK_RANGE[0]:
            self.barrel.rect.left = self.WORK_RANGE[0]
        if self.barrel.rect.right > self.WORK_RANGE[1]:
            self.barrel.rect.right = self.WORK_RANGE[1]

        # create new screw set
        if self.tick_count >= self.next_interval:
            self.tick_count = 0
            #print str(self.timer.elapsed)
            self.screws.append(Screw(self))
            # send corresponding screw release trigger
            if self.next_short:
                #self.send_parallel(self.TRIG_SCREW_STRESS)
                self.next_interval = self.NORMAL_ISI
                self.next_short = False
            #else:
                #self.send_parallel(self.TRIG_SCREW_NORMAL)
            # check for next short interval
            if self.BLOCK_STRESS[self.block_nr-1]:
                self.stress_screw_countdown -= 1
                if self.stress_screw_countdown == 1:
                    self.next_interval = self.STRESS_ISI
                    self.next_short = True
                    self.stress_screw_countdown = random.randint(*(2, 4))

        # update screws vertical position
        elif self.tick_count%self.LOO_INTERVAL:
            for ii in range(len(self.screws)):
                self.screws[ii].update()

        # check for collisions of bottom screw
        b = self.barrel.rect
        b = pygame.Rect((b.left+b.width/2., b.top+b.height/3.), (1, 1))
        for ii in range(len(self.screws)):
            # collision with bottom of screen
            if self.screws[ii].rect_bottom.bottom > self.WORK_HEIGHT:
                self.screws.pop(ii)
                self.missed += 1
                self.send_parallel(self.TRIG_MISSED)
                #self.log.append(self.timer.elapsed)
                #self.log.append(self.TRIG_MISSED)
                break
            # collision with barrel
            elif b.colliderect(self.screws[ii].rect_bottom):
                # matching colors
                if self.screws[ii].color[0]==self.barrel.color[0] and \
                   self.screws[ii].color[1]==self.barrel.color[1] and \
                   self.screws[ii].color[2]==self.barrel.color[2]:
                    self.screws.pop(ii)
                    self.correct += 1
                    self.barrel.reset()
                    self.send_parallel(self.TRIG_CORRECT)
                    #self.log.append(self.timer.elapsed)
                    #self.log.append(self.TRIG_CORRECT)
                    break
                # non-matching colors
                else:
                    self.screws.pop(ii)
                    self.wrong += 1
                    self.barrel.reset()
                    self.send_parallel(self.TRIG_WRONG)
                    #self.log.append(self.timer.elapsed)
                    #self.log.append(self.TRIG_WRONG)
                    break

        # update screen
        self.draw_all()



    def update_speed(self):

        speed = self.BLOCK_SPEED[self.block_nr-1]
        full_ISI = self.WORK_HEIGHT * 1/speed * 1/self.NR_PXLS_TO_SHIFT

        self.LOO_INTERVAL = round(1/(1-speed))
        if not self.BLOCK_STRESS[self.block_nr-1]:
            self.NORMAL_ISI = full_ISI * self.DIST_NORMAL
        else:
            self.NORMAL_ISI = full_ISI * self.DIST_STRESS_L[self.block_nr-1]
            self.STRESS_ISI = full_ISI * self.DIST_STRESS_H[self.block_nr-1]



class Barrel(HandWorkStation):

    def __init__(self, parent):
        self.parent = parent
        self.color = [None, None, None]
        self.rect = self.parent.barrel_image.get_rect()
        self.rect.centerx = self.parent.screenSize[0]/2
        self.rect.bottom = self.parent.WORK_HEIGHT
        self.next_label = 0

    def add_color(self, color):
        if not self.next_label == 3:
            self.color[self.next_label] = color
            self.next_label += 1

    def remove_color(self):
        if not self.next_label == 0:
            self.next_label -= 1
            self.color[self.next_label] = None

    def reset(self):
        self.color = [None, None, None]
        self.next_label = 0





class Screw(HandWorkStation):

    def __init__(self, parent):
        self.parent = parent
        # draw three unequal! random colors
        colors = [None, None, None]
        colors[0] = random.randint(*(0, 3))
        while 1:
            colors[1] = random.randint(*(0, 3))
            if colors[1]!=colors[0]:
                break
        while 1:
            colors[2] = random.randint(*(0, 3))
            if colors[2]!=colors[1] and colors[2]!=colors[0]:
                break
        self.color = colors
        # assign images
        self.image = (self.parent.screw_images[colors[2]],\
                      self.parent.screw_images[colors[1]],\
                      self.parent.screw_images[colors[0]])
        # assign x position according to constraints
        a, b = self.parent.WORK_RANGE
        w, h = self.parent.screw_size
        while 1:
            x = random.randint(-self.parent.MAX_SCREW_DISTANCE, self.parent.MAX_SCREW_DISTANCE)
            if abs(x)>self.parent.MIN_SCREW_DISTANCE:
                if self.parent.last_screw_pos+x>a and self.parent.last_screw_pos+x<b-w:
                    x, y = self.parent.last_screw_pos+x, -h
                    self.parent.last_screw_pos = x
                    break
        self.rect_top = pygame.Rect((x, y*2), (w, h))
        self.rect_mid = pygame.Rect((x, y*1.5), (w, h))
        self.rect_bottom = pygame.Rect((x, y), (w, h))

    def update(self):
        self.rect_top.y += self.parent.NR_PXLS_TO_SHIFT
        self.rect_mid.y += self.parent.NR_PXLS_TO_SHIFT
        self.rect_bottom.y += self.parent.NR_PXLS_TO_SHIFT





class Timer(object):

    def __init__(self):
        self.__start = time.time()

    def reset(self):
        self.__start = time.time()

    def elapsed(self):
        return time.time() - self.__start
    elapsed = property(elapsed)




if __name__ == "__main__":
    fb = HandWorkStation2()
    fb.on_init()
    fb.on_play()
















