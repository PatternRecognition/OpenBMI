'''
Created on 2011-12-5

@author: Administrator
'''
'''Feedbacks.VisualSpeller.CenterSpellerVE
# Copyright (C) 2010  "Nico Schmidt"
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

Created on Mar 31, 2010

@author: "Nico Schmidt"

'''
from PinyinSpellerBase import PinyinSpellerBase, animate, animate_sinusoid
from VEShapes import FilledTriangle, FilledHexagon,FilledHourglass,FilledCross
from VisionEgg.MoreStimuli import FilledCircle, Target2D
from VisionEgg.Text import Text
from lib.P300Layout.CircularLayout import CircularLayout
from VisionEgg.FlowControl import FunctionController



#import pygame
import numpy as NP
        

class PinyinSpellerVE(PinyinSpellerBase):
    '''
    classdocs
    '''
    
    def init(self):
        '''
        initialize parameters
        '''
        
        PinyinSpellerBase.init(self)
        
        self.letter_set = [['b','p','m','f','d','t','n'], \
                           ['l','g','k','h','j','q','x'], \
                           ['zh','ch','sh','r','z','c','s'], \
                           ['y','w','up','enter','del','<','>'], \
                           ['a','o','e','i','u','v','ai'], \
                           ['ei','ui','ao','ou','iu','ie','ve'], \
                           ['er','an','en','in','un','vn','ang'], \
                           ['eng','ing','ong','down','1','2','3']
                           ]
        ## sizes:
        self._nr_elements = 8
        self.letter_radius = 50
        self.speller_radius = 300
        self.font_size_level1 = 45         # letters in level 1
        self.font_size_level2 = 130         # letters in level 2
        self.feedbackbox_size = 200.0
        self.fixationpoint_size = 4.0
        self.font_size_feedback_ErrP = 300
        
        ## stimulus types:
        self.stimulustype_color = True
        self.shape_on=True
        
        self.stimulus_duration = 0.083   # 5 frames @60 Hz = 83ms flash
        self.interstimulus_duration = 0.1


        ## feedback type:
        self.feedback_show_shape = True
        self.feedback_show_shape_at_center =True
        
        ## level 2 appearance:
        self.level_2_symbols = True
        self.level_2_letter_colors = False
        self.level_2_animation = True
        self.backdoor_symbol = "^"
        #        self.backdoor_symbol = unichr(0x2191)
        
        ## colors:
        self.shape_color = (1.0, 1.0, 1.0)
##        self.stimuli_colors = [(1.0, 0.0, 0.0),
##                               (0.0, 1.0, 0.0),
##                               (0.0, 0.0, 1.0),
##                               (1.0, 1.0, 0.0),
##                               (1.0, 0.0, 1.0),
##                               (0.0, 1.0, 1.0)]
        self.stimuli_colors = [[0.0,0.0,1.0],
                               [0.0,0.53,0.006],
                               [1.0,0.0,0.0],
                               [1.0,1.0,0.0],
                               [0.86,0.0,0.86],
                               [0.,0.95,0.95],
                               [1.0,0.5,0.5],
                               [1.0,1.0,1.0]]
        
        self.letter_color = (.5, .5, .5)
        self.feedback_color = (0.9, 0.9, 0.9)
        self.fixationpoint_color = (1.0, 1.0, 1.0)
        self.feedback_ErrP_color = (0.7, 0.1, 0.1)
        
        ## register possible shapes:
        self.registered_shapes = {'circle':FilledCircle,
                                      'cross' : FilledCross,
                                  'hexagon':FilledHexagon,
                                  'hourglass':FilledHourglass,
                                  'triangle':FilledTriangle,
                                  'rectangle':Target2D}
        
        ## define shapes in the form ['name', {parameters}]:
        # where parameters can be VisionEgg parameters eg. 'radius', 'size', 'orientation' etc.
##        self.shapes = [['triangle',  {'size':200}],
##                       ['rectangle', {'size':(180., 50), 'orientation':45.}],
##                       ['rectangle', {'size':(180., 40), 'orientation':-45.}],
##                       ['triangle',  {'size':200, 'orientation':180.}],
##                       ['hourglass', {'size':100}],
##                       ['circle',    {'radius':90}]]
##            self.shapes = [
##            [ "triangle", {"color": [0.0,0.0,1.0], "innerColor":self.bg_color, "innerSize": 60, "size": 200}], 
##            [ "hourglass", {"color": [0.0,0.53,0.006], "size": 100 }], 
##            [ "cross", {"color": [1.0,0.0,0.0], "orientation": 45, "size": [30,180], "innerColor":self.bg_color}], 
##            [ "triangle", {"color": [1.0,1.0,0.0], "innerColor":self.bg_color, "innerSize": 60, "orientation": 180, "size": 200}], 
##            [ "hourglass", {"color": [0.86,0.0,0.86], "orientation": 90, "size": 100}], 
##            [ "cross", {"color": [0.95,0.95,0.95], "size": [30,180], "innerColor":self.bg_color }]
##        ]
        self.shapes = [
            [ "triangle", {"innerColor":self.bg_color, "innerSize": 60, "size": 150}], 
            [ "hourglass", {"size": 80 }], 
            [ "cross", {"orientation": 45, "size": [20,150], "innerColor":self.bg_color}], 
            [ "triangle", {"innerColor":self.bg_color, "innerSize": 60, "orientation": 180, "size": 150}], 
            [ "hourglass", {"orientation": 90, "size": 80}], 
            [ "cross", {"size": [20,150], "innerColor":self.bg_color }],
            ["triangle", {"innerColor":self.bg_color, "innerSize": 60, "orientation": 90, "size": 150}], 
            [ "hourglass", {"orientation": 135, "size": 80}]
        ]
    
    def prepare_mainloop(self):
        '''
        called in pre_mainloop of superclass.
        '''
        
        self.synchronized_countdown = False
        self.do_animation = False
        if self.synchronized_countdown:
            self.do_animation = False
        if not self.do_animation:
            self.level_2_animation = False
        ## init containers for VE elements:
        self._ve_shapes = []
        self._ve_letters = []
        
        self._letter_positions = []
        self._countdown_shape_positions = []
        self._countdown_letter_positions = []
        self._countdown_screen = False
        
        if self.stimulustype_color:            
            assert len(self.stimuli_colors)==self._nr_elements
            ## Added code Lovisa 26/10
            self.shape_color = self.stimuli_colors
            self.letter_color = (0.5, 0.5, 0.5)
            
        else:
            self.shape_color = [self.shape_color]*self._nr_elements
            #self.stimuli_colors = [self.shape_color]*self._nr_elements
            ## End of changes
        
        if not self.feedback_show_shape:
            self.feedback_show_shape_at_center = False

        if not self.shape_on:
            self.shapes=[['triangle',  {'size':200}],
                       ['triangle',  {'size':200}],
                       ['triangle',  {'size':200}],
                       ['triangle',  {'size':200}],
                       ['triangle',  {'size':200}],
                       ['triangle',  {'size':200}],
                       ['triangle',  {'size':200}],
                       ['triangle',  {'size':200}]
                       ]

            
    
    def init_screen_elements(self):
        '''
        overwrite this function in subclass.
        '''
        ## create shapes:
        self._nr_elements = 8
        if self.do_animation:
            for i in xrange(self._nr_elements):
                self._ve_shapes.append(self.registered_shapes[self.shapes[i][0]]( 
                                       position=self._centerPos,                                       #color=self.stimuli_colors[i],
                                       color=self.shape_color[i],
                                       on=False,
                                       **self.shapes[i][1]))

                       
            ## add letters of level 1:
            circle_layout = CircularLayout(nr_elements=self._nr_elements,
                                           radius=self.speller_radius,
                                           start=NP.pi/8.*7)
            circle_layout.positions.reverse()
            self._letter_layout = CircularLayout(nr_elements=self._nr_elements,
                                           radius=self.letter_radius,
                                           start=NP.pi/8.*7)
            self._letter_layout.positions.reverse()
            for i in xrange(self._nr_elements):
                # store countdown position:
                self._countdown_shape_positions.append((self._centerPos[0] + circle_layout.positions[i][0],
                                                        self._centerPos[1] + circle_layout.positions[i][1]))
                # put shape in container:
                self._ve_elements.append(self._ve_shapes[i])
                
                for j in xrange(len(self.letter_set[i])): # warning: self.letter_set must be at least of length self._nr_elements!!!
                    # store position:
                    self._letter_positions.append((self._letter_layout.positions[j][0] + self._centerPos[0],
                                                   self._letter_layout.positions[j][1] + self._centerPos[1]))
                    
                    # store countdown position:
                    self._countdown_letter_positions.append((self._letter_layout.positions[j][0] + self._countdown_shape_positions[-1][0],
                                                             self._letter_layout.positions[j][1] + self._countdown_shape_positions[-1][1]))
                    # add letter:
                    self._ve_letters.append(Text(position=self._letter_positions[-1],
                                                 text=self.letter_set[i][j],
                                                 font_size=self.font_size_level1,
                                                 color=self.letter_color,
                                                 anchor='center',
                                                 on=False))
            
            # add letters of level 2:
            for i in xrange(self._nr_elements):
                self._letter_positions.append(self._centerPos)
                self._countdown_letter_positions.append(self._countdown_shape_positions[i])
                self._ve_letters.append(Text(position=self._centerPos,
                                             text=" ",
                                             font_size=self.font_size_level2,
                                             color=(self.level_2_letter_colors and self.stimuli_colors[i] or self.letter_color),
                                             #color=(self.level_2_letter_colors and self.stimuli_colors[i]),
                                             anchor='center',
                                             on=False))
            # put letters in container:
            self._ve_elements.extend(self._ve_letters)
                    
            
            ## create feedback box:
            self._ve_feedback_box = Target2D(position=self._centerPos,
                                             size=(self.feedbackbox_size, self.feedbackbox_size),
                                             color=self.feedback_color,
                                             on=False)
            
            ## add feedback letters:
            self._ve_feedback_letters = []
            for i in xrange(self._nr_elements):
                self._ve_feedback_letters.append(Text(position=(self._letter_layout.positions[i][0]+self._centerPos[0],
                                                                self._letter_layout.positions[i][1]+self._centerPos[1]),
                                                      color=self.letter_color,
                                                      font_size=self.font_size_level1,
                                                      text=" ",
                                                      on=False,
                                                      anchor="center"))
            self._ve_feedback_letters.append(Text(position=self._centerPos,
                                                  color=self.letter_color,
                                                  font_size=self.font_size_level2,
                                                  text=" ",
                                                  anchor='center',
                                                  on=False))
            
            ## add feedback note (whether or not there was an ErrP detected):
            self._ve_feedback_ErrP = Text(position=self._centerPos,
                                          color=self.feedback_ErrP_color,
                                          text="X",
                                          font_size=self.font_size_feedback_ErrP,
                                          anchor='center',
                                          on=False)
                
            ## add fixation point:
            self._ve_fixationpoint = FilledCircle(radius=self.fixationpoint_size,
                                                  position=self._centerPos,
                                                  color=self.fixationpoint_color,
                                                  on=False)
        ##################### IF NOT DO ANIMATION #########################
        else:                    
            ## add letters of level 1:
            circle_layout = CircularLayout(nr_elements=self._nr_elements,
                                           radius=self.speller_radius,
                                           start=NP.pi/8.*9)
            self._letter_layout = CircularLayout(nr_elements=self._nr_elements,
                                           radius=self.letter_radius,
                                           start=NP.pi/8.*9)
            circle_layout.positions.reverse()
            self._letter_layout.positions.reverse()

            for i in xrange(self._nr_elements):
                self._ve_shapes.append(self.registered_shapes[self.shapes[i][0]]( 
                                       position=(self._centerPos[0] + circle_layout.positions[i][0],
                                                        self._centerPos[1] + circle_layout.positions[i][1]),
                                       #color=self.stimuli_colors[i],
                                       color=self.shape_color[i],
                                       on=False,
                                       **self.shapes[i][1]))



            for i in xrange(self._nr_elements):
                # store countdown position:
                self._countdown_shape_positions.append((self._centerPos[0] + circle_layout.positions[i][0],
                                                        self._centerPos[1] + circle_layout.positions[i][1]))
                # put shape in container:
                self._ve_elements.append(self._ve_shapes[i])
                
                for j in xrange(len(self.letter_set[i])): # warning: self.letter_set must be at least of length self._nr_elements!!!
                    # store position:
                    self._letter_positions.append((self._letter_layout.positions[j][0] + self._centerPos[0],
                                                   self._letter_layout.positions[j][1] + self._centerPos[1]))
                    
                    # store countdown position:

                    self._countdown_letter_positions.append((self._letter_layout.positions[j][0] + self._countdown_shape_positions[-1][0],
                                                             self._letter_layout.positions[j][1] + self._countdown_shape_positions[-1][1]))

                    # add letter:
                    self._ve_letters.append(Text(position=(self._letter_layout.positions[j][0] + self._centerPos[0]+ circle_layout.positions[i][0],
                                                   self._letter_layout.positions[j][1] + self._centerPos[1]+circle_layout.positions[i][1]),
                                                 text=self.letter_set[i][j],
                                                 font_size=self.font_size_level1,
                                                 color=self.letter_color,
                                                 anchor='center',
                                                 on=False))
            
            # add letters of level 2:
            for i in xrange(self._nr_elements):
                self._letter_positions.append((self._letter_layout.positions[i][0] + self._centerPos[0],
                                                   self._letter_layout.positions[i][1] + self._centerPos[1]))
                self._countdown_letter_positions.append((self._letter_layout.positions[i][0] + self._countdown_shape_positions[-1][0],
                                                             self._letter_layout.positions[i][1] + self._countdown_shape_positions[-1][1]))
                self._ve_letters.append(Text(position=(self._letter_layout.positions[i][0] + self._centerPos[0]+ circle_layout.positions[i][0],
                                                   self._letter_layout.positions[i][1] + self._centerPos[1]+circle_layout.positions[i][1]),
                                             text=" ",
                                             font_size=self.font_size_level2,
                                             color=(self.level_2_letter_colors and self.stimuli_colors[i] or self.letter_color),
                                             #color=(self.level_2_letter_colors and self.stimuli_colors[i]),
                                             anchor='center',
                                             on=False))
            # put letters in container:
            self._ve_elements.extend(self._ve_letters)
                    
            
            ## create feedback box:
            self._ve_feedback_box = Target2D(position=self._centerPos,
                                             size=(self.feedbackbox_size, self.feedbackbox_size),
                                             color=self.feedback_color,
                                             on=False)
            
            ## add feedback letters:
            self._ve_feedback_letters = []
            for i in xrange(self._nr_elements):
                self._ve_feedback_letters.append(Text(position=(self._letter_layout.positions[i][0]+self._centerPos[0],
                                                                self._letter_layout.positions[i][1]+self._centerPos[1]),
                                                      color=self.letter_color,
                                                      font_size=self.font_size_level1,
                                                      text=" ",
                                                      on=False,
                                                      anchor="center"))
            self._ve_feedback_letters.append(Text(position=self._centerPos,
                                                  color=self.letter_color,
                                                  font_size=self.font_size_level2,
                                                  text=" ",
                                                  anchor='center',
                                                  on=False))
            
            ## add feedback note (whether or not there was an ErrP detected):
            self._ve_feedback_ErrP = Text(position=self._centerPos,
                                          color=self.feedback_ErrP_color,
                                          text="X",
                                          font_size=self.font_size_feedback_ErrP,
                                          anchor='center',
                                          on=False)
                
            ## add fixation point:
            self._ve_fixationpoint = FilledCircle(radius=self.fixationpoint_size,
                                                  position=self._centerPos,
                                                  color=self.fixationpoint_color,
                                                  on=False)
            
        # put letters in container:
        self._ve_elements.append(self._ve_feedback_box)
        self._ve_elements.extend(self._ve_feedback_letters)
        self._ve_elements.append(self._ve_feedback_ErrP)
        self._ve_elements.append(self._ve_fixationpoint)
    
    
    def set_countdown_screen(self):
        '''
        set screen how it should look during countdown.
        '''
        if self._countdown_screen:
            return
    

        self._countdown_screen = True
        is_level1 = self._current_level==1

        ## turn on visible elements:
        for i in xrange(self._nr_elements): # shapes
            self._ve_shapes[i].set(on=is_level1 or self.level_2_symbols)

        for i in xrange(self._nr_letters): # level 1 letters
            self._ve_letters[i].set(on=is_level1)

        for i in xrange(len(self.letter_set[self._classified_element])): # level 2 letters
            self._ve_letters[self._nr_letters + i].set(on=not is_level1,
                                                       text=(is_level1 and " " or self.letter_set[self._classified_element][i]))
    
        self._ve_letters[self._nr_letters + self._nr_elements-1].set(on=not is_level1,
                                                       text=(is_level1 and " " or self.backdoor_symbol))
        
        ## move all elements with their letters to countdown position:


        def update(t):
            dt = t/self.animation_time
            for i in xrange(self._nr_elements):
                pos = animate_sinusoid(self._centerPos, self._countdown_shape_positions[i], dt)
                self._ve_shapes[i].set(position=pos) # shapes
                self._ve_letters[self._nr_letters + i].set(position=pos) # level 2 letters
            for i in xrange(self._nr_letters):
                pos = animate_sinusoid(self._letter_positions[i], self._countdown_letter_positions[i], dt)
                self._ve_letters[i].set(position=pos) # level 1 letters
        def update2(t):
            for i in xrange(self._nr_elements):
                self._ve_shapes[i].set(position=self._countdown_shape_positions[i])
                self._ve_letters[self._nr_letters + i].set(position=self._countdown_shape_positions[i])
            for i in xrange(self._nr_letters):
                self._ve_letters[i].set(position=self._countdown_letter_positions[i])

        if self.do_animation:     
            self._presentation.set(go_duration=(self.animation_time, 'seconds'))
            self._presentation.add_controller(None,None,FunctionController(during_go_func=update))
        else:
            self._presentation.set(go_duration=(self.interstimulus_duration,'seconds'))
            self._presentation.add_controller(None,None,FunctionController(during_go_func=update2)) 

        
        # send to screen:
        self._presentation.go()
        self._presentation.remove_controller(None,None,None)
        
        for i in xrange(self._nr_elements):
            self._ve_shapes[i].set(position=self._countdown_shape_positions[i])
            self._ve_letters[self._nr_letters + i].set(position=self._countdown_shape_positions[i])
        for i in xrange(self._nr_letters):
            self._ve_letters[i].set(position=self._countdown_letter_positions[i])

    
    def set_standard_screen(self):
        '''
        set screen elements to standard state.
        '''

        self._countdown_screen = False
        
        # move all elements with their letters to standard position:
        
        def update(t):
            dt = t/self.animation_time
            for i in xrange(self._nr_elements):
                pos = animate_sinusoid(self._countdown_shape_positions[i], self._centerPos, dt)
                self._ve_shapes[i].set(position=pos) # shapes
                self._ve_letters[self._nr_letters + i].set(position=pos) # level 2 letters
            for i in xrange(self._nr_letters):
                pos = animate_sinusoid(self._countdown_letter_positions[i], self._letter_positions[i], dt)
                self._ve_letters[i].set(position=pos) # level 1 letters
        
        if self.do_animation:
            self._presentation.set(go_duration=(self.animation_time, 'seconds'))
            self._presentation.add_controller(None,None,FunctionController(during_go_func=update))
        else:
            self._presentation.set(go_duration=(0,'seconds'))    

        # send to screen:
        self._presentation.go()
        self._presentation.remove_controller(None,None,None)
        
        for i in xrange(self._nr_elements):
            self._ve_shapes[i].set(position=self._centerPos, on=False)
            self._ve_letters[self._nr_letters + i].set(position=self._centerPos, on=False)
        for i in xrange(self._nr_letters):
            self._ve_letters[i].set(position=self._letter_positions[i], on=False)
            
        self._ve_countdown.set(on=False, text=" ")
    
    def stimulus(self, i_element, on=True):
        '''
        turn on/off the stimulus elements and turn off/on the normal elements.
        '''

        if self._current_level==1:
            self._ve_shapes[i_element].set(on=on)
            for i in xrange(len(self.letter_set[i_element])):
                self._ve_letters[(self._nr_elements-1) * i_element + i].set(on=on)
        else:
            self._ve_shapes[i_element].set(on=(on and self.level_2_symbols))
            if i_element < len(self.letter_set[self._classified_element]):
                self._ve_letters[self._nr_letters + i_element].set(on=on, text=(on and self.letter_set[self._classified_element][i_element] or " "))
            else:
                self._ve_letters[self._nr_letters + self._nr_elements-1].set(on=on, text=(on and self.backdoor_symbol or " "))
    
    
    def feedback(self):
        '''
        Show classified element / letter(s). 
        '''
        self._show_feedback(True)
            
        ## present:
        self._presentation.set(go_duration=(self.feedback_duration, 'seconds'))
        self._presentation.go()
        
        self._show_feedback(False)
    
    
    def _show_feedback(self, on=True):
        ## turn on/off feedback box:
        if not self.feedback_show_shape_at_center:
            self._ve_feedback_box.set(on=on)
        
        if self._current_level == 1:
            ## turn on/off center letter group:
            if not self.feedback_show_shape_at_center:
                for i in xrange(self._nr_elements):
                    self._ve_feedback_letters[i].set(on=on,
                                                     text=(i<self._nr_elements-1 and 
                                                           self.letter_set[self._classified_element][i] or
                                                           self.backdoor_symbol))
            if self.feedback_show_shape:
                if self.feedback_show_shape_at_center or not on:
                    pos = self._centerPos
                else:
                    pos = self._countdown_shape_positions[self._classified_element]
                    
                ## turn on/off selected element:
                self._ve_shapes[self._classified_element].set(on=on, position=pos)
                
                ## turn on/off letters of selected element:
                idx_start = self._classified_element*(self._nr_elements-1)
                idx_end = idx_start + self._nr_elements-1
                for i in xrange(idx_start, idx_end):
                    self._ve_letters[i].set(on=on, position=(on and
                                                             list(NP.add(pos, self._letter_layout.positions[i % (self._nr_elements-1)])) or
                                                             self._letter_positions[i]))
        else: ### level 2:
            ## check if backdoor classified:
            if self._classified_letter >= len(self.letter_set[self._classified_element]):
                text = self.backdoor_symbol
            else:
                text = self.letter_set[self._classified_element][self._classified_letter]
                
            ## turn on/off letter:
            if self.offline or not self.feedback_show_shape_at_center:
                self._ve_feedback_letters[-1].set(on=on,
                                                  text=text,
                                                  color=(self.level_2_letter_colors and
                                                         self.stimuli_colors[self._classified_letter] or
                                                         self.letter_color))
            if self.feedback_show_shape:
                if self.feedback_show_shape_at_center:
                    pos = self._centerPos
                else:
                    pos = self._countdown_shape_positions[self._classified_element]
                
                ## turn on/off current element:
                self._ve_shapes[self._classified_letter].set(on=(on and self.level_2_symbols),
                                                             position=(on and pos or self._centerPos))
                
                ## turn on/off letter of current element:
                idx = self._nr_letters + self._classified_letter
                self._ve_letters[idx].set(on=on, text=text, position=(on and pos or self._letter_positions[idx]))
                
        
    def switch_level(self):
            
        ## turn on written and desired words:
        self._ve_spelled_phrase.set(on=True)
        self._ve_current_letter.set(on=True)
        self._ve_desired_letters.set(on=True)
        self._ve_letterbox.set(on=True)
        
        if self.use_ErrP_detection and self._ErrP_classifier:
            self._ve_feedback_ErrP.set(on=True)
            self._show_feedback(True)
            self._presentation.set(go_duration=(self.feedback_ErrP_duration, 'seconds'))
            self._presentation.go()
            self._ve_feedback_ErrP.set(on=False)
            self._show_feedback(False)
            return
        
        if self._current_level==1:
            '''level 1: move classified letters to circles '''
            
            if self.level_2_animation:
                ## turn on all elements:
                for i in xrange(self._nr_elements):
                    self._ve_shapes[i].set(on=self.level_2_symbols, position=self._countdown_shape_positions[i])
                
                ## animate letters:
                def update(t):
                    dt = t/self.animation_time
                    self._viewport.parameters.stimuli = self._viewport.parameters.stimuli[:-(self._nr_elements)]
                    feedback_letters = []
                    for i in xrange(self._nr_elements):
                        pos = (animate_sinusoid(NP.add(self._letter_layout.positions[i], self._centerPos), self._countdown_shape_positions[i], dt))
                        font_size = int(round(animate(self.font_size_level1, self.font_size_level2, dt)))
                        color = (self.level_2_letter_colors and 
                                list(animate(self.letter_color, self.stimuli_colors[i], dt)) or 
                                self.letter_color)
                        text = (i==self._nr_elements-1 and 
                                self.backdoor_symbol or 
                                self.letter_set[self._classified_element][i])
                        feedback_letters.append(Text(position=pos,
                                                     color=color,
                                                     font_size=font_size,
                                                     text=text,
                                                     anchor="center"))
                    self._viewport.parameters.stimuli.extend(feedback_letters)
                    if self.feedback_show_shape_at_center:
                        pos = animate_sinusoid(self._centerPos, self._countdown_shape_positions[self._classified_element], dt)
                        self._ve_shapes[self._classified_element].set(position=pos)
                
                # send to screen:
                self._viewport.parameters.stimuli.extend([None]*(self._nr_elements))
                self._presentation.set(go_duration=(self.animation_time, 'seconds'))
                self._presentation.add_controller(None,None,FunctionController(during_go_func=update))
                self._presentation.go()
                self._presentation.remove_controller(None,None,None)
                self._viewport.parameters.stimuli = self._viewport.parameters.stimuli[:-(self._nr_elements)]
                
                ## turn on level 2 letters:
                for i in xrange(self._nr_elements):
                    text = (i==self._nr_elements-1 and 
                            self.backdoor_symbol or 
                            self.letter_set[self._classified_element][i])
                    self._ve_letters[self._nr_letters + i].set(on=True,
                                                               text=text,
                                                               position=self._countdown_letter_positions[self._nr_letters + i])

                #self.set_standard_screen()
            else:
                ## turn on all elements: 
                self.set_standard_screen()
                #self._presentation.set(go_duration=(10, 'seconds'))

         
            
            ## set elements back to center
            #if self.level_2_animation:
           
                
        else:
            ''' level 2: move classified letter to wordbox '''
            
            ## check if backdoor classified:
            if self._classified_letter >= len(self.letter_set[self._classified_element]):
                text = self.backdoor_symbol
            else:
                text = self.letter_set[self._classified_element][self._classified_letter]
            
            ## animate letter, but not if backdoor classified:
            if self._classified_letter < len(self.letter_set[self._classified_element]):
                def update(t):
                    dt = t/self.animation_time
                    self._viewport.parameters.stimuli = self._viewport.parameters.stimuli[:-1]
                    pos = animate_sinusoid(self._centerPos, self._current_letter_position, dt)
                    color = (self.level_2_letter_colors and
                             list(animate_sinusoid(self.stimuli_colors[self._classified_letter], self.current_letter_color, dt)) or
                             list(animate_sinusoid(self.letter_color, self.current_letter_color, dt)))
                    font_size = int(round(animate(self.font_size_level2, self.font_size_current_letter, dt)))    
                    self._viewport.parameters.stimuli.append(Text(position=pos,
                                                                  color=color,
                                                                  font_size=font_size,
                                                                  text=text,
                                                                  anchor='center'))
                # send to screen:
                self._viewport.parameters.stimuli.append(None)
                self._presentation.set(go_duration=(self.animation_time, 'seconds'))
                self._presentation.add_controller(None,None,FunctionController(during_go_func=update))
                self._presentation.go()
                self._presentation.remove_controller(None,None,None)
                self._viewport.parameters.stimuli = self._viewport.parameters.stimuli[:-1]
            else:
                self._presentation.set(go_duration=(self.animation_time, 'seconds'))
                self._presentation.go()
                
        ## turn off feedback box:
        self._ve_feedback_box.set(on=False)
    
    
    def pre__classify(self):
        self._ve_fixationpoint.set(on=True)
        self._ve_spelled_phrase.set(on=False)
        self._ve_current_letter.set(on=False)
        self._ve_desired_letters.set(on=False)
        self._ve_letterbox.set(on=False)
    def post__classify(self):
        self._ve_fixationpoint.set(on=False)


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.DEBUG)
    fb = PinyinSpellerVE()
    fb.on_init()
    fb.on_play()
