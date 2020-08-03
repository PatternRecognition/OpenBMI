'''Feedbacks.Visual_auditory.TM_CenterSpeller
# Copyright (C) 2011  Xingwei An
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

Created on 2011-12-3

@author: "Xingwei An"

'''
from TempBase1 import TempBase1, animate_sinusoid, animate
from VEShapes import FilledTriangle, FilledHexagon,FilledHourglass,FilledCross
from VisionEgg.MoreStimuli import FilledCircle, Target2D
from VisionEgg.Text import Text
from lib.P300Layout.CircularLayout import CircularLayout
from VisionEgg.FlowControl import FunctionController


import logging
#import pygame
import numpy as NP
        

class TM_CenterSpeller(TempBase1):
    '''
    classdocs
    '''
    
    
    def init(self):
        '''
        initialize parameters
        '''

        self.init_pygame()
        TempBase1.init(self)
        
        self.synchronized_countdown = True
        self.do_animation = True
        ## sizes:
        self.letter_radius = 40
        self.speller_radius = 250
        self.font_size_level1 = 45         # letters in level 1
        self.font_size_level2 = 130         # letters in level 2
        self.feedbackbox_size = 200.0
        self.fixationpoint_size = 4.0
        self.font_size_feedback_ErrP = 300
        
        ## stimulus types:
        self.stimulustype_color = True
        self.shape_on=True
        
        # self.stimulus_duration = 0.083   # 5 frames @60 Hz = 83ms flash
        # self.interstimulus_duration = 0.037


        ## feedback type:
        self.feedback_show_shape = True
        self.feedback_show_shape_at_center =True
        

        ## colors:
        self.shape_color = (1.0, 1.0, 1.0)

        
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

        self.shapes = [
            [ "triangle", {"innerColor":self.bg_color, "innerSize": 60, "size": 200}], 
            [ "hourglass", {"size": 100 }], 
            [ "cross", {"orientation": 45, "size": [30,180], "innerColor":self.bg_color}], 
            [ "triangle", {"innerColor":self.bg_color, "innerSize": 60, "orientation": 180, "size": 200}], 
            [ "hourglass", {"orientation": 90, "size": 100}], 
            [ "cross", {"size": [30,180], "innerColor":self.bg_color }]
        ]
    
        
    def prepare_mainloop(self):
        '''
        called in pre_mainloop of superclass.
        '''
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
                       ['triangle',  {'size':200}]]

            
    
    def init_screen_elements(self):
        '''
        overwrite this function in subclass.
        '''
        ## create shapes:

        if self.do_animation:
            for i in xrange(self._nr_elements):
                self._ve_shapes.append(self.registered_shapes[self.shapes[i][0]]( 
                                       position=self._centerPos,
                                       #color=self.stimuli_colors[i],
                                       color=self.shape_color[i],
                                       on=False,
                                       **self.shapes[i][1]))

                       
            ## add letters of level 1:
            circle_layout = CircularLayout(nr_elements=self._nr_elements,
                                           radius=self.speller_radius,
                                           start=NP.pi/self._nr_elements*(self._nr_elements-1))
            circle_layout.positions.reverse()
            self._letter_layout = CircularLayout(nr_elements=self._nr_elements_A,
                                           radius=self.letter_radius,
                                           start=NP.pi/self._nr_elements_A*(self._nr_elements_A-1))
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
            

            # put letters in container:
            self._ve_elements.extend(self._ve_letters)
                    
            
            ## create feedback box:
            self._ve_feedback_box = Target2D(position=self._centerPos,
                                             size=(self.feedbackbox_size, self.feedbackbox_size),
                                             color=self.feedback_color,
                                             on=False)
            
            ## add feedback letters:
            self._ve_feedback_letters = []
            for i in xrange(self._nr_elements_A):
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
                                           start=NP.pi/self._nr_elements*(self._nr_elements-1))
            self._letter_layout = CircularLayout(nr_elements=self._nr_elements_A,
                                           radius=self.letter_radius,
                                           start=NP.pi/self._nr_elements_A*(self._nr_elements_A-1))
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
            

            # put letters in container:
            self._ve_elements.extend(self._ve_letters)
                    
            
            ## create feedback box:
            self._ve_feedback_box = Target2D(position=self._centerPos,
                                             size=(self.feedbackbox_size, self.feedbackbox_size),
                                             color=self.feedback_color,
                                             on=False)
            
            ## add feedback letters:
            self._ve_feedback_letters = []
            for i in xrange(self._nr_elements_A):
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
            self._ve_shapes[i].set(on=is_level1)

        for i in xrange(self._nr_letters): # level 1 letters
            self._ve_letters[i].set(on=is_level1)



        def update(t):
            dt = t/self.animation_time
            for i in xrange(self._nr_elements):
                pos = animate_sinusoid(self._centerPos, self._countdown_shape_positions[i], dt)
                self._ve_shapes[i].set(position=pos) # shapes
                
            for i in xrange(self._nr_letters):
                pos = animate_sinusoid(self._letter_positions[i], self._countdown_letter_positions[i], dt)
                self._ve_letters[i].set(position=pos) # level 1 letters
        def update2(t):
            for i in xrange(self._nr_elements):
                self._ve_shapes[i].set(position=self._countdown_shape_positions[i])
           
            for i in xrange(self._nr_letters):
                self._ve_letters[i].set(position=self._countdown_letter_positions[i])

        if self.do_animation:     
            self._presentation.set(go_duration=(self.animation_time, 'seconds'))
            self._presentation.add_controller(None,None,FunctionController(during_go_func=update))
        else:
            self._presentation.set(go_duration=(self.interstimulus_duration_c,'seconds'))
            self._presentation.add_controller(None,None,FunctionController(during_go_func=update2)) 

        
        # send to screen:
        self._presentation.go()
        self._presentation.remove_controller(None,None,None)
        
        for i in xrange(self._nr_elements):
            self._ve_shapes[i].set(position=self._countdown_shape_positions[i])

        for i in xrange(self._nr_letters):
            self._ve_letters[i].set(position=self._countdown_letter_positions[i])

    
    def set_standard_screen(self):
        '''
        set screen elements to standard state.
        '''

        self._countdown_screen = False
        
        # move all elements with their letters to standard position:
        for i in xrange(self._nr_elements*self._nr_elements_A):
            self._ve_letters[i].set(color = self.letter_color)
        def update(t):
            dt = t/self.animation_time
            for i in xrange(self._nr_elements):
                pos = animate_sinusoid(self._countdown_shape_positions[i], self._centerPos, dt)
                self._ve_shapes[i].set(position=pos) # shapes
#                self._ve_letters[self._nr_letters + i].set(position=pos) # level 2 letters
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

        for i in xrange(self._nr_letters):
            self._ve_letters[i].set(position=self._letter_positions[i], on=False)
            
        self._ve_countdown.set(on=False, text=" ")
    
    def stimulus(self, i_element, on=True):
        '''
        turn on/off the stimulus elements and turn off/on the normal elements.
        '''
        self._ve_spelled_phrase.set(on=True)
        self._ve_current_letter.set(on=True)
        self._ve_desired_letters.set(on=True)
        self._ve_letterbox.set(on=True)
        self._ve_shapes[i_element].set(on=on)
        for i in xrange(len(self.letter_set[i_element])):
            self._ve_letters[(self._nr_elements_A-1) * i_element + i].set(on=on)
        
    def feedback(self):
        self._ve_spelled_phrase.set(on=True)
        self._ve_current_letter.set(on=True)
        self._ve_desired_letters.set(on=True)
        self._ve_letterbox.set(on=True)
        # turn on feedback box and turn off fixationpoint:
        self._ve_feedback_box.set(on=True)
#        self.fixation(False)
        
        ''' present classified letter group in center and move letters to circles '''
            
        ## display letter group:
            
        for i in xrange(self._nr_elements_A):
            self._ve_feedback_letters[i].set(on=True,
                                            text=self.letter_set[self._classified_element][i])
        ## turn on current element:
        #self.stimulus(self._classified_element, True)
            
        ## turn off other letters:
        idx_start = self._classified_element*(self._nr_elements_A)
        idx_end = idx_start + self._nr_elements_A
        for i in xrange(idx_start):
            self._ve_letters[i].set(on=False)
        for i in xrange(idx_end, self._nr_letters):
            self._ve_letters[i].set(on=False)
            
        ## present:
        #self.stimulus(self._classified_element, True)
        self._presentation.set(go_duration=(self.feedback_duration, 'seconds'))
        self._presentation.go()
            
        
        text = self.letter_set[self._classified_element][self._classified_letter]
                
        ## display letter:
        self._ve_feedback_letters[-1].set(on=True, text=text)
            
            # turn on current element stimulusw:
        if not self.offline:
            self.stimulus(self._classified_letter, True)
            

            
        ## present:
        self._presentation.set(go_duration=(1, 'seconds'))
        self._presentation.go()
            
            ## turn off current element stimulus:
        if not self.offline:
            self.stimulus(self._classified_letter, False)
#            self._ve_letters[self._nr_letters + self._classified_letter].set(on=False)
        self._ve_feedback_letters[-1].set(on=False)
         
            ## animate letter, but not if backdoor classified:
        if self._classified_letter < len(self.letter_set[self._classified_element]):
            self._presentation.set(go_duration=(self.animation_time, 'seconds'))
            self._viewport.parameters.stimuli.append(None)
            def update(t):
                dt = t/self.animation_time
                self._viewport.parameters.stimuli = self._viewport.parameters.stimuli[:-1]
                pos = animate_sinusoid(self._centerPos, self._current_letter_position, dt)
                color = animate_sinusoid(self.letter_color, self.current_letter_color, dt)
                font_size = int(round(animate(self.font_size_level2, self.font_size_current_letter, dt)))
                self._viewport.parameters.stimuli.append(Text(position=pos,
                                                                  color=color,
                                                                  font_size=font_size,
                                                                  text=text,
                                                                  anchor='center'))
            self._presentation.add_controller(None,None,FunctionController(during_go_func=update))
                
                # send to screen:
            self._presentation.go()
            self._presentation.remove_controller(None,None,None)
            self._viewport.parameters.stimuli = self._viewport.parameters.stimuli[:-1]

            ## turn off current element:
        for i in xrange(idx_start, idx_end):
            self._ve_letters[i].set(on=False)
        for i in xrange(self._nr_elements_A):
            self._ve_feedback_letters[i].set(on=False)       
                    
            ## turn on level 1 letters:
        for i in xrange(self._nr_letters):
            self._ve_letters[i].set(on=True)
        
        ## turn off feedback box and turn on fixationpoint:
        self._ve_feedback_box.set(on=False)
#        self.fixation()    
        


    
    def pre__classify(self):
        self._ve_fixationpoint.set(on=True)
        self._ve_spelled_phrase.set(on=False)
        self._ve_current_letter.set(on=False)
        self._ve_desired_letters.set(on=False)
        self._ve_letterbox.set(on=False)
    def post__classify(self):
        self._ve_fixationpoint.set(on=False)
    

if __name__ == "__main__":
    
    logging.basicConfig(level=logging.DEBUG)
    

    screenPos = [100, 100, 800, 600]
    tp =TM_CenterSpeller()
    tp.on_init()
    tp.on_play()
    
   
    


        