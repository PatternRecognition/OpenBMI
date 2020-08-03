'''Feedbacks.VisualSpeller.CakeSpellerVE
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

Created on Mar 30, 2010

@author: "Nico Schmidt"

This module was developed as part of the study
Treder MS, Schmidt NS, Blankertz B (2011). Gaze-independent brain-computer interfaces based on covert attention and feature attention. J Neu Eng, 8.

Requires VisionEgg and Pygame.

'''
from VisualSpellerVE import VisualSpellerVE, animate, animate_sigmoid #,animate_sinusoid
from VEShapes import FilledTriangle
from VisionEgg.MoreStimuli import Target2D, FilledCircle
from VisionEgg.Text import Text
from lib.P300Layout.CircularLayout import CircularLayout
from VisionEgg.FlowControl import FunctionController

import numpy as NP

class CakeSpellerVE(VisualSpellerVE):
    '''
    Visual speller as a cake with six pieces.
    '''


    def init(self):
        '''
        initialize parameters
        '''
        VisualSpellerVE.init(self)
        
        ## sizes:
        self.letter_radius = 70
        self.speller_radius = 380
        self.fixationpoint_size = 5.0
        self.edge_size = 5
        self.font_size_level1 = 70         # letters in level 1
        self.font_size_level2 = 130         # letters in level 2
        self.feedbackbox_size = 200.0
        
        ## colors:
        self.shape_color = (0.0, 0.0, 0.0)
        self.edge_color = (1.0, 1.0, 1.0)
        self.stimuli_colors = [(1.0, 0.0, 0.0),
                               (0.0, 1.0, 0.0),
                               (0.0, 0.0, 1.0),
                               (1.0, 1.0, 0.0),
                               (1.0, 0.0, 1.0),
                               (0.0, 1.0, 1.0)]
        self.letter_color = (1.0, 1.0, 1.0)
        self.letter_stimulus_color = (0.0, 0.0, 0.0)
        self.fixationpoint_color = (1.0, 1.0, 1.0)
        self.feedback_color = (0.7, 0.7, 0.7)
        self.countdown_color =self.bg_color
       
    def prepare_mainloop(self):
        '''
        called in pre_mainloop of superclass.
        '''
        assert len(self.stimuli_colors)==self._nr_elements
        
        ## init containers for VE elements:
        self._ve_shapes = []
        self._ve_edges = []
        self._ve_letters = []
        
    
    def init_screen_elements(self):
        '''
        Initialize screen elements
        '''        
        self._letter_positions = []
        ## create triangles:
        self._letter_layout = CircularLayout(nr_elements=self._nr_elements,
                                       radius=self.letter_radius,
                                       start=NP.pi/6.*5)
        self._letter_layout.positions.reverse()
        a = self.speller_radius / 2.
        b = a * NP.sqrt(3) / 3.
        self._shape_positions = [(self._centerPos[0],     self._centerPos[1] + 2*b),
                                 (self._centerPos[0] + a, self._centerPos[1] + b),
                                 (self._centerPos[0] + a, self._centerPos[1] - b),
                                 (self._centerPos[0],     self._centerPos[1] - 2*b),
                                 (self._centerPos[0] - a, self._centerPos[1] - b),
                                 (self._centerPos[0] - a, self._centerPos[1] + b)]
        orientaion = [180., 0.,
                      180., 0.,
                      180., 0.]
        for i in xrange(self._nr_elements):
            self._ve_edges.append(FilledTriangle(size=self.speller_radius,
                                                 position=self._shape_positions[i],
                                                 orientation=orientaion[i],
                                                 color=self.edge_color))
            self._ve_shapes.append(FilledTriangle(size=self.speller_radius - self.edge_size,
                                                  position=self._shape_positions[i],
                                                  orientation=orientaion[i],
                                                  color=self.shape_color))
            
            ## add the letters of level 1:
            for j in xrange(len(self.letter_set[i])): # warning: self.letter_set must be at least of length self._nr_elements!!!
                self._letter_positions.append((self._letter_layout.positions[j][0]+self._shape_positions[i][0],
                                               self._letter_layout.positions[j][1]+self._shape_positions[i][1]))
                self._ve_letters.append(Text(position=self._letter_positions[-1],
                                             text=self.letter_set[i][j],
                                             font_size=self.font_size_level1,
                                             color=self.letter_color,
                                             anchor='center'))
        
        ## add letters of level 2:
        for i in xrange(self._nr_elements):
            self._ve_letters.append(Text(position=self._shape_positions[i],
                                         text=" ",
                                         font_size=self.font_size_level2,
                                         color=self.letter_color,
                                         anchor='center',
                                         on=False))
        
        
        ## add fixation point:
        self._ve_fixationpoint = FilledCircle(radius=self.fixationpoint_size,
                                              position=self._centerPos,
                                              color=self.fixationpoint_color)
                
        
        ## create feedback box:
        self._ve_feedback_box = Target2D(position=self._centerPos,
                                         size=(self.feedbackbox_size, self.feedbackbox_size),
                                         color=self.feedback_color,
                                         on=False)
        
        ## add feedback letters:
        self._ve_feedback_letters = []
        for i in xrange(self._nr_elements-1):
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
        
        ## put all in elements container:
        self._ve_elements.extend(self._ve_edges)
        self._ve_elements.extend(self._ve_shapes)
        self._ve_elements.extend(self._ve_letters)
        self._ve_elements.append(self._ve_feedback_box)
        self._ve_elements.extend(self._ve_feedback_letters)
        self._ve_elements.append(self._ve_fixationpoint)
    
    
    def set_countdown_screen(self):
        '''
        set screen elements to countdown state
        '''
        self.set_standard_screen(False)
 
    
    
    def set_standard_screen(self, std=True):
        '''
        set screen elements to standard state.
        '''
        is_level1 = self._current_level==1
        for i in xrange(self._nr_elements):
            self._ve_shapes[i].set(color=(std and self.shape_color or self.stimuli_colors[i]), on=True)
            self._ve_letters[self._nr_letters + i].set(on=not is_level1, color=(std and self.letter_color or self.letter_stimulus_color))
        for i in xrange(self._nr_letters):
            self._ve_letters[i].set(on=is_level1, color=(std and self.letter_color or self.letter_stimulus_color))
    
    def stimulus(self, i_element, on=True):
        '''
        turn on/off the stimulus elements and turn off/on the normal elements.
        '''
        self._ve_shapes[i_element].set(color=(on and self.stimuli_colors[i_element] or self.shape_color))
        if self._current_level==1:
            for i in xrange(len(self.letter_set[i_element])):
                self._ve_letters[(self._nr_elements-1)*i_element + i].set(color=(on and self.letter_stimulus_color or self.letter_color))
        else:
            self._ve_letters[self._nr_letters + i_element].set(color=(on and self.letter_stimulus_color or self.letter_color))

    def fixation(self, state=True):
        """
        turn on/off the fixation elements.

        """
        self._ve_fixationpoint.set(on=state)
    
    def feedback(self):
        # turn on feedback box and turn off fixationpoint:
        self._ve_feedback_box.set(on=True)
        self.fixation(False)
        
        if self._current_level == 1:
            
            '''level 1: present classified letter group in center and move letters to circles '''
            
            ## display letter group:
            for i in xrange(self._nr_elements-1):
                self._ve_feedback_letters[i].set(on=True,
                                                 text=self.letter_set[self._classified_element][i])
            ## turn on current element:
            self.stimulus(self._classified_element, True)
            
            ## turn off other letters:
            idx_start = self._classified_element*(self._nr_elements-1)
            idx_end = idx_start + self._nr_elements-1
            for i in xrange(idx_start):
                self._ve_letters[i].set(on=False)
            for i in xrange(idx_end, self._nr_letters):
                self._ve_letters[i].set(on=False)
            
            ## present:
            self.stimulus(self._classified_element, False)
            self._presentation.set(go_duration=(self.feedback_duration, 'seconds'))
            self._presentation.go()
            
            ## turn off current element:
            for i in xrange(idx_start, idx_end):
                self._ve_letters[i].set(on=False)
            for i in xrange(self._nr_elements-1):
                self._ve_feedback_letters[i].set(on=False)
            
            ## animate letters:
            self._presentation.set(go_duration=(self.animation_time, 'seconds'))
            self._viewport.parameters.stimuli.extend([None]*(self._nr_elements-1))
            def update(t):
                dt = t/self.animation_time
                self._viewport.parameters.stimuli = self._viewport.parameters.stimuli[:-(self._nr_elements-1)]
                feedback_letters = []
                for i in xrange(self._nr_elements-1):
                    pos = animate_sigmoid(NP.add(self._letter_layout.positions[i], self._centerPos), self._shape_positions[i], dt)
                    font_size = int(round(animate(self.font_size_level1, self.font_size_level2, dt)))
                    feedback_letters.append(Text(position=pos,
                                                 color=self.letter_color,
                                                 font_size=font_size,
                                                 text=self.letter_set[self._classified_element][i],
                                                 anchor="center"))
                self._viewport.parameters.stimuli.extend(feedback_letters)
            self._presentation.add_controller(None,None,FunctionController(during_go_func=update))
            
            # send to screen:
            self._presentation.go()
            self._presentation.remove_controller(None,None,None)
            self._viewport.parameters.stimuli = self._viewport.parameters.stimuli[:-(self._nr_elements-1)]
                
            ## turn on level 2 letters:
            for i in xrange(len(self.letter_set[self._classified_element])):
                self._ve_letters[self._nr_letters + i].set(on=True, text=self.letter_set[self._classified_element][i])

        else: 
            ''' level 2: present classified letter and move it to wordbox '''
            
            ## check if backdoor classified:
            if self._classified_letter >= len(self.letter_set[self._classified_element]):
                text = ' '
            else:
                text = self.letter_set[self._classified_element][self._classified_letter]
                
            ## display letter:
            self._ve_feedback_letters[-1].set(on=True, text=text)
            
            ## turn on current element stimulusw:
            if not self.offline:
                self.stimulus(self._classified_letter, True)
            
            ## turn off other letters:
            for i in xrange(self._classified_letter):
                self._ve_letters[self._nr_letters + i].set(on=False)
            for i in xrange(self._classified_letter+1, self._nr_elements):
                self._ve_letters[self._nr_letters + i].set(on=False)
            
            ## present:
            self._presentation.set(go_duration=(1, 'seconds'))
            self._presentation.go()
            
            ## turn off current element stimulus:
            if not self.offline:
                self.stimulus(self._classified_letter, False)
                self._ve_letters[self._nr_letters + self._classified_letter].set(on=False)
            self._ve_feedback_letters[-1].set(on=False)
            
            ## animate letter, but not if backdoor classified:
            if self._classified_letter < len(self.letter_set[self._classified_element]):
                self._presentation.set(go_duration=(self.animation_time, 'seconds'))
                self._viewport.parameters.stimuli.append(None)
                def update(t):
                    dt = t/self.animation_time
                    self._viewport.parameters.stimuli = self._viewport.parameters.stimuli[:-1]
                    pos = animate_sigmoid(self._centerPos, self._current_letter_position, dt)
                    color = animate_sigmoid(self.letter_color, self.current_letter_color, dt) 
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
            else:
                self._presentation.set(go_duration=(self.animation_time, 'seconds'))
                self._presentation.go()
                
                    
            ## turn on level 1 letters:
            for i in xrange(self._nr_letters):
                self._ve_letters[i].set(on=True)
        
        ## turn off feedback box and turn on fixationpoint:
        self._ve_feedback_box.set(on=False)
        self.fixation()



if __name__ == '__main__':
    fb = CakeSpellerVE()
    fb.on_init()
    fb.on_play()



        
