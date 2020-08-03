'''Feedbacks.BGUI.BGUI
# Copyright (C) 2012  "Javier Pascual"
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
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

Created on Jan 01, 2012

@author: "Javier Pascual"

'''
from BGUIBaseHybrid import BGUIBaseHybrid, animate_sigmoid

from lib.P300Layout.CircularLayout import CircularLayout
from VisionEgg.FlowControl import FunctionController
from MenuElement import *
from LevelContainer import *

#import numpy as NP
import copy


class BGUI(BGUIBaseHybrid):
    '''
    classdocs
    '''
     
    def init(self):
        '''
        initialize parameters
        '''
        BGUIBaseHybrid.init(self)
      
        self.show_command_recieved = False
        
     
        
    def prepare_mainloop(self):
        '''
        called in pre_mainloop of superclass.
        '''
        print("prepare_mainloop")
        self._countdown_screen = False
     
    
    
    def init_screen_elements(self):                    
                                 
        ## add fixation point:
        self._ve_fixationpoint = FilledCircle(radius=self.fixationpoint_size,
                                              position=self._centerPos,
                                              color=self.fixationpoint_color,
                                              on=False)       
        self._ve_elements.append(self._ve_fixationpoint)
    
         
        if not self.free:
            ## create feedback box:
            self._ve_feedback_box = Target2D(position=self._centerPos,
                                             size=(self.feedback_attr["box_size"], self.feedback_attr["box_size"]),
                                             color=self.feedback_attr["color"],
                                             on=False)            
            ## put letters in container:
            self._ve_elements.append(self._ve_feedback_box)
            
        
        
    def set_countdown_screen(self):
        '''
        set screen how it should look during countdown.
        '''
        if self._countdown_screen:
            return
        
        self._countdown_screen = True
        
        ## turn on visible elements:                                         
        self._levels.set_on(True)
                                       
        ## move all elements with their letters to countdown position (from the center to the circle):
        self._presentation.set(go_duration=(self.animation_time, 'seconds'))
       
        def update(t):
            dt = t/self.animation_time
            for i in xrange(self._levels.GetNumRealElements()):               
                pos = animate_sigmoid(self._centerPos, self._levels.get_elem_pos(i), dt)                
                self._levels.set_elem_pos(i, pos)
                                           
        self._presentation.add_controller(None,None,FunctionController(during_go_func=update))
                    
        self._presentation.go()
        self._presentation.remove_controller(None,None,None)

        self._levels.reset_elements_pos();        
                
            

    def set_standard_screen(self):
        '''
        set screen elements to standard state.
        '''
        print("set_standard_screen")
        self._countdown_screen = False
        
        if(self._levels.GetMode() == LevelContainer.P300):
            # move all elements to the center:            
            self._presentation.set(go_duration=(self.animation_time, 'seconds'))
            
            def update(t):
                dt = t/self.animation_time
                for i in xrange(self._levels.GetNumRealElements()):  
                    pos = animate_sigmoid(self._levels.get_elem_pos(i) , self._centerPos, dt)                
                    self._levels.set_elem_pos(i, pos)                          
                    
            self._presentation.add_controller(None, None, FunctionController(during_go_func=update))
            
            # send to screen:
            self._presentation.go()
            self._presentation.remove_controller(None,None,None)
            
            for i in xrange(self._levels.GetNumElems()):            
                self._levels.set_elem_pos(i, self._centerPos)    
            
            self._levels.set_on(False)
            
        elif((self._levels.GetMode() == LevelContainer.MI) or (self._levels.GetMode() == LevelContainer.MISync)):
            # lets draw the arrow! (should be inside the Level class :S )                    
            self._levels.show_arrow(True)
            
        self._ve_countdown.set(on=False, text=" ")
    


    def stimulus(self, i_element, on=True):
        '''
        turn on/off the stimulus elements and turn off/on the normal elements.
        '''            
        self._levels.set_elem_on(i_element,on)            
   
        
    def feedback(self):
        '''
        Show classified element / letter(s). 
        '''
        pygame.time.wait(700)
            
        self._levels.set_on(False)         
        e = self._show_feedback(True)
                       
        self._levels.show_message(True)
       
        self._presentation.set(go_duration=(self.feedback_attr["duration"], 'seconds'))
        self._presentation.go()            
             
        self._show_feedback(False,e)             
             
        self._levels.show_message(False)                
    
        pygame.time.wait(200)
    
        
        
    def _show_feedback(self, on, e=-1):
        
        ## turn on/off feedback box showing the selected action:
        if not self.feedback_attr["show_shape_at_center"] and not self.free:
            self._ve_feedback_box.set(on=on)
                
        if(self._levels.GetState() == LevelContainer.STATE_SELECT):
                if(e != -1):
                    self._levels.set_elem_pos(e, self._centerPos)      
                    self._levels.set_elem_on(e,on, False)
                else:                                
                    self._levels.set_elem_pos(self._classified_element, self._centerPos)      
                    self._levels.set_elem_on(self._classified_element,on, False)
                return self._classified_element                
        elif(self._levels.GetState()  == LevelContainer.STATE_CONFIRM):
            if self._confirmed:
                self._levels.set_elem_pos(0, self._centerPos)
                self._levels.set_elem_on(0,on, False)
                return 0
            else:
                if(self._levels.GetMode() == LevelContainer.P300):
                    self._levels.set_elem_pos(1, self._centerPos)
                    self._levels.set_elem_on(1,on, False)
                else:
                    self._levels.set_elem_pos(1, self._centerPos)
                    self._levels.set_elem_on(1,on, False)    
                return 1                               
        else:
            if self._action_stopped:
                self._levels.set_elem_pos(0, self._centerPos)
                self._levels.set_elem_on(0,on)
                return 0
        
           
    def pre__classify(self):
        self._ve_fixationpoint.set(on=True)
        self._spelled_actions.set_all_on(on=False)            
                        
        
    def post__classify(self):
        self._ve_fixationpoint.set(on=False)


######################################################################################
######################################################################################

if __name__ == '__main__':
    fb = BGUI()
    fb.load_variables("feedback_base.json")
    fb.load_variables("feedback_online_LinCtrl.json")
    fb.on_init()    
    fb.on_play()
