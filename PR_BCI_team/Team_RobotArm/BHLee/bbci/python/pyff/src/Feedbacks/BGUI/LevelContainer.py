'''
Created on 24.01.2012

@author: jpascual
'''
from VEShapes import FilledTriangle, FilledHexagon, FilledHourglass, FilledCross,StripeField,DotField,MyArrow
from VisionEgg.MoreStimuli import FilledCircle, Target2D

from MenuElement import *
from Level import LevelBase
from P300Level import *
from MotorImageryLevel import *
from MotorImageryLevelSync import *
from MotorImageryLinCtrlLevel import *

import threading

class LevelContainer(object):
    '''
    classdocs
    '''
    
    ## States
    STATE_SELECT = 0;
    STATE_CONFIRM = 1;
    
    ## Triggers
    STIMULUS = [ [11, 12, 13, 14, 15, 16, 17] , [21, 22, 23, 24, 25, 26, 27] , [21, 22, 23, 24, 25, 26, 27] ]
    RESPONSE = [ 51, 52, 53, 54, 55, 56, 57  ]
    RESPONSE_CONFIRMATION = [ [61, 62, 63, 64, 65, 66, 67] , [71, 72, 73, 74, 75, 76, 77]]  
    
    BEGIN_LEVEL = [231,232,233]     
    END_LEVEL = [241,242,243]     
    TIMEOUT = 400
    ## Kinds of input
    P300 = 0
    MI = 1
    MISync = 2
    
    
    def __init__(self, c):
        '''
        Constructor
        '''              
        self._state = self.STATE_SELECT
        self._countdown = c 
        self._levels_list = []            
        self.lock =  threading.Lock()
                                            
    def GetNumRealElements(self):
        return self._levels_list[self._state].GetNumRealElements()
         
    def GetNumElems(self):
        return self._levels_list[self._state].GetNumElems()
    
    def SetState(self, new_state):
        self._state = new_state

    def GetState(self):
        return self._state
    
    def GetMode(self):
        return  self._levels_list[self._state].GetMode()
    
    def GetCountDown(self):
        return self._countdown[self._state]*self._levels_list[self._state].GetNumRealElements();

    def GetBeginLevelMarker(self):
        return self.BEGIN_LEVEL[self._state]
    
    def GetEndLevelMarker(self):
        return self.END_LEVEL[self._state]
          
    def GetTimeoutMarker(self):
        return self.TIMEOUT
          
    def GetMarker(self,elem):
        return self.STIMULUS[self._state][elem] 
    
    def GetResponseMarker(self,elem):
        return self.RESPONSE[elem] 
    
    def GetResponseConfirmationMarker(self,confirmed,elem):
        return self.RESPONSE_CONFIRMATION[confirmed][elem]

    def GetActionStoppedMarker(self,stopped):
        return self.RESPONSE_STOP_ACTION[stopped]           

    def get_level_elem(self,l,i):
        return self._levels_list[l].get_elem(i)
   
    def get_elem(self,i):
        return self._levels_list[self._state].get_elem(i)

    def show_message(self, b):
        self._levels_list[self._state].show_message(b)
               
    def show_arrow(self, b):
        return self._levels_list[self._state].show_arrow(b)
       
    def setClassifierOutput(self, cl_name, cl_value):
        with self.lock:     
            return self._levels_list[self._state].setClassifierOutput(cl_name, cl_value)
    
    def update(self,dt):
        with self.lock:        
            return self._levels_list[self._state].update(dt)
               
    def set_on(self, b):        
        self._levels_list[self._state].set_on(b)                  
                  
    def set_elem_on(self, i, b, with_shape = True):
        self._levels_list[self._state].set_elem_on(i,b,with_shape)                  

    def reset_elements_pos(self):
        self._levels_list[self._state].reset_elements_pos()   
         
    def set_elem_pos(self, elem, pos):                        
        self._levels_list[self._state].set_elem_pos(elem, pos)   
     
    def get_elem_pos(self, elem):                        
        return self._levels_list[self._state].get_elem_pos(elem)   
    
    def get_layout_pos(self, elem):                        
        return self._levels_list[self._state].layout[elem]   
            
    
    def AddLevel(self, center, colors, kind, textures, radius, visual,msg, font_size, font_color, nrealelements, classes, screenwidth):
        
        if(kind == LevelBase.P300):
            l = P300Level(nrealelements)
        elif(kind == LevelBase.MI):
            l = MotorImageryLevel(nrealelements, radius, classes, screenwidth)
        elif(kind == LevelBase.MI_SYNC):
            l = MotorImageryLevelSync(nrealelements, radius, classes, screenwidth)
        else:
            l = MotorImageryLinCtrlLevel(nrealelements, radius, classes, screenwidth)
            
        l.start_up_elems(textures, visual, center, radius, colors,msg, font_size, font_color)
        self._levels_list.append(l)           
           
           