'''
Created on 24.01.2012

@author: jpascual
'''
from VEShapes import FilledTriangle, FilledHexagon, FilledHourglass, FilledCross,StripeField,DotField,MyArrow,PowerBar, PowerBar2
from VisionEgg.MoreStimuli import FilledCircle, Target2D
import random
from MenuElement import *

class LevelBase(object):
    '''
    classdocs
    '''    
        
    ## Kinds of input
    P300 = 0
    MI = 1
    MI_SYNC = 2
    LIN_CTRL = 3                
    
    def __init__(self, mode, n):
        '''
        Constructor
        '''
        self._nr_elements_to_choose = n
         
        ## register possible shapes:
        self.registered_shapes = {  'circle':FilledCircle,
                                    'cross':FilledCross,
                                    'hexagon':FilledHexagon,
                                    'hourglass':FilledHourglass,
                                    'rectangle':Target2D,
                                    'triangle':FilledTriangle,
                                    'arrow':MyArrow,
                                    'PowerBar':PowerBar,
                                    'PowerBar2':PowerBar2}                            
        self._mode = mode        
        self._elems_container = None                                                      
        self._ve_shapes = []
        
                            
    def StartUp(self, center, colors):
        self.center = center
        self.stimuli_colors = colors       
    
    
    def GetNumRealElements(self):
        return self._nr_elements_to_choose
            
            
    def GetNumElems(self):
        return self._nr_elements_level
    
    
    def GetMode(self):
        return self._mode
    
    
    def GetCountDown(self):
        return self._countdown[self._state];
   
   
    def get_elem(self,i):      
        return self._elems_container.subelems[i]
        
        
    def get_current_elem_path(self, i):
        return self._elems_container.subelems[i].path
                   
             
    def set_on(self, b):

        for i in xrange(self._nr_elements_to_choose):
            self._elems_container.subelems[i].set_all_on(b)            
            self._ve_shapes[i].set(on=b)
             
             
    def set_elem_on(self, i, b, with_shape):
           
        self._elems_container.subelems[i].set_all_on(b)   
        if with_shape:  
            self._ve_shapes[i].set(on=b)             
    
    
    def random_pos(self):
        p = [0,1]
        random.shuffle(p)
        
        if(p[0]==0):
            pos1 = self._elems_container.subelems[0].original_pos
            pos2 = self._elems_container.subelems[1].original_pos
                  
            self._elems_container.subelems[0].original_pos = pos2
            self._elems_container.subelems[1].original_pos = pos1
            
            self._elems_container.subelems[1].texture.set(position=pos1)       
            self._ve_shapes[1].set(position=pos1) 
               
            self._elems_container.subelems[0].texture.set(position=pos2)       
            self._ve_shapes[0].set(position=pos2) 
        
        
    def set_elem_pos(self, elem, pos):        
        
        self._elems_container.subelems[elem].texture.set(position=pos)       
        self._ve_shapes[elem].set(position=pos) 


    def get_elem_pos(self, elem):        
        
        return self._elems_container.subelems[elem].original_pos
          
                 
    def reset_elements_pos(self):
        for i in xrange(self.GetNumElems()):
            self._elems_container.subelems[i].reset_position()            
               
     
    def get_shapes(self):
        ## override in subclasses
        print "override!!"
        
        
    def start_up_shapes(self, visual, center, colors):
     
        ## create shapes:
        shapes = self.get_shapes()
        
        p = [x for x in range(len(self.shapes))]
        q = [x for x in range(len(self.shapes))]
        
        if(self._mode != self.LIN_CTRL):
            random.shuffle(p)
            random.shuffle(q)
        
        for i in xrange(len(self.shapes)):            
            self._ve_shapes.append( self.registered_shapes[self.shapes[p[i]][0]](  position=center,
                                                                                color=colors[q[i]],
                                                                                on=False,
                                                                                **shapes[p[i]][1]))
            # put shape in container:
            visual.append(self._ve_shapes[i])   
        
        
    def show_message(self, b):
        self._ve_message.set(on=b)
    
    
    def start_up_elems(self, textures, visual, center, radius, colors, msg, font_size, font_color):
        
        self._ve_message =   Text( position=(center[0], center[1]+200),
                                   text=msg,
                                   font_size=font_size,
                                   color=font_color,
                                   anchor='center', 
                                   on=False)
                                              
        visual.append(self._ve_message);   
        
        print("LevelBase start_up_elems: elements to choose =%d" % (self._nr_elements_to_choose))
        
        p = list()
        if(self._mode != LevelBase.LIN_CTRL):
            if(self._nr_elements_to_choose > 1):           
                if(self._nr_elements_to_choose == 2):
                    layout = CircularLayout( nr_elements=self._nr_elements_to_choose,
                                                radius=radius,
                                                start=3.1415/2.*2)
                else:
                    layout = CircularLayout( nr_elements=self._nr_elements_to_choose,
                                             radius=radius,
                                            start=3.1415/6.*5)
                
                layout.positions.reverse()   
                p = layout.positions
            else:
                p.append([0, 0])
        else:
            w = 1100.0
            
            for i in range(self._nr_elements_to_choose+1):                
                wpos = i * w/5 + w/10                     
                if(i!=self._nr_elements_to_choose/2):
                    p.append((wpos-w/2, 50))
            
        self._elems_container = FooMenuElement("", None)
        self._nr_elements_level = len(textures) 
        self.start_up_shapes(visual, center, colors)        
         
        for i in xrange(self._nr_elements_to_choose):
            elem = MenuElementImage( id = i,
                                     parent=self._elems_container,
                                     position=(p[i][0] + center[0],
                                               p[i][1] + center[1]),
                                     texture = textures[i]);           
            
            self._elems_container.append(elem);
            visual.append(elem.texture)            
        
        for i in xrange(len(textures)- self._nr_elements_to_choose):
            elem = MenuElementImage( id = i +  self._nr_elements_to_choose,
                                     parent=self._elems_container,
                                     position=(center[0],
                                               center[1]),
                                     texture = textures[i+ self._nr_elements_to_choose]);           
            
            self._elems_container.append(elem);
            visual.append(elem.texture)            
            
        self.set_on(False)    
                               