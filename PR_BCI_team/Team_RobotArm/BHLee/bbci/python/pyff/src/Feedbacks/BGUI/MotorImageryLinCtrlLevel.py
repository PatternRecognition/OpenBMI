'''
Created on 21.05.2012    

@author: jpascual
'''

from math import pow 
from Level import LevelBase
from VisionEgg.Text import Text
import socket

class MotorImageryLinCtrlLevel(LevelBase):
    '''
    classdocs
    '''
    
    def __init__(self, nrealelements, radius, classes,screenwidth):
        '''
        Constructor
        '''
        LevelBase.__init__(self, LevelBase.LIN_CTRL, nrealelements)
        
        self.idle_value = 1
        
        self.r = 90.0
        self.radius = radius
        self.screenwidth = screenwidth
        self.classes = classes   
        self.power_maxsize = 1100      
        self.value = 0
        self._orientation = 1
        
        # IMPORTANT PARAMETERS
        ##########################################################################################################
        
        self._selection_time = [3, 4, 4, 3]  # in seconds
        #self.update_coef = 1
        #self.anti_rotation_bias = -0.1                        
        self.powerbar_growth_time   = 0.1         # how fast the powerbar grows ( 20: slow, 1:fast)             
          
        ##########################################################################################################
                
        self.initial_power_length   = 0                 
                                
        self.shape_powerbar = {'orientation':90.0, 'inisize':(70., self.power_maxsize), 'size':(60., 100), 'num_thresholds':4, 'width_thresholds':50}
                
        self.shapes = [ ['circle',   {'radius':self.r}],
                        ['circle',   {'radius':self.r}],
                        ['circle',   {'radius':self.r}],
                        ['circle',   {'radius':self.r}]]       
                
        self._elap_thresholds = [0,0,0,0];
        self.reset_elap_thresholds()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self.addr = '192.168.123.105'
        self.port = 1234


    def reset_elap_thresholds(self):        
        for i in xrange(4):
            self._elap_thresholds[i] = 0;    
                
                
    def setClassifierOutput(self, cl_name,cl_value):
                                        
        self.value =  -1*cl_value
                    
        #if(self.value > 0):
        #    print("%s cl = %.2f ; power lenght= %.2f ) " % (self.classes[1], self.value, self.power_length))
        #else:
        #    print("%s cl = %.2f ; power lenght= %.2f ) " % (self.classes[0], self.value, self.power_length))
                               
    
    def get_shapes(self):
        return self.shapes
    
    
    def update_powerbar(self):
            
        v = self.value * self.idle_value * self._ve_powerbar.g;        
        #print("<BCI>NormalPos<Pos>%.2f</Pos><Time>%.3f</Time></BCI> " % ((2*self.power_length/self.power_maxsize - 1), self.total_elapsed_time))
        self.sock.sendto("<BCI>NormalPos<Pos>%.2f</Pos><Time>%.3f</Time></BCI> " % ((2*self.power_length/self.power_maxsize - 1), self.total_elapsed_time),(self.addr, self.port))                 

        if((self.power_length + (1.0 /self.powerbar_growth_time) * v) <0):
            self.power_length = 0;
        elif ((self.power_length + (1.0 /self.powerbar_growth_time) * v) > self.power_maxsize):
            self.power_length = self.power_maxsize
        else:
            self.power_length += (1.0 /self.powerbar_growth_time) * v                        
        
        return self._ve_powerbar.setValue(self.power_length)
        
               
                
    def update(self, total_time):      

        self.dt = total_time - self.total_elapsed_time          
        self.total_elapsed_time = total_time
        
        v = self.update_powerbar()
        if(v >= 0):
            self._elap_thresholds[v] += self.dt;
            #print("%d -> %.2f %.2f %.2f %.2f" % (v, self._elap_thresholds[0], self._elap_thresholds[1], self._elap_thresholds[2], self._elap_thresholds[3]))
            if(self._elap_thresholds[v] > self._selection_time[v]):
                print("%d SELECTED!!" % v)
                return v
        else:
            self.reset_elap_thresholds()
            
        return -1    
    


    def set_on(self, b):
                       
        LevelBase.set_on(self,b)
                        
        #self.sock.sendto("<BCI>NormalPos<Pos>0.1</Pos><Time>0</Time></BCI> ",(self.addr, self.port))     
  
        if(b == False):                       
            self._ve_powerbar.resetValue()                        
            self.power_length = self.power_maxsize/2
            self.reset_elap_thresholds()
        
        self._ve_powerbar.set(on=b)  

        self._message1.set(on=b)
        self._message2.set(on=b)
        
        if b:
            self.total_elapsed_time = 0;
            print("<BCI>StartControl</BCI>")       
            self.sock.sendto("<BCI>StartControl</BCI> ",(self.addr, self.port))     
        else:
            print("<BCI>StopControl</BCI>")       
            self.sock.sendto("<BCI>StopControl</BCI> ",(self.addr, self.port))     
             
        self.value = 0
        
        
        
    def start_up_shapes(self, visual, center, colors):

        LevelBase.start_up_shapes(self,visual, center, colors)            
        
        t1 = self.classes[1] 
        t2 = self.classes[0]

        self.center = center
                                                    
        offset = 150
   
        x1 = center[0] + self.power_maxsize/2 + 40
        x2 = center[0] - self.power_maxsize/2 - 40
              
        self._ve_powerbar =  self.registered_shapes['PowerBar2'](  position=[center[0] + self.power_maxsize/2, center[1]+300],
                                                                    screen_center =[center[0], center[1]],                                                                
                                                                    on=False,
                                                                    **self.shape_powerbar)
            
        visual.append(self._ve_powerbar)
                        
        self._message1 =   Text( position=(x1, center[1]-200),
                                   text=t1,
                                   font_size=30,
                                   color=[0.8, 0.45, 1],
                                   anchor='center', 
                                   on=False)
        
        self._message2 =   Text( position=(x2, center[1]-200 ),
                                   text=t2,
                                   font_size=30,
                                   color=[0.8, 0.45, 1],
                                   anchor='center', 
                                   on=False)
                                                                    
        visual.append(self._message1)
        visual.append(self._message2)                           



