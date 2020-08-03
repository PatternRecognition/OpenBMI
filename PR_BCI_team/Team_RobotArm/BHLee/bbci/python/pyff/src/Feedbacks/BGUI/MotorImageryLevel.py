'''
Created on 24.01.2012    

@author: jpascual
'''

from math import pow 
from Level import LevelBase
from VisionEgg.Text import Text

class MotorImageryLevel(LevelBase):
    '''
    classdocs
    '''
    
    def __init__(self, nrealelements, radius, classes,screenwidth):
        '''
        Constructor
        '''
        LevelBase.__init__(self, LevelBase.MI, nrealelements)
        
        self.idle_value = 1
        
        self.r = 160.0
        self.radius = radius
        self.screenwidth = screenwidth
        self.classes = classes
        self.arrow_max_length       = radius - self.r + 100;
        self.arrow_rotation_time    = 18        
        self.arrow_shrinkage_time   = 2      
        self.power_maxsize = 795
                      
        self.value = 0
        self._orientation = -1
        
        
        # IMPORTANT PARAMETERS
        ##########################################################################################################
        self.ONLY_POWER_BAR = False
        self.QUEUED = True
        
        self.update_coef = 1
        self.anti_rotation_bias = -0.1
        
        if(nrealelements == 2):          
            self.initial_arrow_length   = 0.10                    
            self.powerbar_growth_time   = 80           # how fast the arrow grows ( 100: slow, 50:fast) (for 2 elements)
        else:
            self.initial_arrow_length   = 0.2        
            self.arrow_growth_time      = 3                     
            self.powerbar_growth_time   = 0.15         # how fast the powerbar grows ( 20: slow, 1:fast)             
     
        self.threshold_increase_row = 0.9               # 0.9: narrow, 0.7: thick. Should be always > 0.5 !!!
        self.threshold_rotate_row = 0.9
        ##########################################################################################################
        
          
        self.initial_power_length   = 0
        
        self.arrow_length = self.initial_arrow_length           
        self.power_length = 400
                
        self.min_length = self.arrow_length*self.arrow_max_length ;

        self.shape_arrow = {'orientation':0.0, 'size':(20.0,  self.min_length ), 'anchor':'top'}
        self.shape_arrow = {'orientation':0.0, 'size':(20.0,  self.min_length ), 'anchor':'top'}
    
     
        
        self.shape_powerbar = {'orientation':90.0, 'inisize':(70., self.power_maxsize), 'size':(60., 100), 'threshold1':self.threshold_increase_row, 'threshold2':1-self.threshold_rotate_row}
        
        
        self.shapes = [ ['circle',   {'radius':self.r}],
                        ['circle',   {'radius':self.r}],
                        ['circle',   {'radius':self.r}],
                        ['circle',   {'radius':self.r}],
                        ['circle',   {'radius':self.r}],
                        ['circle',   {'radius':self.r}]]       
        
        
    def change_orientation(self):    
        if(self._orientation == 1):
            self._orientation = -1
        else:
            self._orientation = 1
               
        self.arrow_length =  self.initial_arrow_length    
        self._ve_arrow.set(orientation = (self._ve_arrow.parameters.orientation + 180)%360);            
                 
                
    def setClassifierOutput(self, cl_name,cl_value):
        if((self._ve_arrow.parameters.on) or ( self.ONLY_POWER_BAR)):
            
            if(cl_name == 'cfy_idle'):
                
                if(cl_value < -1.5):
                    cl = -1.5
                elif(cl_value > 1.5):
                    cl = 1.5
                else:
                    cl = cl_value
                    
                self.idle_value = 1-(cl + 1.5)/3;
                self._msg_idle.set(text=('IDLE: %.2f' % self.idle_value))
            else:                    
                self.value =  -1*cl_value
                    
                #if(self.value > 0):
                #    print("%s cl = %.2f ; power lenght= %.2f ) " % (self.classes[1], self.value, self.power_length))
                #else:
                #    print("%s cl = %.2f ; power lenght= %.2f ) " % (self.classes[0], self.value, self.power_length))
               
               
    def detectCollision(self):
        if(self.arrow_length > 0.6):
            p = self._ve_arrow.arrowpoint
                   
            for i in xrange(self._nr_elements_level):
                c = self._ve_shapes[i].parameters.position
                r = self._ve_shapes[i].parameters.radius
                
                #print("%d : arr = (%.2f - %.2f),  pos = (%.2f - %.2f) rad = %.2f, %.2f : %.2f\n" % (i, p[0], p[1],  c[0], c[1], r, (pow((p[0]-c[0]), 2) + pow((p[1]-c[1]), 2) ) , pow(r,2)))
                if ( (pow((p[0]-c[0]), 2) + pow((p[1]-c[1]), 2) ) < pow(r,2) ):
                    print("COLLISION!!!!!!!!!")
                    return i                         
            
        return -1        
    
    
    def get_shapes(self):
        return self.shapes
    
    
    def update_powerbar(self):
            
        v = self.value * self.idle_value;        

        if((self.power_length + (1.0 /self.powerbar_growth_time) * v) <0):
            self.power_length = 0;
        elif ((self.power_length + (1.0 /self.powerbar_growth_time) * v) > self.power_maxsize):
            self.power_length = self.power_maxsize
        else:
            self.power_length += (1.0 /self.powerbar_growth_time) * v                        

        return self._ve_powerbar.setValue(self.power_length)
        
               
                
    def update(self,dt):      
        
        r = -1                
        #print("update: %.2f, value = %.2f" % (dt, self.value))
        if self.ONLY_POWER_BAR:
            v = self.update_powerbar()
        else:     
            if(self._ve_arrow.parameters.on):
                if(self._nr_elements_to_choose == 2):
                                                
                    r = self.increase_arrow(dt)                           
                else:                
                    v = self.update_powerbar()
                                   
                    if(v>0):                
                        r = self.increase_arrow(dt)               
                    elif (v<0):
                        r = self.rotate_arrow(dt)
                        self.value -= self.anti_rotation_bias
                    else:
                        self.decrease_arrow()
        self.value = self.value * self.update_coef
        return r
    
    
    def increase_arrow(self, dt):
           
        r = -1
        
        if(self.arrow_length < 1.0):
            
            if(self._nr_elements_to_choose == 2):   
                self.arrow_length += (1.0 /self.powerbar_growth_time) * self.value * self._orientation            
                if(self.arrow_length>1.0):
                    self.arrow_length = 0.999
                    
                if(self.arrow_length >= self.initial_arrow_length):
                    self._ve_arrow.set(size=(self._ve_arrow.parameters.size[0], self.arrow_length*self.arrow_max_length))
                else:
                    self.change_orientation()                                                
            else:                                                        
                self.arrow_length += (1.0 /self.arrow_growth_time) * dt                
                self._ve_arrow.set(size=(self._ve_arrow.parameters.size[0], self.arrow_length*self.arrow_max_length))
       
        self._ve_arrow.update_arrow_point();                
        r = self.detectCollision()

        return r
        
        
    def decrease_arrow(self):          
        if(self.arrow_length >= self.initial_arrow_length):
            self.arrow_length -= 0.007
            self._ve_arrow.set(size=(self._ve_arrow.parameters.size[0], self.arrow_length*self.arrow_max_length))          
            self._ve_arrow.update_arrow_point();            
    
    
    def rotate_arrow(self,dt):
        phi = self._ve_arrow.parameters.orientation        
        phi += (360.0 / self.arrow_rotation_time) * dt
        phi = phi % 360.0
        self._ve_arrow.set(orientation = phi);
        self._ve_arrow.update_arrow_point();
        return self.detectCollision()


    def set_on(self, b):
        
        self._orientation = -1
        
        if(b == False): 
            if(self._nr_elements_to_choose == 2):
                self.random_pos()
        
        if (not self.ONLY_POWER_BAR):    
            LevelBase.set_on(self,b)

            if(self._nr_elements_to_choose == 2):
                self.arrow_length = self.initial_arrow_length            
            else:
                self.arrow_length = self.initial_arrow_length
            
            self._ve_arrow.set(size=(self._ve_arrow.parameters.size[0], self.arrow_length*self.arrow_max_length))
                
        if(b == False):            
            self.show_arrow(b)
            if(self._nr_elements_to_choose == 2):  
                self._ve_arrow.set(orientation = self.__init_orientation, size=(20.0, self.min_length)  )                                                                         
                               
            if(self._nr_elements_to_choose != 2):                   
                self._ve_powerbar.resetValue()
                self._ve_arrow.set(orientation = self.__init_orientation)
                
            self.power_length = 400

        if(self._nr_elements_to_choose != 2):
            self._ve_powerbar.set(on=b)  

        self._message1.set(on=b)
        self._message2.set(on=b)
        
        
    def show_arrow(self,b):
        
        if (not self.ONLY_POWER_BAR): 
            self._ve_arrow.set(on=b)              
            self.value = 0
        
        
    def start_up_shapes(self, visual, center, colors):

        LevelBase.start_up_shapes(self,visual, center, colors)            
        
        t1 = self.classes[1] 
        t2 = self.classes[0]

        self.center = center
        
        if(self._nr_elements_to_choose == 2):
            x1 = center[0] + self.radius
            x2 = center[0] - self.radius
            
            y1 = center[1] + 150
            y2 = y1            
                     
        else:
                                      
            t1 = t1 + " (turning)"
            t2 = t2 + " (growing)"
          
          

            if (self.ONLY_POWER_BAR):
                offset = 50
                y1 = center[1] - 350
                y2 = center[1] - 350
                
            else:  
                offset = 250
                y1 = center[1] - 530
                y2 = center[1] - 530
            
            x1 = center[0] + self.power_maxsize/2 + 40
            x2 = center[0] - self.power_maxsize/2 - 40

              
            self._ve_powerbar =  self.registered_shapes['PowerBar'](  position=[center[0]+self.power_maxsize/2, center[1]-offset],
                                                                      screen_center =[center[0], center[1]],
                                                                    color=colors[0],
                                                                    on=False,
                                                                    **self.shape_powerbar)
            
            visual.append(self._ve_powerbar)
                        
        self._message1 =   Text( position=(x1, y1),
                                   text=t1,
                                   font_size=30,
                                   color=[0.8, 0.45, 1],
                                   anchor='center', 
                                   on=False)
        
        self._message2 =   Text( position=(x2, y2 ),
                                   text=t2,
                                   font_size=30,
                                   color=[0.8, 0.45, 1],
                                   anchor='center', 
                                   on=False)
                
        self._msg_idle =   Text( position=(x1 + 50, y1 - 50 ),
                                   text="IDLE: 1",
                                   font_size=50,
                                   color=[0.8, 0.45, 1],
                                   anchor='center', 
                                   on=False)
     
        self._ve_arrow = self.registered_shapes['arrow'](  position=center,
                                                                    color=colors[0],
                                                                    on=False,
                                                                    **self.shape_arrow)
          
            
        if(self._nr_elements_to_choose == 2):
            self.__init_orientation = 90.0   
        else:         
            self.__init_orientation = 0.0
                         
        visual.append(self._ve_arrow)
        visual.append(self._message1)
        visual.append(self._message2)        
        visual.append(self._msg_idle)                     



