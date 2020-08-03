from __future__ import division  

import math 
from lib.P300Layout.CircularLayout import CircularLayout
import numpy as NP
from VisionEgg.Text import Text
from VisionEgg.Textures import *

OK = 1
CANCEL = 2
ELEM = 3
BACK = 4
USER_DEFINED = 5
                              
rP = 15  # Rounding Function Precision, numbers after the decimal

##---------------------------------------------------------------------------------------

class MenuElementText(Text):
    def __init__(self, parent, font_size, color, text, position, anchor, on = True):
        
        Text.__init__(self, on=on, font_size=font_size, color=color, text=text, position=position, anchor=anchor)        
        
        self.original_pos = position
        self.order = 0
        self.letter = text
        self.parent = parent
        self.subelems = []
        self._letters = []
        self._vegg_elem = None
        self.countdown_position = (0,0)
        self.path = "dddd"
        
    def reset_position(self):
        self.position= self.original_pos
    
    def set_countdown_position(self, pos):
        self.countdown_position= pos    
                           
    def set_all_on(self,on):   
        if (self.letter is not ""):
            self.set(on=on)       
                 
        for i in xrange(len(self.subelems)):
            self.subelems[i].set_all_on(on)      
        
    def num_elements(self):
        return len(self.subelems)

    def append(self, elem):
        elem.order = len(self.subelems)
        self.subelems.append(elem)
        self._letters.append(elem.letter_set())        
        
    def get_text(self):
        return self.letter
    
    def letter_set(self):
        return self._letters
        
    def startup(self, radius):
        self.letter_layout = CircularLayout( nr_elements=self.num_elements(),
                                             radius=radius,
                                             start=NP.pi/6.*5)
        self.letter_layout.positions.reverse()    


class MenuElementImage(object):
    
    def __init__(self, id, parent, position, texture):
        
        self.id = id
         
        self.texture = texture
        self.texture.set(position = position, on = False)
 
        self.on = False
        self.text =""
        self.parent = parent
        self.original_pos = position
    
    def set_max_alpha(self,a):
        self.texture.set(max_alpha=0.4)
            
    def set_all_on(self,on):       
        self.texture.set(on=on)    
        
    def get_text(self):
        return "#"     
    
    def reset_position(self):
        self.position= self.original_pos
    

                           
                           
class FooMenuElement(object):
    
    def __init__(self, letter, parent):
        self.order = 0
        self.letter = letter
        self.parent = parent
        self.subelems = []
        self._vegg_elem = None
        
        if(letter is not ""):
            self._letters.append(letter)
    
    def set_all_on(self,on):
        
        for i in xrange(len(self.subelems)):
            self.subelems[i].set_all_on(on)          
                
    def num_elements(self):
        return len(self.subelems)

    def append(self, elem):
        elem.order = len(self.subelems)
        self.subelems.append(elem)
        
    def get_text(self):
        return self.letter
    
        
    def startup(self, radius):
        self.letter_layout = CircularLayout( nr_elements=self.num_elements(),
                                             radius=radius,
                                             start=NP.pi/6.*5)
        self.letter_layout.positions.reverse()
        
class MenuElement(object):  
    
        def __init__(self, container, parent, radius, color, type=ELEM, active=False):
            self.type = type                
            self.radius = radius
            self.color = color
            self.orig_color = color
            self.center = (0,0)
            self.subelems = []
            self.active = active
            self.container = container
            self.maxSize = radius
            self.parent = parent
                
        def append(self, elem):
            self.subelems.append(elem)      
             
        def drawChildren(self):              
            for elem in self.subelems:
                elem.draw()
                        
        def draw(self):
            print("draw")
            #pygame.draw.circle(self.container.screen, self.color, (self.center), self.radius)                                
                                       
        def detectColision(self,x,y):
                centerX, centerY = self.center
                return (  ((x-centerX)*(x-centerX) + (y - centerY)*(y - centerY) ) < self.radius*self.radius )

        def setCenter(self, point):
                self.center = point

        def detectColisions(self, (x, y)):
                activated = None
                for elem in self.subelems:
                        if(elem.detectColision(x,y)):
                                print "colision!: ", elem.name
                                elem.activate()
                                activated = elem
                        else:
                                elem.deactivate()
                return activated
        
        def activate(self):
                self.active = True
                self.color = (200,0,0)
                
                if((self.type == BACK) or (self.type == CANCEL)):
                    return self.parent.parent
                elif(self.type== USER_DEFINED):
                    command="hola"
                    try:                    
                        command_module = __import__("myapp.commands.%s" % command, fromlist=["myapp.commands"])
                    except ImportError:
                        # Display error message

                        command_module.run()
                else:
                    return self
                
        def deactivate(self):
                self.active = False
                self.color = self.orig_color

        def updateSubelements(self):
                if len(self.subelems) == 0: 
                    return
                   
                angle = int(360/len(self.subelems))   
                angle = math.radians(angle)
                
                cT = math.cos(angle)
                sT = math.sin(angle) 
                
                (x, y) = self.container.center
                  
                pX = round(x-self.container.menuRadius, rP)
                pY = round(self.container.height/2, rP)                              
                           
                for i in range(len(self.subelems)):
                    cT = math.cos(angle*i)
                    sT = math.sin(angle*i)
                    
                    newX = (pX - x) * cT - (pY -y) * sT
                    newY = (pX - x) * sT + (pY -y) * cT
                                                                         
                    self.subelems[i].setCenter( [round(newX + x, rP), round(newY + y, rP)] )
                                                        
class TextElement(MenuElement):
    def __init__(self, name, container, parent, radius, color, type=ELEM, active=False):
        MenuElement.__init__(self, container, parent, radius, color, type, active)
        self.text= name     
                       
    def draw(self):
        MenuElement.draw(self)
        x, y = self.center  
        towrite = self.text
        #FIXME font = pygame.font.Font(None, 50)
        #text = font.render(towrite, True, (255, 255, 255), self.color)

        # Create a rectangle
        #textRect = text.get_rect()

        # Center the rectangle
        #textRect.centerx = x
        #textRect.centery = y        
                                        
        #self.container.screen.blit(text, textRect)

        
class ImageElement(MenuElement):   
    def __init__(self, img, container, parent, radius, color, type=ELEM, active=False):
        MenuElement.__init__(self, container, parent, radius, color, type, active)
        self.img = img
        
    def draw(self):               
        MenuElement.draw(self)
        x, y = self.center  
        self.container.screen.blit(self.img, (x - self.maxSize/2, y - self.maxSize/2))
        
 ##---------------------------------------------------------------------------------------
class MenuElementUserDefined(TextElement):  
    def activate(self):
        print "MENU ELEMENT USER DEFINED :: ACTIVATE !!!!!!!!!"
        
        try:                    
            command_module = __import__("%s" % self.command, fromlist=["test_classes"])
            command_module.activate()            
        except ImportError:
            # Display error message
            print("ERROR importing file %s" % self.command)
        
        return None
                