"""PolygonStimulus and helper functions"""

import VisionEgg
import VisionEgg.ParameterTypes as ve_types
from VisionEgg.Core import *
from math import *
import OpenGL.GL as gl
import numpy as np
import pylab as p
# used to display text
import OpenGL.GLUT as glut
import pylab as py
from VisionEgg.Text import *


class Figure_Many(Stimulus):             
    """Displays symmetric figure - with letters"""
    
    #defines the parameters to be used by figure
    parameters_and_defaults = {
        'time_passed':(0.0,
            ve_types.Real),
        'color_on':((1.0,1.0,1.0),
            ve_types.AnyOf(ve_types.Sequence3(ve_types.Real))),
        'color_off':((0.0,0.0,0.0),
            ve_types.AnyOf(ve_types.Sequence3(ve_types.Real))),
        'center_position':(( 320.0, 240.0 ),
            ve_types.AnyOf(ve_types.Sequence2(ve_types.Real))),
        'radius_size':(64.0,
            ve_types.Real),
        'corners_number':(6,
            ve_types.Integer),
        'frequency_used':(3.0,
            ve_types.Real),
        'letters_number':(5,
            ve_types.Integer),
        'font_size':(18,
            ve_types.Real),
        'empty_letter_spot':(0,
            ve_types.Integer), 
        'letters_to_use':('ABCDE',
            ve_types.String),
        'animate': (0,
            ve_types.Real),
        'screen_center':(( 320.0, 240.0 ),
            ve_types.AnyOf(ve_types.Sequence2(ve_types.Real)))}
          
         
    def __init__(self, **kw):
        """ initializes the figure """
        VisionEgg.Core.Stimulus.__init__(self,**kw)
        
        # Calculate points
        corners = self.parameters.corners_number
        radius = self.parameters.radius_size
        offset = 360 / (corners * 2) # If there is two parallel sides, we want them horizontal

        self.many_letters_font_size = 50
        self.color_letters_off = self.parameters.color_on
        #initialize letters  
        distance_between = 15
        self.letters_position = self.calculate_center_of_each_letter(len(self.parameters.letters_to_use), self.parameters.font_size, distance_between, self.parameters.center_position)    
        self.letters_to_use = self.parameters.letters_to_use
        
    def draw(self):
        """ draws the figure into the screen """
        p = self.parameters     
        corners = self.parameters.corners_number
        radius = self.parameters.radius_size
        offset = 360 / (corners * 2) # If there is two parallel sides, we want them horizontal
        
        gl.glPushMatrix()
        gl.glLoadIdentity()
        corners = self.parameters.corners_number
        radius = self.parameters.radius_size   
        if int(p.time_passed * p.frequency_used * 2.0) % 2: 
            self.colors = p.color_on
            self.colors_font = p.color_off
        else:
            self.colors = p.color_off
            self.colors_font = p.color_on
        
        #when it will already work, this must be removed
        self.center = p.center_position
        if p.animate > 0.0:
            self.animate_figure()                
        else:    
            self.center = p.center_position
        #gives the graphical card colors and draws it to the screen
        gl.glColor3f(self.colors[0], self.colors[1], self.colors[2])
        
        gl.glTranslate(self.center[0], self.center[1], 0.0)
        angles = [radians(v) for v in range(offset, 360 + offset , 360 / corners)]
        self.points = [(radius * sin(v), radius * cos(v)) for v in angles ]           
        gl.glBegin(gl.GL_POLYGON)
        for (x, y) in self.points: 
            gl.glVertex3f(x, y, 0.0)        
        gl.glEnd()
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
        gl.glPopMatrix()
        
        gl.glColor3f(self.colors_font[0], self.colors_font[1], self.colors_font[2])
        gl.glPushMatrix()
        
        if p.font_size <= 10:
            font_to_use = glut.GLUT_BITMAP_HELVETICA_10   
        elif p.font_size <= 12:
            font_to_use = glut.GLUT_BITMAP_HELVETICA_12      
        elif p.font_size <= 13:
            font_to_use = glut.GLUT_BITMAP_8_BY_13            
        elif p.font_size <= 15:
            font_to_use = glut.GLUT_BITMAP_9_BY_15                               
        elif p.font_size <= 18:
            font_to_use = glut.GLUT_BITMAP_HELVETICA_18               
        else:
            font_to_use = glut.GLUT_BITMAP_TIMES_ROMAN_24    
        
        
        for i in range (p.letters_number + 1):
            gl.glRasterPos2f(self.letters_position[i][0], self.letters_position[i][1]);
            glut.glutBitmapCharacter(font_to_use, ord(self.letters_to_use[i])) 
        gl.glPopMatrix() 
        
        
    def calculate_center_of_each_letter(self, number_of_figures, radius_of_figure, distance_between, center):
        """ calculates the postion of each letter - done only in the initial state """
        start_at = np.pi/2
        
        if number_of_figures == 1:
            return(self.screen_size[0]/2, self.screen_size[1]/2)
        else:
            angle_between = 2 * np.pi / number_of_figures
            circle_size = (radius_of_figure + distance_between) * number_of_figures
            radius = circle_size/ (np.pi*2)
            all_positions = []
            a = center[0]
            b = center[1]
            for i in range(0, number_of_figures):
                all_positions.append((a + radius *p.cos(i * angle_between + start_at) - (0.25 * radius_of_figure),b + radius *p.sin(i * angle_between + start_at) - (0.25 * radius_of_figure))) #u must move this point to the correct location
            return all_positions
        
    def animate_figure(self):
        p = self.parameters
        #if it is time to show this figure   
        fraction = 18     
        if p.time_passed > ((p.animate/fraction)*(p.empty_letter_spot)):
            self.colors = p.color_on
            self.colors_font = p.color_off
            next_figure = ((p.animate/fraction) * (p.empty_letter_spot + 1))
            #if this is time to animate this figure
            if p.time_passed <= next_figure - 0.3 * (p.animate/fraction):
                self.center = p.screen_center
            elif p.time_passed <= next_figure:
                a1 = (p.screen_center[1] - p.center_position[1])
                a2 = (p.screen_center[0] - p.center_position[0])
                my_time = p.time_passed - p.empty_letter_spot * (p.animate/fraction)
                if np.abs(a2) < 0.001:
                #if the figure is on the vertical line with the center
                    x = p.screen_center[0]
                    y = p.screen_center[1] + (p.center_position[1]-p.screen_center[1]) * (my_time)/p.animate*6
                else:  
                    a = (a1/a2)
                    b = p.center_position[1] - a * p.center_position[0]
                    x = p.screen_center[0] + (p.center_position[0]-p.screen_center[0]) * (my_time)/p.animate*6 
                    y = a * x + b
                self.center = (x, y)
            else:
                self.center = p.center_position
        else:
            self.colors = (0.5,0.5,0.5)
            self.colors_font = (0.5,0.5,0.5)
            self.center = p.center_position
                
        self.letters_position = self.calculate_center_of_each_letter(len(self.parameters.letters_to_use), self.parameters.font_size, 15, self.center)    
        #self.letters_to_use = self.parameters.letters_to_use
