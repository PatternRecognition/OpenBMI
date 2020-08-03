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

class Top_Letters(Stimulus):             
    """Displays the text on the top of the screen"""
    #defines the parameters to be used by figure
    parameters_and_defaults = {
        'color':((1.0,1.0,1.0),
            ve_types.AnyOf(ve_types.Sequence3(ve_types.Real))),
        'position':(( 320.0, 240.0 ),
            ve_types.AnyOf(ve_types.Sequence2(ve_types.Real))),
        'text':("a",
            ve_types.String)}
     
    def __init__(self, **kw):
        """initializes the top text"""
        VisionEgg.Core.Stimulus.__init__(self,**kw)
        self.letters = self.parameters.text
    
    def set_my_letters(self, letters):
        self.letters = letters  
          
    def draw(self):
        """draws text on the top"""
        p = self.parameters     
        gl.glPushMatrix()

        color = p.color
        gl.glColor3f(color[0], color[1], color[2])
        font_to_use = glut.GLUT_BITMAP_HELVETICA_18           
        gl.glRasterPos2f(p.position[0], p.position[1]);
        for i in range(len(self.letters)):
            gl.glRasterPos2f(p.position[0] + (i * 15), p.position[1]);
            glut.glutBitmapCharacter(font_to_use, ord(self.letters[i])) 
            

        #glut.glutB
        gl.glPopMatrix() 
        
