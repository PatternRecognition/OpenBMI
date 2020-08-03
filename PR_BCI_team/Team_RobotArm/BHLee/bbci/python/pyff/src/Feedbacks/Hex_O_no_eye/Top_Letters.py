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
        'special_color':((1.0,0,0),
            ve_types.AnyOf(ve_types.Sequence3(ve_types.Real))),
        'center_position':(( 320.0, 240.0 ),
            ve_types.AnyOf(ve_types.Sequence2(ve_types.Real))),
        'character':('a',
            ve_types.String),
        'my_number':(0,
            ve_types.Integer),
        'letter_no_used': (0,
            ve_types.Integer)}
     
    def __init__(self, **kw):
        """initializes the top text"""
        VisionEgg.Core.Stimulus.__init__(self,**kw)
        
    def draw(self):
        """draws text on the top"""
        p = self.parameters     
        gl.glPushMatrix()
        if p.my_number == p.letter_no_used:
            color = p.special_color
        else:
            color = p.color
        gl.glColor3f(color[0], color[1], color[2])
        font_to_use = glut.GLUT_BITMAP_HELVETICA_18           
        gl.glRasterPos2f(p.center_position[0], p.center_position[1]);
        glut.glutBitmapCharacter(font_to_use, ord(p.character)) 
        gl.glPopMatrix() 
        
