# The Vision Egg: Dots
#
# Copyright (C) 2001-2003 Andrew Straw. (modified by Maria Kramarek 2010)
# Copyright (C) 2005,2008 California Institute of Technology
#
# URL: <http://www.visionegg.org/>
#
# Distributed under the terms of the GNU Lesser General Public License
# (LGPL). See LICENSE.TXT that came with this file.

"""
Chosen position dot stimuli.

"""

####################################################################
#
#        Import all the necessary packages
#
####################################################################

import logging
import time

import VisionEgg
import VisionEgg.Core
import VisionEgg.ParameterTypes as ve_types
import OpenGL.GLUT as glut
#from LoadImage import *

import numpy.oldnumeric as Numeric, numpy.oldnumeric.random_array as RandomArray
import math, types, string
import pylab as pyl
import VisionEgg.GL as gl # get all OpenGL stuff in one namespace

### C version of draw_dots() isn't (yet) as fast as Python version:
##import VisionEgg._draw_in_c
##draw_dots = VisionEgg._draw_in_c.draw_dots # draw in C for speed

import Image
import ImageEnhance
import numpy as np

#this class converts the image to the array consisting of 0s and 1s, where 0 means light colors in the 
#given image and 1 means dark colors. The images must be given
class LoadImage():
    
    def init(self, number = 1, no_of_all = 4, format = (600, 600)):
        self.format = format     
        self.number = number
        self.no_of_all = 4 #no_of_all   
        
    def convert(self, im_load):
        image_arrays = []
        xs = []
        ys = []    
        for i in im_load:
            im = Image.open(i)
            #print "format:", im.format, ", size:", im.size, ", mode:", im.mode
            if not im.mode == self.format:
                imgarray = np.zeros(self.format)
            #resizes image to wanted format
            im = im.resize(self.format)

            #converts image to black and white
            im = im.convert("L")

            img =  list(im.getdata())

            k = 0
            y = self.format[0] - 1
            while y > -1:
            #for y in range (self.format[0]):
                for x in range (self.format[1]):
                    if img[k] > 128:
                        imgarray[x][y] = 0
                    else:
                        imgarray[x][y] = 1
                        if ((y % 2) == (self.number % 2) and k % 2 == 0):
                            xs.append(x)
                            ys.append(y)
                    k = k + 1
                y = y - 1
            image_arrays.append(imgarray)
            
        return image_arrays, xs, ys

def draw_dots(xs,ys,zs):
    """Python method for drawing dots.  May be replaced by a faster C version."""
    if not (len(xs) == len(ys) == len(zs)):
        raise ValueError("All input arguments must be same length")
    gl.glBegin(gl.GL_POINTS)
    for i in xrange(len(xs)):
        gl.glVertex3f(xs[i],ys[i],zs[i])
    gl.glEnd()
    
class Animate(VisionEgg.Core.Stimulus):
    """

    Parameters
    ==========
    anchor                  -- (String)
                               Default: center
    anti_aliasing           -- (Boolean)
                               Default: True
    color                   -- (AnyOf(Sequence3 of Real or Sequence4 of Real))
                               Default: (1.0, 1.0, 1.0)
    depth                   -- (Real)
                               Default: (determined at runtime)
    #dot_lifespan_sec        -- (Real)
                               Default: 5.0
    dot_size                -- (Real)
                               Default: 4.0
    on                      -- (Boolean)
                               Default: True
    position                -- (Sequence2 of Real)
                               Default: (320.0, 240.0)
    signal_direction_deg    -- (Real)
                               Default: 90.0
    signal_fraction         -- (Real)
                               Default: 0.5
    size                    -- (Sequence2 of Real)
                               Default: (300.0, 300.0)
    velocity_pixels_per_sec -- (Real)
                               Default: 10.0

    Constant Parameters
    ===================
    num_dots -- (UnsignedInteger)
                Default: 100
    """

    parameters_and_defaults = {
        'starting_position' : ( ( 320.0, 240.0 ), # in eye coordinates
                       ve_types.Sequence2(ve_types.Real)),
        'ending_position' : ( ( 320.0, 240.0 ), # in eye coordinates
                       ve_types.Sequence2(ve_types.Real) ),
        'anchor' : ('center',
                    ve_types.String),
        'size' :   ( ( 300.0, 300.0 ), # in eye coordinates
                     ve_types.Sequence2(ve_types.Real) ),
        'signal_fraction' : ( 0.5,
                              ve_types.Real ),
        'signal_direction_deg' : ( 90.0,
                                   ve_types.Real ),
        'velocity_pixels_per_sec' : ( 10.0,
                                      ve_types.Real ),
        'dot_lifespan_sec' : ( 25.0,
                               ve_types.Real ),
        'color' : ((1.0,1.0,1.0),
                   ve_types.AnyOf(ve_types.Sequence3(ve_types.Real),
                                  ve_types.Sequence4(ve_types.Real))),
        'dot_size' : (4.0, # pixels
                      ve_types.Real),
        'anti_aliasing' : ( True,
                            ve_types.Boolean ),
        'depth' : ( None, # set for depth testing
                    ve_types.Real ),
        'center' : (None,  # DEPRECATED -- don't use
                    ve_types.Sequence2(ve_types.Real),
                    "",
                    VisionEgg.ParameterDefinition.DEPRECATED),
        'number' : ( 1,
                               ve_types.Integer ),
        'no_of_all':( 1,
                               ve_types.Integer ),
        'picture': ( "A.JPG",
                               ve_types.String ),
        'time_animation':(3.0, ve_types.Real), 
        'time_before':(3.0, ve_types.Real), 
        'time_passed':(0.0,
            ve_types.Real),
        'display': (  ( "a", "b", "c", "d" ), # in eye coordinates
                       ve_types.Sequence4(ve_types.String)),
        'level':(1,
                               ve_types.Integer ),
        'letter_position':(( 320.0, 240.0 ), # in eye coordinates
                       ve_types.Sequence2(ve_types.Real))
        
        }

    constant_parameters_and_defaults = {
        'num_dots' : ( 100,
                       ve_types.UnsignedInteger ),
        }


    def __init__(self, **kw):
        VisionEgg.Core.Stimulus.__init__(self,**kw)
        p = self.parameters
        # store positions normalized between 0 and 1 so that re-sizing is ok
        num_dots = self.constant_parameters.num_dots # shorthand
        sd = LoadImage()
        sd.init(number = self.parameters.number, no_of_all = self.parameters.no_of_all,format = self.parameters.size )
        image_arrays, xs, ys = sd.convert([self.parameters.picture])

        self.x_positions, self.y_positions =np.array(xs), np.array(ys)
        self.last_time_sec = VisionEgg.time_func()
        self.start_times_sec = None # setup variable, assign later
        self._gave_alpha_warning = 0
        
        self.x_margin =  self.parameters.starting_position[0] - (self.parameters.size[0] /2)
        self.y_margin =  self.parameters.starting_position[1] - (self.parameters.size[1] /2)
        
        self.xs = self.x_positions+ self.x_margin
        self.ys = self.y_positions + self.y_margin
        self.zs = (0.0,)*len(xs) # make N tuple with repeat value of depth
        self.old_xs = self.xs
        self.old_ys = self.ys
        #self.vertices = np.array([self.xs, self.ys, self.zs]).astype(np.int32).T
        #self.n_vertices = len(self.xs)
        
        
        self.gl_list = gl.glGenLists(1)
        gl.glNewList(self.gl_list, gl.GL_COMPILE)
        draw_dots(self.xs, self.ys, self.zs)
        gl.glEndList()
        self.step_size = [((p.ending_position[0] - p.starting_position[0])), ((p.ending_position[1] - p.starting_position[1]))]
   
        

    def draw(self):
        # XXX This method is not speed-optimized. I just wrote it to
        # get the job done. (Nonetheless, it seems faster than the C
        # version commented out above.)
        p = self.parameters # shorthand
        if p.time_passed == 0.0:
            self.xs = self.old_xs
            self.ys = self.old_ys    

        #if int(p.time_passed * p.frequency_used * 2.0) % 2: 

#                if len(p.color)==3:
        gl.glColor3f(*p.color)
#                elif len(p.color)==4:
#                    gl.glColor4f(*p.color)
        gl.glPointSize(p.dot_size)
        # Clear the modeview matrix
        
        #
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glDisable(gl.GL_TEXTURE_2D)
        
        #gl.glPushMatrix()
        #start = time.time()
        
        #for i in xrange(2000):
            #gl.glCallList(self.gl_list)
            #draw_dots2(self.vertices, self.n_vertices)
        if p.time_passed > p.time_before:
            self.xs = self.old_xs + (self.step_size[0]*((p.time_passed - p.time_before)/p.time_animation))
            self.ys = self.old_ys + (self.step_size[1]*((p.time_passed - p.time_before)/p.time_animation))
             
        draw_dots(self.xs, self.ys, self.zs)
        
        gl.glColor3f(1.0, 1.0, 1.0)
        
        gl.glPushMatrix()
        
        #if p.font_size <= 10:
        #    font_to_use = glut.GLUT_BITMAP_HELVETICA_10   
        #elif p.font_size <= 12:
        #font_to_use = glut.GLUT_BITMAP_HELVETICA_12      
        #elif p.font_size <= 13:
        #    font_to_use = glut.GLUT_BITMAP_8_BY_13            
        #elif p.font_size <= 15:
        #font_to_use = glut.GLUT_BITMAP_9_BY_15                               
        #elif p.font_size <= 18:
        #    font_to_use = glut.GLUT_BITMAP_HELVETICA_18               
        #else:
        font_to_use = glut.GLUT_BITMAP_TIMES_ROMAN_24    
        
        posesY = [24, 0, -24, -48] 
        posesX = [-24, 0, 24] 
        x = -1
        y = 0

        current_display = p.display[p.number-1]
        for i in range (0, len(current_display)):
            x = x +1
            if x == len(posesX):
                x = 0
                y = y + 1
            gl.glRasterPos2f(p.letter_position[0] + posesX[x], p.letter_position[1] + posesY[y]);
            glut.glutBitmapCharacter(font_to_use, ord(current_display[i]))
        gl.glPopMatrix() 
        
        #stop  = time.time()
        #gl.glCallList(self.gl_list)
        #print (stop-start)/2000.*1000.
        #draw_dots2(self.vertices, self.n_vertices)
        #draw_dots(self.xs, self.ys, self.zs)
        #gl.glPopMatrix()
