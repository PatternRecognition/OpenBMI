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

#from LoadImage import *

import numpy.oldnumeric as Numeric, numpy.oldnumeric.random_array as RandomArray
import math, types, string

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
    
    def init(self, number = 1, no_of_all = 4, format = (600, 600), pixel_size = 1, use_solid = False):
        self.format = format     
        self.number = number
        self.no_of_all = 4 #no_of_all   
        self.pixel_size = pixel_size
        self.use_solid = use_solid
        
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
            s = 0
            g = 0
            y = self.format[0] - 1
            while y > -1:
            #for y in range (self.format[0]):
                if self.use_solid:
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

                else:                
                    for x in range (self.format[1]):
                        if img[k] > 128:
                            imgarray[x][y] = 0
                        else:
                            imgarray[x][y] = 1
                            #if ((y % 2) == (self.number % 2) and k % 2 == 0):
                            #statement1 = ((y % (self.pixel_size * 2)) == (self.number % 2) or ((y-1)% (self.pixel_size * 2))== (self.number % 2))
                            #box_size = (self.pixel_size * (self.number % 2) + 1)
                            #statement1 = ((y > box_size - self.pixel_size and y < box_size))
                            #statement2 = (((k-1)% box_size) == (self.number % box_size) or (k % box_size) == (self.number % box_size))
                            #box_size = (self.pixel_size * int(self.number / 2) + 1)
                            #statement2 =   ((y > box_size - self.pixel_size and y < box_size))
                        
                            statement1 =  s == (self.number % 2)
                            statement2 =  g == int((self.number -1)/ 2) 
                            if statement1 and statement2:
                                xs.append(x)
                                ys.append(y)
                        k = k + 1
                        if (k % self.pixel_size) == 0:
                            s = (s + 1) % 2
                    y = y - 1
                    if (y % self.pixel_size) == 0:
                        g = (g + 1) % 2
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
    
class Points(VisionEgg.Core.Stimulus):
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
        'on' : ( True,
                 ve_types.Boolean ),
        'position' : ( ( 320.0, 240.0 ), # in eye coordinates
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
        'frequency_used':(3.0, ve_types.Real), 
        'time_passed':(0.0,
            ve_types.Real),
        'pixel_size':(4.0, # pixels
                      ve_types.Real),
        'use_solid':(True,
                            ve_types.Boolean ),
        
        }

    constant_parameters_and_defaults = {
        'num_dots' : ( 100,
                       ve_types.UnsignedInteger ),
        }


    def __init__(self, **kw):
        VisionEgg.Core.Stimulus.__init__(self,**kw)
        # store positions normalized between 0 and 1 so that re-sizing is ok
        num_dots = self.constant_parameters.num_dots # shorthand
        sd = LoadImage()
        sd.init(number = self.parameters.number, no_of_all = self.parameters.no_of_all,format = self.parameters.size, pixel_size = self.parameters.pixel_size, use_solid = self.parameters.use_solid)
        image_arrays, xs, ys = sd.convert([self.parameters.picture])

        self.x_positions, self.y_positions =np.array(xs), np.array(ys)
        self.last_time_sec = VisionEgg.time_func()
        self.start_times_sec = None # setup variable, assign later
        self._gave_alpha_warning = 0
        
        self.x_margin =  self.parameters.position[0] - (self.parameters.size[0] /2)
        self.y_margin =  self.parameters.position[1] - (self.parameters.size[1] /2)
        
        self.xs = self.x_positions + self.x_margin
        self.ys = self.y_positions + self.y_margin
        self.zs = (0.0,)*len(xs) # make N tuple with repeat value of depth
        #self.vertices = np.array([self.xs, self.ys, self.zs]).astype(np.int32).T
        #self.n_vertices = len(self.xs)
        
        
        self.gl_list = gl.glGenLists(1)
        gl.glNewList(self.gl_list, gl.GL_COMPILE)
        draw_dots(self.xs, self.ys, self.zs)
        gl.glEndList()
   
        

    def draw(self):
        # XXX This method is not speed-optimized. I just wrote it to
        # get the job done. (Nonetheless, it seems faster than the C
        # version commented out above.)
        p = self.parameters # shorthand
        if int(p.time_passed * p.frequency_used * 2.0) % 2: 

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
                    #draw_dots(self.xs, self.ys, self.zs)
                #stop  = time.time()
                gl.glCallList(self.gl_list)
                #print (stop-start)/2000.*1000.
                #draw_dots2(self.vertices, self.n_vertices)
                #draw_dots(self.xs, self.ys, self.zs)
                #gl.glPopMatrix()
