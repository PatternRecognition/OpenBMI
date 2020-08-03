'''Feedbacks.VisualSpeller.HexShape
# Copyright (C) 2010  "Nico Schmidt"
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

Created on Mar 11, 2010

@author: "Nico Schmidt"

log:
2010-05-13: Added Hourglass and some other things (Matthias Treder)
2011-02-18: Added FilledCross stimulus (Priska Herger)
'''

import math
from math import cos, sin, radians
import logging
import VisionEgg.Core
import VisionEgg.ParameterTypes as ve_types
from numpy import sqrt

import VisionEgg.GL as gl # get all OpenGL stuff in one namespace

rP = 15  # Rounding Function Precision, numbers after the decimal

class MyArrow(VisionEgg.Core.Stimulus):
   

    parameters_and_defaults = {
        'on':(True,
              ve_types.Boolean),
        'color':((1.0,1.0,1.0),
                 ve_types.AnyOf(ve_types.Sequence3(ve_types.Real),
                                ve_types.Sequence4(ve_types.Real))),
        'anti_aliasing':(True,
                         ve_types.Boolean),
        'orientation':(0.0, # 0.0 degrees = right, 90.0 degrees = up
                       ve_types.Real),
        'position' : ( ( 320.0, 240.0 ), # In eye coordinates
                       ve_types.AnyOf(ve_types.Sequence2(ve_types.Real),
                                      ve_types.Sequence3(ve_types.Real),
                                      ve_types.Sequence4(ve_types.Real)),
                       "units: eye coordinates"),
        'anchor' : ('center',
                    ve_types.String),
        'size':((64.0,16.0), # In eye coordinates
                ve_types.Sequence2(ve_types.Real)),
        }

    __slots__ = VisionEgg.Core.Stimulus.__slots__ + (
        '_gave_alpha_warning',
        )

    def __init__(self,**kw):
        VisionEgg.Core.Stimulus.__init__(self,**kw)
        self._gave_alpha_warning = 0
        self.arrowpoint = self.rotateAndScale(self.parameters.position[0],self.parameters.position[1]+self.parameters.size[1]+30, self.parameters.position,-self.parameters.orientation)
  
  
    def update_arrow_point(self):
        self.arrowpoint = self.rotateAndScale(self.parameters.position[0],self.parameters.position[1]+self.parameters.size[1]+30, self.parameters.position,-self.parameters.orientation)
        

    def rotateAndScale(self, x,y, p,angle):

        angle = radians(angle)
        
        cT = cos(angle)
        sT = sin(angle) 
  
        newX = (x - p[0])*cT - (y - p[1])*sT         
        newY = (x - p[0])*sT + (y - p[1])*cT    
                            
        return((round(newX + p[0], rP), round(newY + p[1], rP)))
   
            
    def draw(self):
        p = self.parameters # Shorthand
        if p.on:
                    
            w = p.size[0]
            h = p.size[1]
                   
            # Calculate center
            #center = VisionEgg._get_center(p.position,p.anchor,p.size)
            center = (p.position[0],p.position[1])
            
            gl.glMatrixMode(gl.GL_MODELVIEW)
            gl.glPushMatrix()            
            gl.glTranslate(center[0],center[1],0.0)
            gl.glRotate(-p.orientation,0.0,0.0,1.0)

            gl.glColor3f(*p.color)
                
            gl.glDisable(gl.GL_DEPTH_TEST)
            gl.glDisable(gl.GL_TEXTURE_2D)
            gl.glBlendFunc(gl.GL_SRC_ALPHA,gl.GL_ONE_MINUS_SRC_ALPHA)
            gl.glEnable(gl.GL_BLEND)
        
            gl.glBegin(gl.GL_QUADS) # Draw Rectangle
            gl.glVertex2f(  w, h)
            gl.glVertex2f( -w, h)
            gl.glVertex2f( -w, 0)
            gl.glVertex2f(  w, 0)                    
            gl.glEnd() # GL_QUADS

            gl.glBegin(gl.GL_TRIANGLES) # Draw Triangle
            gl.glVertex2f(  w+12, h)
            gl.glVertex2f( -w-12, h)
            gl.glVertex2f( 0, h + 30)
            gl.glEnd() # GL_TRIANGLES            


            # Calculate coverage value for each pixel of outline
            # and store as alpha
            gl.glEnable(gl.GL_LINE_SMOOTH)
            # Now specify how to use the alpha value
            gl.glBlendFunc(gl.GL_SRC_ALPHA,gl.GL_ONE_MINUS_SRC_ALPHA)
            gl.glEnable(gl.GL_BLEND)

            # Draw a second polygon in line mode, so the edges are anti-aliased
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK,gl.GL_LINE)
            
            gl.glBegin(gl.GL_QUADS) # Draw Rectangle
            gl.glVertex2f(  w, h)
            gl.glVertex2f( -w, h)
            gl.glVertex2f( -w, 0)
            gl.glVertex2f(  w, 0)                    
            gl.glEnd() # GL_QUADS
            
            gl.glBegin(gl.GL_TRIANGLES)                             
            gl.glVertex2f(  w+12, h)
            gl.glVertex2f( -w-12, h)
            gl.glVertex2f( 0, h + 30)            
            gl.glEnd() # GL_QUADS

            # Set the polygon mode back to fill mode
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK,gl.GL_FILL)
        
            gl.glDisable(gl.GL_LINE_SMOOTH)
            
            gl.glPopMatrix()


class PowerBar2(VisionEgg.Core.Stimulus):
   

    parameters_and_defaults = {
        'on':(True,
              ve_types.Boolean),       
        'color':((0.7,0.2,0.7),
                 ve_types.AnyOf(ve_types.Sequence3(ve_types.Real),
                                ve_types.Sequence4(ve_types.Real))),
        'color_act':((1.0,0.0,1.0),
                 ve_types.AnyOf(ve_types.Sequence3(ve_types.Real),
                                ve_types.Sequence4(ve_types.Real))),            
        'colorBG':((0.0,0.0,0.0),
                 ve_types.AnyOf(ve_types.Sequence3(ve_types.Real),
                                ve_types.Sequence4(ve_types.Real))),       

        'color_frame':((0.4,0.4,0.4),
                     ve_types.AnyOf(ve_types.Sequence3(ve_types.Real),
                                ve_types.Sequence4(ve_types.Real))),
        'screen_center' : ( ( 320.0, 240.0 ), # In eye coordinates
                       ve_types.AnyOf(ve_types.Sequence2(ve_types.Real),
                                      ve_types.Sequence3(ve_types.Real),
                                      ve_types.Sequence4(ve_types.Real)),
                       "units: eye coordinates"),
        'position' : ( ( 320.0, 240.0 ), # In eye coordinates
                       ve_types.AnyOf(ve_types.Sequence2(ve_types.Real),
                                      ve_types.Sequence3(ve_types.Real),
                                      ve_types.Sequence4(ve_types.Real)),
                       "units: eye coordinates"),
        'inisize':((64.0,16.0), # In eye coordinates
                ve_types.Sequence2(ve_types.Real)),
        'size':((64.0,16.0), # In eye coordinates
                ve_types.Sequence2(ve_types.Real)),
        'num_thresholds':(0.0, # 0.0 degrees = right, 90.0 degrees = up
                       ve_types.Real),
        'width_thresholds':(0.0, # 0.0 degrees = right, 90.0 degrees = up
                       ve_types.Real),
        'gravity':(0.6, 
                       ve_types.Real),
        'orientation' : (0.0, 
                         ve_types.Real)         
        }



    def __init__(self,**kw):
        VisionEgg.Core.Stimulus.__init__(self,**kw)
        
        self.th = [];
        p = self.parameters       
        
        self.resetValue()
        
        for i in range(p.num_thresholds+1):
            if(i!=p.num_thresholds/2):
                self.th.append(i * p.inisize[1]/5 + p.inisize[1]/10)       

        self.center = (p.position[0],p.position[1]-p.inisize[1]/2)
        
        
    def resetValue(self):
        self.value = self.parameters.inisize[1]/2
        self.active = False
        self.g = 1 
                 
                 
    def setValue(self,v):
               
        if((v < self.parameters.inisize[1]) and (v > 0)):        
            self.value = v
            
            for i in range(len(self.th)):            
                if((self.value > self.th[i]-self.parameters.width_thresholds) and (self.value < self.th[i]+self.parameters.width_thresholds)):
                    self.active = True            
                    self.g = self.parameters.gravity
                     
                    return 3-i
                else:
                    self.g = 1
                    
        self.active = False
        return -1
            

                
    def draw(self):
        p = self.parameters # Shorthand
        
        if p.on:
         
            iw = p.inisize[0]
            ih = p.inisize[1]                                                                                                
            
            gl.glMatrixMode(gl.GL_MODELVIEW)
            gl.glPushMatrix()
                
            gl.glTranslate(self.center[0],self.center[1],0.0)
            
            gl.glRotate(p.orientation,0.0,0.0,1.0)
                        
            gl.glDisable(gl.GL_DEPTH_TEST)
            gl.glDisable(gl.GL_TEXTURE_2D)
            gl.glBlendFunc(gl.GL_SRC_ALPHA,gl.GL_ONE_MINUS_SRC_ALPHA)
            gl.glEnable(gl.GL_BLEND)
        
            gl.glColor3f(*p.color_frame)
            gl.glBegin(gl.GL_QUADS) # Draw Container 
            gl.glVertex2f(  iw+5, ih+6)
            gl.glVertex2f( -iw-5, ih+6)
            gl.glVertex2f( -iw-5, -6)
            gl.glVertex2f(  iw+5, -6)                    
            gl.glEnd() # GL_QUADS
            
            gl.glColor3f(*p.colorBG)
            
            gl.glBegin(gl.GL_QUADS) # Draw background
            gl.glVertex2f(  iw, ih)
            gl.glVertex2f( -iw, ih)
            gl.glVertex2f( -iw, 0)
            gl.glVertex2f(  iw, 0)                    
            gl.glEnd() # GL_QUADS

            if(self.active):
                gl.glColor3f(*p.color_act)
            else:
                gl.glColor3f(*p.color)
                        
            gl.glBegin(gl.GL_QUADS) # Draw Rectangle
            gl.glVertex2f(  iw-5, ih/2)
            gl.glVertex2f( -iw+5, ih/2)
            gl.glVertex2f( -iw+5, self.value)
            gl.glVertex2f(  iw-5, self.value)                    
            gl.glEnd() # GL_QUADS                
                      
            gl.glColor3f(*p.color_frame)
            gl.glBegin(gl.GL_LINES)
            
            gl.glVertex2f(-iw, ih/2)
            gl.glVertex2f(iw, ih/2)
                        
            for i in range(len(self.th)):
                gl.glVertex2f(-iw, self.th[i]-p.width_thresholds)
                gl.glVertex2f(iw, self.th[i]-p.width_thresholds)
                
                gl.glVertex2f(-iw, self.th[i]+p.width_thresholds)
                gl.glVertex2f(iw, self.th[i]+p.width_thresholds)                
                       
            gl.glEnd()
                      
            gl.glPopMatrix()



class PowerBar(VisionEgg.Core.Stimulus):

    parameters_and_defaults = {
        'on':(True,
              ve_types.Boolean),
        'color':((1.0,1.0,1.0),
                 ve_types.AnyOf(ve_types.Sequence3(ve_types.Real),
                                ve_types.Sequence4(ve_types.Real))),
        'color2':((0.7,0.2,0.7),
                 ve_types.AnyOf(ve_types.Sequence3(ve_types.Real),
                                ve_types.Sequence4(ve_types.Real))),
        'color2_act':((1.0,0.0,1.0),
                 ve_types.AnyOf(ve_types.Sequence3(ve_types.Real),
                                ve_types.Sequence4(ve_types.Real))),            
        'colorBG':((0.0,0.0,0.0),
                 ve_types.AnyOf(ve_types.Sequence3(ve_types.Real),
                                ve_types.Sequence4(ve_types.Real))),       

        'color_frame':((0.4,0.4,1.0),
                     ve_types.AnyOf(ve_types.Sequence3(ve_types.Real),
                                ve_types.Sequence4(ve_types.Real))),
        'screen_center' : ( ( 320.0, 240.0 ), # In eye coordinates
                       ve_types.AnyOf(ve_types.Sequence2(ve_types.Real),
                                      ve_types.Sequence3(ve_types.Real),
                                      ve_types.Sequence4(ve_types.Real)),
                       "units: eye coordinates"),
        'position' : ( ( 320.0, 240.0 ), # In eye coordinates
                       ve_types.AnyOf(ve_types.Sequence2(ve_types.Real),
                                      ve_types.Sequence3(ve_types.Real),
                                      ve_types.Sequence4(ve_types.Real)),
                       "units: eye coordinates"),
        'inisize':((64.0,16.0), # In eye coordinates
                ve_types.Sequence2(ve_types.Real)),
        'size':((64.0,16.0), # In eye coordinates
                ve_types.Sequence2(ve_types.Real)),
        'threshold1':(0.0, # 0.0 degrees = right, 90.0 degrees = up
                       ve_types.Real),
        'threshold2':(0.0, # 0.0 degrees = right, 90.0 degrees = up
                       ve_types.Real),
        'orientation' : (0.0, 
                         ve_types.Real),         
        }



    def __init__(self,**kw):
        VisionEgg.Core.Stimulus.__init__(self,**kw)
        self.value = self.parameters.inisize[1]/2
        self.t1 = self.parameters.inisize[1]*self.parameters.threshold1
        self.t2 = self.parameters.inisize[1]*self.parameters.threshold2
        self.active = False

    def resetValue(self):
        self.value = self.parameters.inisize[1]/2
        self.active = False
                 
    def setValue(self,v):
        #print("     self.value = %.2f, v = %2.f" % (self.value, v))

        if((v < self.parameters.inisize[1]-10) and (v > 10)):        
            self.value = v
            
            #print("value = %.2f - th1 = %.2f - th2 = %.2f" % (self.value, self.parameters.inisize[1]*self.parameters.threshold1 ,self.parameters.inisize[1]*self.parameters.threshold2))
          
        if(self.value > self.parameters.inisize[1]*self.parameters.threshold1):
            self.active = True
            print("1")
            return 1
        elif(self.value < self.parameters.inisize[1]*self.parameters.threshold2):
            self.active = True
            print("-1")
            return -1
        else:
            self.active = False
            return 0
            

                
    def draw(self):
        p = self.parameters # Shorthand
        
        if p.on:
         
            iw = p.inisize[0]
            ih = p.inisize[1]
                    
            w = p.size[0]
            h = p.size[1]
                                                                                    
            # Calculate center
      
            center = (p.position[0],p.position[1]-ih/2)
            
            gl.glMatrixMode(gl.GL_MODELVIEW)
            gl.glPushMatrix()
            gl.glBegin(gl.GL_LINES)
            gl.glColor3f(*p.color2)
            step = 2 * math.pi / 6
            for i in range(6):
                phi = 0 + i * step
                x = round (350 * math.cos(phi)) 
                y = round (350 * math.sin(phi))

                gl.glVertex3f(p.screen_center[0], p.screen_center[1], 0.0)
                gl.glVertex3f(p.screen_center[0]+x,p.screen_center[1]+y, 0.0)
                
            gl.glEnd()
            
            gl.glTranslate(center[0],center[1],0.0)
            
            gl.glRotate(p.orientation,0.0,0.0,1.0)
            
            gl.glColor3f(*p.color_frame)
                
            gl.glDisable(gl.GL_DEPTH_TEST)
            gl.glDisable(gl.GL_TEXTURE_2D)
            gl.glBlendFunc(gl.GL_SRC_ALPHA,gl.GL_ONE_MINUS_SRC_ALPHA)
            gl.glEnable(gl.GL_BLEND)
        
            gl.glColor3f(*p.color)
            gl.glBegin(gl.GL_QUADS) # Draw Container 
            gl.glVertex2f(  iw, ih)
            gl.glVertex2f( -iw, ih)
            gl.glVertex2f( -iw, 0)
            gl.glVertex2f(  iw, 0)                    
            gl.glEnd() # GL_QUADS
            
            gl.glColor3f(*p.colorBG)
            
            gl.glBegin(gl.GL_QUADS) # Draw background
            gl.glVertex2f(  iw-5, ih-5)
            gl.glVertex2f( -iw+5, ih-5)
            gl.glVertex2f( -iw+5, 5)
            gl.glVertex2f(  iw-5, 5)                    
            gl.glEnd() # GL_QUADS

            if(self.active == True):
                gl.glColor3f(*p.color2_act)
            else:
                gl.glColor3f(*p.color2)

                        
            gl.glBegin(gl.GL_QUADS) # Draw Rectangle
            gl.glVertex2f(  iw-10, ih/2)
            gl.glVertex2f( -iw+10, ih/2)
            gl.glVertex2f( -iw+10, self.value-10)
            gl.glVertex2f(  iw-10, self.value-10)                    
            gl.glEnd() # GL_QUADS                
  
           
            
            gl.glColor3f(*p.color)
            gl.glBegin(gl.GL_LINES)
            
            gl.glVertex3f(-iw, ih/2, 0.0)
            gl.glVertex3f(iw, ih/2, 0.0)
            
            gl.glVertex3f(-iw, self.t1, 0.0)
            gl.glVertex3f(iw, self.t1, 0.0)
            
            gl.glVertex3f(-iw, self.t2, 0.0)
            gl.glVertex3f(iw, self.t2, 0.0)

            gl.glEnd()
                      
            gl.glPopMatrix()
                    
class FilledHexagon(VisionEgg.Core.Stimulus):
    """Hexagonal stimulus. Adapted from http://www.visionegg.org
    
    Parameters 
    ========== 
    anchor        -- specifies how position parameter is interpreted (String) 
                     Default: center 
    anti_aliasing -- (Boolean) 
                     Default: True 
    center        -- DEPRECATED: don't use (Sequence2 of Real) 
                     Default: (determined at runtime) 
    color         -- (AnyOf(Sequence3 of Real or Sequence4 of Real)) 
                     Default: (1.0, 1.0, 1.0) 
    on            -- draw stimulus? (Boolean) (Boolean) 
                     Default: True 
    orientation   -- (Real) 
                     Default: 0.0 
    position      -- units: eye coordinates (AnyOf(Sequence2 of Real or Sequence3 of Real or Sequence4 of Real)) 
                     Default: (320.0, 240.0) 
    radius        -- the radius of the hexagon (dist. from center to a vertex). units: eye coordinates (Real) 
                     Default: 50.0 
    """ 
    
    parameters_and_defaults = { 
        'on' : (True, 
                ve_types.Boolean, 
                "draw stimulus? (Boolean)"), 
        'color' : ((1.0,1.0,1.0), 
                   ve_types.AnyOf(ve_types.Sequence3(ve_types.Real), 
                                  ve_types.Sequence4(ve_types.Real))), 
        'anti_aliasing' : (True, 
                          ve_types.Boolean), 
        'orientation' : (0.0, 
                         ve_types.Real), 
        'position' : ((320.0, 240.0), 
                       ve_types.AnyOf(ve_types.Sequence2(ve_types.Real), 
                                      ve_types.Sequence3(ve_types.Real), 
                                      ve_types.Sequence4(ve_types.Real)), 
                       "units: eye coordinates"), 
        'anchor' : ('center', 
                    ve_types.String, 
                    "specifies how position parameter is interpreted"), 
        'radius' : (30.0,
                   ve_types.Real, 
                   "units: eye coordinates"), 
        'center' : (None, 
                    ve_types.Sequence2(ve_types.Real), 
                    "DEPRECATED: don't use"), 
        } 
    
    __slots__ = ( 
        '_gave_alpha_warning', 
        ) 
   
    def __init__(self,**kw):
        VisionEgg.Core.Stimulus.__init__(self,**kw)
        self._gave_alpha_warning = 0
    
    def draw(self):
        p = self.parameters # shorthand 
        if p.center is not None: 
            if not hasattr(VisionEgg.config,"_GAVE_CENTER_DEPRECATION"): 
                logger = logging.getLogger('VisionEgg.MoreStimuli') 
                logger.warning("Specifying FilledHexagon by deprecated " 
                               "'center' parameter deprecated.  Use " 
                               "'position' parameter instead.  (Allows " 
                               "use of 'anchor' parameter to set to " 
                               "other values.)") 
                VisionEgg.config._GAVE_CENTER_DEPRECATION = 1 
            p.anchor = 'center' 
            p.position = p.center[0], p.center[1] # copy values (don't copy ref to tuple) 
        if p.on: 
            # calculate center 
            center = VisionEgg._get_center(p.position,p.anchor,(p.radius*2.0, p.radius*2.0)) 
            gl.glMatrixMode(gl.GL_MODELVIEW) 
            gl.glPushMatrix() 
            gl.glTranslate(center[0],center[1],0.0) 
            gl.glRotate(p.orientation,0.0,0.0,1.0) 
        
            if len(p.color)==3: 
                gl.glColor3f(*p.color) 
            elif len(p.color)==4: 
                gl.glColor4f(*p.color) 
            gl.glDisable(gl.GL_DEPTH_TEST) 
            gl.glDisable(gl.GL_TEXTURE_2D) 
            gl.glBlendFunc(gl.GL_SRC_ALPHA,gl.GL_ONE_MINUS_SRC_ALPHA) 
            gl.glEnable(gl.GL_BLEND) 
        
            a = p.radius/2.0
            r2 = p.radius**2
            b = sqrt(r2 - r2/4.0)
        
            gl.glBegin(gl.GL_TRIANGLE_FAN) 
            gl.glVertex3f(0.0, 0.0, 0.0)
            gl.glVertex3f( a, b, 0.0)
            gl.glVertex3f( p.radius, 0.0, 0.0)
            gl.glVertex3f( a,-b, 0.0)
            gl.glVertex3f(-a,-b, 0.0)
            gl.glVertex3f(-p.radius, 0.0, 0.0)
            gl.glVertex3f(-a, b, 0.0)
            gl.glVertex3f( a, b, 0.0)
            gl.glEnd() # GL_TRIANGLE_FAN 
        
            if p.anti_aliasing: 
                if not self._gave_alpha_warning: 
                    if len(p.color) > 3 and p.color[3] != 1.0: 
                        logger = logging.getLogger('VisionEgg.MoreStimuli') 
                        logger.warning("The parameter anti_aliasing is " 
                                       "set to true in the Target2D " 
                                       "stimulus class, but the color " 
                                       "parameter specifies an alpha " 
                                       "value other than 1.0.  To " 
                                       "acheive anti-aliasing, ensure " 
                                       "that the alpha value for the " 
                                       "color parameter is 1.0.") 
                        self._gave_alpha_warning = 1 
        
                # We've already drawn a filled polygon (aliased), now 
                # redraw the outline of the polygon (with 
                # anti-aliasing).  (Using GL_POLYGON_SMOOTH results in 
                # artifactual lines where triangles were joined to 
                # create quad, at least on some OpenGL 
                # implementations.) 
        
                # Calculate coverage value for each pixel of outline 
                # and store as alpha 
                gl.glEnable(gl.GL_LINE_SMOOTH) 
                # Now specify how to use the alpha value 
                gl.glBlendFunc(gl.GL_SRC_ALPHA,gl.GL_ONE_MINUS_SRC_ALPHA) 
                gl.glEnable(gl.GL_BLEND) 
        
                # Draw a second polygon in line mode, so the edges are anti-aliased 
                gl.glPolygonMode(gl.GL_FRONT_AND_BACK,gl.GL_LINE) 
                gl.glBegin(gl.GL_TRIANGLE_FAN) 
                gl.glVertex3f(0.0, 0.0, 0.0)
                gl.glVertex3f( a, b, 0.0)
                gl.glVertex3f( p.radius, 0.0, 0.0)
                gl.glVertex3f( a,-b, 0.0)
                gl.glVertex3f(-a,-b, 0.0)
                gl.glVertex3f(-p.radius, 0.0, 0.0)
                gl.glVertex3f(-a, b, 0.0)
                gl.glVertex3f( a, b, 0.0)
                gl.glEnd() # GL_TRIANGLE_FAN
        
                # Set the polygon mode back to fill mode 
                gl.glPolygonMode(gl.GL_FRONT_AND_BACK,gl.GL_FILL) 
                gl.glDisable(gl.GL_LINE_SMOOTH) 
            gl.glPopMatrix()

class HexagonOpening(VisionEgg.Core.Stimulus):
    """Hexagonal stimulus with opening. Adapted from http://www.visionegg.org
    
    Parameters 
    ========== 
    anchor        -- specifies how position parameter is interpreted (String) 
                     Default: center 
    center        -- DEPRECATED: don't use (Sequence2 of Real) 
                     Default: (determined at runtime) 
    color         -- (AnyOf(Sequence3 of Real or Sequence4 of Real)) 
                     Default: (1.0, 1.0, 1.0) 
    egde_color    -- (AnyOf(Sequence3 of Real or Sequence4 of Real)) 
                     Default: (1.0, 1.0, 1.0) 
    on            -- draw stimulus? (Boolean) (Boolean) 
                     Default: True 
    orientation   -- (Real) 
                     Default: 0.0 
    position      -- units: eye coordinates (AnyOf(Sequence2 of Real or Sequence3 of Real or Sequence4 of Real)) 
                     Default: (320.0, 240.0) 
    radius        -- the radius of the hexagon (dist. from center to a vertex). units: eye coordinates (Real) 
                     Default: 50.0 
    opening_radius -- the radius of the hexagon (dist. from center to a vertex). units: eye coordinates (Real) 
                     Default: 10.0 
    """ 
    
    parameters_and_defaults = { 
        'on' : (True, 
                ve_types.Boolean, 
                "draw stimulus? (Boolean)"), 
        'color' : ((1.0,1.0,1.0), 
                   ve_types.AnyOf(ve_types.Sequence3(ve_types.Real), 
                                  ve_types.Sequence4(ve_types.Real))), 
        'edge_color' : ((0.0,0.0,0.0), 
                   ve_types.AnyOf(ve_types.Sequence3(ve_types.Real), 
                                  ve_types.Sequence4(ve_types.Real))), 
        'orientation' : (0.0, 
                         ve_types.Real), 
        'position' : ((320.0, 240.0), 
                       ve_types.AnyOf(ve_types.Sequence2(ve_types.Real), 
                                      ve_types.Sequence3(ve_types.Real), 
                                      ve_types.Sequence4(ve_types.Real)), 
                       "units: eye coordinates"), 
        'anchor' : ('center', 
                    ve_types.String, 
                    "specifies how position parameter is interpreted"), 
        'radius' : (30.0,
                   ve_types.Real, 
                   "units: eye coordinates"), 
        'opening_radius' : (10.0,
                   ve_types.Real, 
                   "units: eye coordinates"), 
        'center' : (None, 
                    ve_types.Sequence2(ve_types.Real), 
                    "DEPRECATED: don't use"), 
        } 
    
    __slots__ = ( 
        '_gave_alpha_warning', 
        ) 
   
    def __init__(self,**kw):
        VisionEgg.Core.Stimulus.__init__(self,**kw)
        self._gave_alpha_warning = 0
    
    def draw(self):
        p = self.parameters # shorthand 
        if p.center is not None: 
            if not hasattr(VisionEgg.config,"_GAVE_CENTER_DEPRECATION"): 
                logger = logging.getLogger('VisionEgg.MoreStimuli') 
                logger.warning("Specifying FilledHexagon by deprecated " 
                               "'center' parameter deprecated.  Use " 
                               "'position' parameter instead.  (Allows " 
                               "use of 'anchor' parameter to set to " 
                               "other values.)") 
                VisionEgg.config._GAVE_CENTER_DEPRECATION = 1 
            p.anchor = 'center' 
            p.position = p.center[0], p.center[1] # copy values (don't copy ref to tuple) 
        if p.on: 
            # calculate center 
            center = VisionEgg._get_center(p.position,p.anchor,(p.radius*2.0, p.radius*2.0)) 
            gl.glMatrixMode(gl.GL_MODELVIEW) 
            gl.glPushMatrix() 
            gl.glTranslate(center[0],center[1],0.0) 
            gl.glRotate(p.orientation,0.0,0.0,1.0) 
        
            if len(p.color)==3: 
                gl.glColor3f(*p.color) 
            elif len(p.color)==4: 
                gl.glColor4f(*p.color) 
            gl.glDisable(gl.GL_DEPTH_TEST) 
            gl.glDisable(gl.GL_TEXTURE_2D) 
            gl.glBlendFunc(gl.GL_SRC_ALPHA,gl.GL_ONE_MINUS_SRC_ALPHA) 
            gl.glEnable(gl.GL_BLEND) 
            gl.glEnable(gl.GL_LINE_SMOOTH) 
        
            a = p.radius/2.0
            r2 = p.radius**2
            b = sqrt(r2 - r2/4.0)
            out = (
                (a, b),
                (p.radius, 0.),
                (a, -b),
                (-a, -b),
                (-p.radius, 0.),
                (-a, b),
                (a, b)
                )
            a = p.opening_radius/2.0
            r2 = p.opening_radius**2
            b = sqrt(r2 - r2/4.0)
            inn = (
                (a, b),
                (p.opening_radius, 0.),
                (a, -b),
                (-a, -b),
                (-p.opening_radius, 0.),
                (-a, b),
                (a, b)
                )
        
            gl.glBegin(gl.GL_TRIANGLE_STRIP) 
            for i in xrange(7):
                j = i % 6
                gl.glVertex3f(inn[j][0], inn[j][1], 0.0)
                gl.glVertex3f(out[j][0], out[j][1], 0.0)
            gl.glEnd()

            if len(p.edge_color)==3: 
                gl.glColor3f(*p.edge_color) 
            elif len(p.edge_color)==4: 
                gl.glColor4f(*p.edge_color) 
 
            gl.glBegin(gl.GL_LINES)
            for i in xrange(6):
                gl.glVertex3f(inn[i][0], inn[i][1], 0.0)
                gl.glVertex3f(out[i][0], out[i][1], 0.0)
            gl.glEnd()
            gl.glDisable(gl.GL_LINE_SMOOTH)
##            for i in (0, 2, 3, 5):
##                gl.glBegin(gl.GL_LINE)
##                gl.glVertex3f(inn[i][0], inn[i][1], 0.0)
##                gl.glVertex3f(out[i][0], out[i][1], 0.0)
##                gl.glEnd()
##            for i in (1, 4):
##                gl.glBegin(gl.GL_LINE)
##                gl.glVertex3f(inn[i][0], inn[i][1], 0.0)
##                gl.glVertex3f(out[i][0], out[i][1], 0.0)
##                gl.glEnd()

            gl.glPopMatrix()

class StripeField(VisionEgg.Core.Stimulus):
    """Field of arrow-shaped stripes. Adapted from http://www.visionegg.org
    
    Parameters 
    ========== 
    anti_aliasing -- (Boolean) 
                     Default: True 
    anchor        -- specifies how position parameter is interpreted (String) 
                     Default: center 
    center        -- DEPRECATED: don't use (Sequence2 of Real) 
                     Default: (determined at runtime) 
    color         -- (AnyOf(Sequence3 of Real or Sequence4 of Real)) 
                     Default: (1.0, 1.0, 1.0) 
    egde_color    -- (AnyOf(Sequence3 of Real or Sequence4 of Real)) 
                     Default: (1.0, 1.0, 1.0) 
    on            -- draw stimulus? (Boolean) (Boolean) 
                     Default: True 
    orientation   -- (Real) 
                     Default: 0.0 
    width        -- width of the stripe field
                     Default: 50.0 
    angle        -- width of the stripe field
                     Default: 50.0
    dist_stripes -- distance between stripes
                     Default: 20.0
    num_stripes  -- stripe number
                     Default: 12
    line_width   -- line width
                     Default: 1.0
    """ 
    
    parameters_and_defaults = { 
        'on' : (True, 
                ve_types.Boolean, 
                "draw stimulus? (Boolean)"), 
        'anti_aliasing': (True,
                ve_types.Boolean,
                "antialiasing (Boolean)"),
        'color' : ((1.0,1.0,1.0), 
                   ve_types.AnyOf(ve_types.Sequence3(ve_types.Real), 
                                  ve_types.Sequence4(ve_types.Real))), 
        'edge_color' : ((0.0,0.0,0.0), 
                   ve_types.AnyOf(ve_types.Sequence3(ve_types.Real), 
                                  ve_types.Sequence4(ve_types.Real))), 
        'orientation' : (0.0, 
                         ve_types.Real), 
        'anchor' : ('center', 
                    ve_types.String, 
                    "specifies how position parameter is interpreted"), 
        'width' : (30.0,
                   ve_types.Real, 
                   "units: eye coordinates"), 
        'center' : (None, 
                    ve_types.Sequence2(ve_types.Real), 
                    "DEPRECATED: don't use"), 
        'angle' : (.3,
                    ve_types.Real,
                    "Arrow angle (0.5 -> 90 degree)"),
        'dist_stripes' : (20.,
                    ve_types.Real,
                    "Stripe distance"),
        'num_stripes' : (12,
                    ve_types.Integer,
                    "Number of stripes"),
        'line_width'  : (1.0,
                    ve_types.Real,
                    "Line width"),
        } 
    
    __slots__ = ( 
        '_gave_alpha_warning', 
        ) 
   
    def __init__(self,**kw):
        VisionEgg.Core.Stimulus.__init__(self,**kw)
        self._gave_alpha_warning = 0
    
    def draw(self):
        p = self.parameters # shorthand 
        if p.on: 
            gl.glMatrixMode(gl.GL_MODELVIEW) 
            gl.glPushMatrix() 
            gl.glTranslate(p.center[0],p.center[1],0.0) 
            gl.glRotate(p.orientation,0.0,0.0,1.0) 
        
            if len(p.color)==3: 
                gl.glColor3f(*p.color) 
            elif len(p.color)==4: 
                gl.glColor4f(*p.color) 
            gl.glDisable(gl.GL_DEPTH_TEST)
            gl.glDisable(gl.GL_TEXTURE_2D)
            gl.glBlendFunc(gl.GL_SRC_ALPHA,gl.GL_ONE_MINUS_SRC_ALPHA)
            gl.glEnable(gl.GL_BLEND)
            if p.anti_aliasing:
                gl.glEnable(gl.GL_LINE_SMOOTH)
            else:
                gl.glDisable(gl.GL_LINE_SMOOTH)
            old_width = gl.glGetFloatv(gl.GL_LINE_WIDTH)
            gl.glLineWidth(p.line_width)

            arrow_offset = p.width * p.angle
            for i in xrange(- p.num_stripes / 2 - 1, p.num_stripes / 2 - 2):
                offset = i * p.dist_stripes
                gl.glBegin(gl.GL_LINE_STRIP)
                gl.glVertex3f(-p.width / 2., offset, 0.)
                gl.glVertex3f(0., offset + arrow_offset, 0.)
                gl.glVertex3f(p.width / 2., offset, 0.)
                gl.glEnd()

            gl.glLineWidth(old_width)

            gl.glPopMatrix()


class DotField(VisionEgg.Core.Stimulus):
    """Field of dots. Adapted from http://www.visionegg.org

    Parameters
    ==========
    anti_aliasing -- (Boolean)
                     Default: True
    anchor        -- specifies how position parameter is interpreted (String)
                     Default: center
    center        -- DEPRECATED: don't use (Sequence2 of Real)
                     Default: (determined at runtime)
    color         -- (AnyOf(Sequence3 of Real or Sequence4 of Real))
                     Default: (1.0, 1.0, 1.0)
    egde_color    -- (AnyOf(Sequence3 of Real or Sequence4 of Real))
                     Default: (1.0, 1.0, 1.0)
    on            -- draw stimulus? (Boolean) (Boolean)
                     Default: True
    orientation   -- (Real)
                     Default: 0.0
    width         -- width of the dot field
                     Default: 50.0
    angle         -- width of the dot field
                     Default: 50.0
    dot_size      -- dot size
                     Default: 20.0
    num_dots      -- dot number
                     Default: 12
    line_width    -- line width
                     Default: 1.0
    circle_steps  -- circle approximation precision (8=octagon)
                     Default: 8.0
    """

    parameters_and_defaults = {
        'on' : (True,
                ve_types.Boolean,
                "draw stimulus? (Boolean)"),
        'anti_aliasing': (True,
                ve_types.Boolean,
                "antialiasing (Boolean)"),
        'color' : ((1.0,1.0,1.0),
                   ve_types.AnyOf(ve_types.Sequence3(ve_types.Real),
                                  ve_types.Sequence4(ve_types.Real))),
        'edge_color' : ((0.0,0.0,0.0),
                   ve_types.AnyOf(ve_types.Sequence3(ve_types.Real),
                                  ve_types.Sequence4(ve_types.Real))),
        'orientation' : (0.0,
                         ve_types.Real),
        'anchor' : ('center',
                    ve_types.String,
                    "specifies how position parameter is interpreted"),
        'width' : (30.0,
                   ve_types.Real,
                   "units: eye coordinates"),
        'height' : (30.0,
                   ve_types.Real,
                   "units: eye coordinates"),
        'center' : (None,
                    ve_types.Sequence2(ve_types.Real),
                    "DEPRECATED: don't use"),
        'dot_size' : (2.,
                    ve_types.Real,
                    "Dot size"),
        'dot_distance' : (10.,
                    ve_types.Real,
                    "Distance between dots"),
        'line_width'  : (1.0,
                    ve_types.Real,
                    "Line width"),
        'circle_steps'  : (8,
                    ve_types.Integer,
                    "Circle approximation precision"),
        }

    __slots__ = (
        '_gave_alpha_warning',
        )

    def __init__(self,**kw):
        VisionEgg.Core.Stimulus.__init__(self,**kw)
        self._gave_alpha_warning = 0

    def draw(self):
        p = self.parameters # shorthand
        if p.on:
            gl.glMatrixMode(gl.GL_MODELVIEW)
            gl.glPushMatrix()
            gl.glTranslate(p.center[0],p.center[1],0.0)
            gl.glRotate(p.orientation,0.0,0.0,1.0)

            if len(p.color)==3:
                gl.glColor3f(*p.color)
            elif len(p.color)==4:
                gl.glColor4f(*p.color)
            gl.glDisable(gl.GL_DEPTH_TEST)
            gl.glDisable(gl.GL_TEXTURE_2D)
            gl.glBlendFunc(gl.GL_SRC_ALPHA,gl.GL_ONE_MINUS_SRC_ALPHA)
            gl.glEnable(gl.GL_BLEND)
            if p.anti_aliasing:
                gl.glEnable(gl.GL_LINE_SMOOTH)
            else:
                gl.glDisable(gl.GL_LINE_SMOOTH)
            old_width = gl.glGetFloatv(gl.GL_LINE_WIDTH)
            gl.glLineWidth(p.line_width)

            def circle(x, y, coords):
                gl.glBegin(gl.GL_TRIANGLE_FAN)
                gl.glVertex3f(x, y, 0.)
                for p in coords:
                    gl.glVertex3f(x + p[0], y + p[1], 0.)
                gl.glVertex3f(x + coords[0][0], y + coords[0][1], 0.)
                gl.glEnd()

            # calculate dot dimensions
            r = p.dot_size / 2.
            circle_coords = []
            for i in xrange(p.circle_steps):
                alpha = (float(i) / float(p.circle_steps)) * 2 * math.pi
                circle_coords.append((r * math.cos(alpha), r * math.sin(alpha)))

            # draw dot field
            y = - p.height / 2
            while y < p.height / 2:
                x = - p.width / 2
                while x < p.width / 2:
                    circle(x, y, circle_coords)
                    x += p.dot_distance
                y += p.dot_distance

            gl.glLineWidth(old_width)

            gl.glPopMatrix()


class FilledTriangle(VisionEgg.Core.Stimulus):
    """Triangle stimulus. Adapted from http://www.visionegg.org
    
    Parameters 
    ========== 
    anchor        -- specifies how position parameter is interpreted (String) 
                     Default: center 
    anti_aliasing -- (Boolean) 
                     Default: True 
    center        -- DEPRECATED: don't use (Sequence2 of Real) 
                     Default: (determined at runtime) 
    color         -- (AnyOf(Sequence3 of Real or Sequence4 of Real)) 
                     Default: (1.0, 1.0, 1.0) 
    on            -- draw stimulus? (Boolean) (Boolean) 
                     Default: True 
    orientation   -- (Real) 
                     Default: 0.0 
    position      -- units: eye coordinates (AnyOf(Sequence2 of Real or Sequence3 of Real or Sequence4 of Real)) 
                     Default: (320.0, 240.0) 
    size          -- size of triangle as edge length. units: eye coordinates (Real) 
                     Default: 50.0 
    innerSize     -- if inner part of triangle is to be cut out, specify innerSize < size here
    innerColor    -- if inner part of triangle is to be cut out, specify according color
    """ 
    
    parameters_and_defaults = { 
        'on' : (True, 
                ve_types.Boolean, 
                "draw stimulus? (Boolean)"), 
        'color' : ((1.0,1.0,1.0), 
                   ve_types.AnyOf(ve_types.Sequence3(ve_types.Real), 
                                  ve_types.Sequence4(ve_types.Real))), 
        'anti_aliasing' : (True, 
                          ve_types.Boolean), 
        'orientation' : (0.0, 
                         ve_types.Real), 
        'position' : ((320.0, 240.0), 
                       ve_types.AnyOf(ve_types.Sequence2(ve_types.Real), 
                                      ve_types.Sequence3(ve_types.Real), 
                                      ve_types.Sequence4(ve_types.Real)), 
                       "units: eye coordinates"), 
        'anchor' : ('center', 
                    ve_types.String, 
                    "specifies how position parameter is interpreted"), 
        'size' : (30.0,
                 ve_types.Real, 
                 "units: eye coordinates"), 
        'innerSize' : (0.0,
                 ve_types.Real, 
                 "units: eye coordinates"), 
        'innerColor' : ((0.0,0.0,0.0), 
                   ve_types.AnyOf(ve_types.Sequence3(ve_types.Real), 
                                  ve_types.Sequence4(ve_types.Real))), 
        'watermark_size' : (0.0,
                            ve_types.Real,
                            "units: eye coordinates"),
        'watermark_color' : ((0.0,0.0,0.0,0.0),
                             ve_types.Sequence4(ve_types.Real)),
        'center' : (None, 
                    ve_types.Sequence2(ve_types.Real), 
                    "DEPRECATED: don't use")
        } 
    
    __slots__ = ( 
        '_gave_alpha_warning', 
        ) 
   
    def __init__(self,**kw):
        VisionEgg.Core.Stimulus.__init__(self,**kw)
        self._gave_alpha_warning = 0
    
    def draw(self):
        p = self.parameters # shorthand 
        if p.center is not None: 
            if not hasattr(VisionEgg.config,"_GAVE_CENTER_DEPRECATION"): 
                logger = logging.getLogger('VisionEgg.MoreStimuli') 
                logger.warning("Specifying FilledTriangle by deprecated " 
                               "'center' parameter deprecated.  Use " 
                               "'position' parameter instead.  (Allows " 
                               "use of 'anchor' parameter to set to " 
                               "other values.)") 
                VisionEgg.config._GAVE_CENTER_DEPRECATION = 1 
            p.anchor = 'center' 
            p.position = p.center[0], p.center[1] # copy values (don't copy ref to tuple) 
        if p.on: 
            # calculate center 
            center = self._my_get_center(p.position,p.anchor,p.size) 
            gl.glMatrixMode(gl.GL_MODELVIEW) 
            gl.glPushMatrix() 
            gl.glTranslate(center[0],center[1],0.0) 
            gl.glRotate(p.orientation,0.0,0.0,1.0) 
        
            if len(p.color)==3: 
                gl.glColor3f(*p.color) 
            elif len(p.color)==4: 
                gl.glColor4f(*p.color) 
            gl.glDisable(gl.GL_DEPTH_TEST) 
            gl.glDisable(gl.GL_TEXTURE_2D) 
            gl.glBlendFunc(gl.GL_SRC_ALPHA,gl.GL_ONE_MINUS_SRC_ALPHA) 
            gl.glEnable(gl.GL_BLEND) 
            
            a = p.size/2.0
            b = a/3.0 * sqrt(3.0)
            r = b*2.0
                    
            gl.glBegin(gl.GL_TRIANGLES)
            gl.glVertex3f( a, -b, 0.0)
            gl.glVertex3f(-a, -b, 0.0)
            gl.glVertex3f(0.0, r, 0.0)
            gl.glEnd() # GL_TRIANGLE_FAN     

            if not (p.innerSize == 0.0):
                # Cut out inner part
                if len(p.innerColor)==3: 
                    gl.glColor3f(*p.innerColor) 
                elif len(p.innerColor)==4: 
                    gl.glColor4f(*p.innerColor) 
                gl.glDisable(gl.GL_DEPTH_TEST) 
                gl.glDisable(gl.GL_TEXTURE_2D) 
                gl.glBlendFunc(gl.GL_SRC_ALPHA,gl.GL_ONE_MINUS_SRC_ALPHA) 
                gl.glEnable(gl.GL_BLEND) 
                
                a2 = p.innerSize/2.0
                b2 = a2/3.0 * sqrt(3.0)
                r2 = b2*2.0
                        
                gl.glBegin(gl.GL_TRIANGLES)
                gl.glVertex3f( a2, -b2, 0.0)
                gl.glVertex3f(-a2, -b2, 0.0)
                gl.glVertex3f(0.0, r2, 0.0)
                gl.glEnd() # GL_TRIANGLE_FAN 
                
        	if not (p.watermark_size == 0.0):
        	    # place and paint watermark
        	    gl.glColor4f(*p.watermark_color) 
                
                gl.glDisable(gl.GL_DEPTH_TEST) 
                gl.glDisable(gl.GL_TEXTURE_2D) 
                gl.glBlendFunc(gl.GL_SRC_ALPHA,gl.GL_ONE_MINUS_SRC_ALPHA)
                gl.glEnable(gl.GL_BLEND)
                
                a3 = p.watermark_size/2.0
                b3 = a3/3.0 * sqrt(3.0)
                r3 = b3*2.0
                        
                gl.glBegin(gl.GL_TRIANGLES)
                gl.glVertex3f( a3, -b3, 0.0)
                gl.glVertex3f(-a3, -b3, 0.0)
                gl.glVertex3f(0.0, r3, 0.0)
                gl.glEnd() # GL_TRIANGLE_FAN
        
            if p.anti_aliasing: 
                if not self._gave_alpha_warning: 
                    if len(p.color) > 3 and p.color[3] != 1.0: 
                        logger = logging.getLogger('VisionEgg.MoreStimuli') 
                        logger.warning("The parameter anti_aliasing is " 
                                       "set to true in the Target2D " 
                                       "stimulus class, but the color " 
                                       "parameter specifies an alpha " 
                                       "value other than 1.0.  To " 
                                       "acheive anti-aliasing, ensure " 
                                       "that the alpha value for the " 
                                       "color parameter is 1.0.") 
                        self._gave_alpha_warning = 1 
        
                # We've already drawn a filled polygon (aliased), now 
                # redraw the outline of the polygon (with 
                # anti-aliasing).  (Using GL_POLYGON_SMOOTH results in 
                # artifactual lines where triangles were joined to 
                # create quad, at least on some OpenGL 
                # implementations.) 
        
                # Calculate coverage value for each pixel of outline 
                # and store as alpha 
                gl.glEnable(gl.GL_LINE_SMOOTH) 
                # Now specify how to use the alpha value 
                gl.glBlendFunc(gl.GL_SRC_ALPHA,gl.GL_ONE_MINUS_SRC_ALPHA) 
                gl.glEnable(gl.GL_BLEND) 
        
                # Draw a second polygon in line mode, so the edges are anti-aliased 
                gl.glPolygonMode(gl.GL_FRONT_AND_BACK,gl.GL_LINE)
                gl.glBegin(gl.GL_TRIANGLES)
                gl.glVertex3f( a, -b, 0.0)
                gl.glVertex3f(-a, -b, 0.0)
                gl.glVertex3f(0.0, r, 0.0)
                gl.glEnd() # GL_TRIANGLE_FAN
        
                # Set the polygon mode back to fill mode 
                gl.glPolygonMode(gl.GL_FRONT_AND_BACK,gl.GL_FILL) 
                gl.glDisable(gl.GL_LINE_SMOOTH) 
            gl.glPopMatrix()


    def _my_get_center(self, position, anchor, size):
        """Private helper function"""
        if anchor == 'center':
            center = position
        else:
            a = size/2.0
            b = a/3.0 * sqrt(3.0)
            r = b*2.0
            if anchor == 'lowerleft': # lower left point of the triangle
                center = (position[0] + a, position[1] + b)
            elif anchor == 'lowerright': # lower reight point of the triangle
                center = (position[0] - a, position[1] + b)
            elif anchor == 'upperright':
                center = (position[0] - a, position[1] - r)
            elif anchor == 'upperleft':
                center = (position[0] + a, position[1] - r)
            elif anchor == 'left':
                center = (position[0] + a, position[1])
            elif anchor == 'right':
                center = (position[0] - a, position[1])
            elif anchor == 'bottom':
                center = (position[0],position[1] + b)
            elif anchor == 'top':
                center = (position[0],position[1] - r)
            else:
                raise ValueError("No anchor position %s"%anchor)
        return center


class FilledHourglass(VisionEgg.Core.Stimulus):
    """Hourglass stimulus, consisting of two triangles pointed to each other.
       Adapted from http://www.visionegg.org
       
    
    Parameters 
    ========== 
    anchor        -- specifies how position parameter is interpreted (String) 
                     Default: center 
    anti_aliasing -- (Boolean) 
                     Default: True 
    center        -- DEPRECATED: don't use (Sequence2 of Real) 
                     Default: (determined at runtime) 
    color         -- (AnyOf(Sequence3 of Real or Sequence4 of Real)) 
                     Default: (1.0, 1.0, 1.0) 
    on            -- draw stimulus? (Boolean) (Boolean) 
                     Default: True 
    orientation   -- (Real) 
                     Default: 0.0 
    position      -- units: eye coordinates (AnyOf(Sequence2 of Real or Sequence3 of Real or Sequence4 of Real)) 
                     Default: (320.0, 240.0) 
    size          -- size of one triangle part of the hourglass as edge length. units: eye coordinates (Real) 
                     Default: 50.0 
    """ 
    
    parameters_and_defaults = { 
        'on' : (True, 
                ve_types.Boolean, 
                "draw stimulus? (Boolean)"), 
        'color' : ((1.0,1.0,1.0), 
                   ve_types.AnyOf(ve_types.Sequence3(ve_types.Real), 
                                  ve_types.Sequence4(ve_types.Real))), 
        'anti_aliasing' : (False, 
                          ve_types.Boolean), 
        'orientation' : (0.0, 
                         ve_types.Real), 
        'position' : ((320.0, 240.0), 
                       ve_types.AnyOf(ve_types.Sequence2(ve_types.Real), 
                                      ve_types.Sequence3(ve_types.Real), 
                                      ve_types.Sequence4(ve_types.Real)), 
                       "units: eye coordinates"), 
        'anchor' : ('center', 
                    ve_types.String, 
                    "specifies how position parameter is interpreted"), 
        'size' : (30.0,
                 ve_types.Real, 
                 "units: eye coordinates"), 
        'watermark_size' : (0.0,
                            ve_types.Real,
                            "units: eye coordinates"),
        'watermark_color' : ((0.0,0.0,0.0,0.0),
                             ve_types.Sequence4(ve_types.Real)),
        'center' : (None, 
                    ve_types.Sequence2(ve_types.Real), 
                    "DEPRECATED: don't use"), 
        } 
    
    __slots__ = ( 
        '_gave_alpha_warning', 
        ) 
   
    def __init__(self,**kw):
        VisionEgg.Core.Stimulus.__init__(self,**kw)
        self._gave_alpha_warning = 0
    
    def draw(self):
        p = self.parameters # shorthand 
        if p.center is not None: 
            if not hasattr(VisionEgg.config,"_GAVE_CENTER_DEPRECATION"): 
                logger = logging.getLogger('VisionEgg.MoreStimuli') 
                logger.warning("Specifying FilledTriangle by deprecated " 
                               "'center' parameter deprecated.  Use " 
                               "'position' parameter instead.  (Allows " 
                               "use of 'anchor' parameter to set to " 
                               "other values.)") 
                VisionEgg.config._GAVE_CENTER_DEPRECATION = 1 
            p.anchor = 'center' 
            p.position = p.center[0], p.center[1] # copy values (don't copy ref to tuple) 
        if p.on: 
            # calculate center 
            center = self._my_get_center(p.position,p.anchor,p.size) 
            gl.glMatrixMode(gl.GL_MODELVIEW) 
            gl.glPushMatrix() 
            gl.glTranslate(center[0],center[1],0.0) 
            gl.glRotate(p.orientation,0.0,0.0,1.0) 

            # Upper triangle
            if len(p.color)==3: 
                gl.glColor3f(*p.color) 
            elif len(p.color)==4: 
                gl.glColor4f(*p.color) 
            gl.glDisable(gl.GL_DEPTH_TEST) 
            gl.glDisable(gl.GL_TEXTURE_2D) 
            gl.glBlendFunc(gl.GL_SRC_ALPHA,gl.GL_ONE_MINUS_SRC_ALPHA) 
            gl.glEnable(gl.GL_BLEND) 
            
            a = p.size/2.0
            b = a/3.0 * sqrt(3.0)
            r = b*2.0
                    
            gl.glBegin(gl.GL_TRIANGLES)
            #gl.glVertex3f( a, -b, 0.0)
            #gl.glVertex3f(-a, -b, 0.0)
            #gl.glVertex3f(0.0, r, 0.0)
            gl.glVertex3f( a, -(r+b), 0.0)
            gl.glVertex3f(-a, -(r+b), 0.0)
            gl.glVertex3f(0.0, 0.0, 0.0)
            
            gl.glEnd() # GL_TRIANGLE_FAN   
            gl.glBegin(gl.GL_TRIANGLES)
            gl.glVertex3f( a, r+b, 0.0)
            gl.glVertex3f(-a, r+b, 0.0)
            gl.glVertex3f(0.0, 0.0, 0.0)
            gl.glEnd() # GL_TRIANGLE_FAN                

            # Lower triangle
            #gl.glMatrixMode(gl.GL_MODELVIEW) 
            #gl.glPushMatrix() 
            #gl.glTranslate(center[0],center[1],0.0) 
        
            #if len(p.color)==3: 
            #    gl.glColor3f(*p.color) 
            #elif len(p.color)==4: 
            #    gl.glColor4f(*p.color) 
            #gl.glDisable(gl.GL_DEPTH_TEST) 
            #gl.glDisable(gl.GL_TEXTURE_2D) 
            #gl.glBlendFunc(gl.GL_SRC_ALPHA,gl.GL_ONE_MINUS_SRC_ALPHA) 
            #gl.glEnable(gl.GL_BLEND) 
            #        

            if not (p.watermark_size == 0.0):
                # paint and place watermark
                gl.glColor4f(*p.watermark_color)

                gl.glDisable(gl.GL_DEPTH_TEST) 
                gl.glDisable(gl.GL_TEXTURE_2D) 
                gl.glBlendFunc(gl.GL_SRC_ALPHA,gl.GL_ONE_MINUS_SRC_ALPHA) 
                gl.glEnable(gl.GL_BLEND)

                w = p.watermark_size/2
        
                gl.glBegin(gl.GL_QUADS)
                gl.glVertex3f(-w,-w, 0.0)
                gl.glVertex3f( w,-w, 0.0)
                gl.glVertex3f( w, w, 0.0)
                gl.glVertex3f(-w, w, 0.0)
                gl.glEnd() # GL_QUADS            
        
            if p.anti_aliasing: 
                if not self._gave_alpha_warning: 
                    if len(p.color) > 3 and p.color[3] != 1.0: 
                        logger = logging.getLogger('VisionEgg.MoreStimuli') 
                        logger.warning("The parameter anti_aliasing is " 
                                       "set to true in the Target2D " 
                                       "stimulus class, but the color " 
                                       "parameter specifies an alpha " 
                                       "value other than 1.0.  To " 
                                       "acheive anti-aliasing, ensure " 
                                       "that the alpha value for the " 
                                       "color parameter is 1.0.") 
                        self._gave_alpha_warning = 1 
        
                # We've already drawn a filled polygon (aliased), now 
                # redraw the outline of the polygon (with 
                # anti-aliasing).  (Using GL_POLYGON_SMOOTH results in 
                # artifactual lines where triangles were joined to 
                # create quad, at least on some OpenGL 
                # implementations.) 
        
                # Calculate coverage value for each pixel of outline 
                # and store as alpha 
                gl.glEnable(gl.GL_LINE_SMOOTH) 
                # Now specify how to use the alpha value 
                gl.glBlendFunc(gl.GL_SRC_ALPHA,gl.GL_ONE_MINUS_SRC_ALPHA) 
                gl.glEnable(gl.GL_BLEND) 
        
                # Draw a second polygon in line mode, so the edges are anti-aliased 
                gl.glPolygonMode(gl.GL_FRONT_AND_BACK,gl.GL_LINE)
                gl.glBegin(gl.GL_TRIANGLES)
                gl.glVertex3f( a, -b, 0.0)
                gl.glVertex3f(-a, -b, 0.0)
                gl.glVertex3f(0.0, r, 0.0)
                gl.glEnd() # GL_TRIANGLE_FAN
      
                # Set the polygon mode back to fill mode 
                gl.glPolygonMode(gl.GL_FRONT_AND_BACK,gl.GL_FILL) 
                gl.glDisable(gl.GL_LINE_SMOOTH) 
            gl.glPopMatrix()


    def _my_get_center(self, position, anchor, size):
        """Private helper function"""
        if anchor == 'center':
            center = position
        else:
            a = size/2.0
            b = a/3.0 * sqrt(3.0)
            r = b*2.0
            if anchor == 'lowerleft': # lower left point of the triangle
                center = (position[0] + a, position[1] + b)
            elif anchor == 'lowerright': # lower reight point of the triangle
                center = (position[0] - a, position[1] + b)
            elif anchor == 'upperright':
                center = (position[0] - a, position[1] - r)
            elif anchor == 'upperleft':
                center = (position[0] + a, position[1] - r)
            elif anchor == 'left':
                center = (position[0] + a, position[1])
            elif anchor == 'right':
                center = (position[0] - a, position[1])
            elif anchor == 'bottom':
                center = (position[0],position[1] + b)
            elif anchor == 'top':
                center = (position[0],position[1] - r)
            else:
                raise ValueError("No anchor position %s"%anchor)
        return center


class FilledCross(VisionEgg.Core.Stimulus):
    """Cross stimulus, consisting of a square with a rectangle aligned at each edge.
       Adapted from http://www.visionegg.org

    Parameters 
    ========== 
    anchor        -- specifies how position parameter is interpreted (String) (not implemented)
                     Default: center 
    anti_aliasing -- (Boolean) 
                     Default: True 
    center        -- DEPRECATED: don't use (Sequence2 of Real) 
                     Default: (determined at runtime) 
    color         -- (AnyOf(Sequence3 of Real or Sequence4 of Real)) 
                     Default: (1.0, 1.0, 1.0) 
    on            -- draw stimulus? (Boolean) (Boolean) 
                     Default: True 
    orientation   -- (Real) 
                     Default: 0.0 
    position      -- units: eye coordinates (AnyOf(Sequence2 of Real or Sequence3 of Real or Sequence4 of Real)) 
                     Default: (320.0, 240.0) 
	size          -- (Sequence2 of Real) edge length of square, edge length of whole stimulus
	                 units: eye coordinates (Real) 
                     Default: (50.0, 80.0)
    innerColor    -- if inner square of cross is to be colored, specify color (AnyOf(Sequence3 of Real or Sequence4 of Real)) 
				     Default: (1.0, 1.0, 1.0)
    """ 

    parameters_and_defaults = { 
        'on' : (True, 
                ve_types.Boolean, 
                "draw stimulus? (Boolean)"), 
        'color' : ((1.0,1.0,1.0), 
                   ve_types.AnyOf(ve_types.Sequence3(ve_types.Real), 
                                  ve_types.Sequence4(ve_types.Real))), 
		'innerColor' : ((0.0,0.0,0.0),
		                ve_types.AnyOf(ve_types.Sequence3(ve_types.Real), 
	                                  ve_types.Sequence4(ve_types.Real))),
        'anti_aliasing' : (False, 
                          ve_types.Boolean), 
        'orientation' : (0.0, 
                         ve_types.Real), 
        'position' : ((320.0, 240.0), 
                       ve_types.AnyOf(ve_types.Sequence2(ve_types.Real), 
                                      ve_types.Sequence3(ve_types.Real), 
                                      ve_types.Sequence4(ve_types.Real)), 
                       "units: eye coordinates"), 
        'anchor' : ('center', 
                    ve_types.String, 
                    "specifies how position parameter is interpreted"), 
        'size' : ((30.0, 70.0),
                  ve_types.Sequence2(ve_types.Real),
                  "units: eye coordinates"),
        'watermark_size' : (0.0,
                            ve_types.Real,
                            "units: eye coordinates"),
        'watermark_color' : ((0.0,0.0,0.0,0.0),
                             ve_types.Sequence4(ve_types.Real)),
        'center' : (None, 
                    ve_types.Sequence2(ve_types.Real), 
                    "DEPRECATED: don't use"), 
        } 

    __slots__ = ( 
        '_gave_alpha_warning', 
        ) 

    def __init__(self,**kw):
        VisionEgg.Core.Stimulus.__init__(self,**kw)
        self._gave_alpha_warning = 0

    def draw(self):
        p = self.parameters # shorthand 
        if p.center is not None: 
            if not hasattr(VisionEgg.config,"_GAVE_CENTER_DEPRECATION"): 
                logger = logging.getLogger('VisionEgg.MoreStimuli') 
                logger.warning("Specifying FilledTriangle by deprecated " 
                               "'center' parameter deprecated.  Use " 
                               "'position' parameter instead.  (Allows " 
                               "use of 'anchor' parameter to set to " 
                               "other values.)") 
                VisionEgg.config._GAVE_CENTER_DEPRECATION = 1 
            p.anchor = 'center' 
            p.position = p.center[0], p.center[1] # copy values (don't copy ref to tuple) 
        if p.on: 
            # calculate center 
            center = p.position #self._my_get_center(p.position,p.anchor,p.size) 
            gl.glMatrixMode(gl.GL_MODELVIEW) 
            gl.glPushMatrix() 
            gl.glTranslate(center[0],center[1],0.0) 
            gl.glRotate(p.orientation,0.0,0.0,1.0) 

            w = p.size[0]/2.0
            h = (p.size[1] - w)/2.0

            # paint and place square
            if len(p.innerColor)==3: 
                gl.glColor3f(*p.innerColor) 
            elif len(p.innerColor)==4: 
                gl.glColor4f(*p.innerColor)

            gl.glDisable(gl.GL_DEPTH_TEST) 
            gl.glDisable(gl.GL_TEXTURE_2D) 
            gl.glBlendFunc(gl.GL_SRC_ALPHA,gl.GL_ONE_MINUS_SRC_ALPHA) 
            gl.glEnable(gl.GL_BLEND)

            gl.glBegin(gl.GL_QUADS)
            gl.glVertex3f(-w,-w, 0.0)
            gl.glVertex3f( w,-w, 0.0)
            gl.glVertex3f( w, w, 0.0)
            gl.glVertex3f(-w, w, 0.0)
            gl.glEnd() # GL_QUADS

            # paint and place rectangles
            if len(p.color)==3: 
                gl.glColor3f(*p.color) 
            elif len(p.color)==4: 
                gl.glColor4f(*p.color)

            gl.glDisable(gl.GL_DEPTH_TEST) 
            gl.glDisable(gl.GL_TEXTURE_2D) 
            gl.glBlendFunc(gl.GL_SRC_ALPHA,gl.GL_ONE_MINUS_SRC_ALPHA) 
            gl.glEnable(gl.GL_BLEND)

            gl.glBegin(gl.GL_QUADS)
            gl.glVertex3f(w, w, 0.0)
            gl.glVertex3f(w, -w, 0.0)
            gl.glVertex3f(w+h, -w, 0.0)
            gl.glVertex3f(w+h, w, 0.0)
            gl.glEnd() # GL_QUADS

            gl.glBegin(gl.GL_QUADS)
            gl.glVertex3f(-w, w, 0.0)
            gl.glVertex3f(w, w, 0.0)
            gl.glVertex3f(w, w+h, 0.0)
            gl.glVertex3f(-w, w+h, 0.0)
            gl.glEnd() # GL_QUADS

            gl.glBegin(gl.GL_QUADS)
            gl.glVertex3f(-w, -w, 0.0)
            gl.glVertex3f(-w, w, 0.0)
            gl.glVertex3f(-(w+h), w, 0.0)
            gl.glVertex3f(-(w+h), -w, 0.0)
            gl.glEnd() # GL_QUADS

            gl.glBegin(gl.GL_QUADS)
            gl.glVertex3f(w, -w, 0.0)
            gl.glVertex3f(-w, -w, 0.0)
            gl.glVertex3f(-w, -(w+h), 0.0)
            gl.glVertex3f(w, -(w+h), 0.0)
            gl.glEnd() # GL_QUADS

            if not (p.watermark_size == 0.0):
                # paint and place watermark
                gl.glColor4f(*p.watermark_color)

                gl.glDisable(gl.GL_DEPTH_TEST) 
                gl.glDisable(gl.GL_TEXTURE_2D) 
                gl.glBlendFunc(gl.GL_SRC_ALPHA,gl.GL_ONE_MINUS_SRC_ALPHA) 
                gl.glEnable(gl.GL_BLEND)

                w2 = p.watermark_size/2
        
                gl.glBegin(gl.GL_QUADS)
                gl.glVertex3f(-w2, 0, 0.0)
                gl.glVertex3f(0, -w2, 0.0)
                gl.glVertex3f(w2, 0, 0.0)
                gl.glVertex3f(0, w2, 0.0)
                gl.glEnd() # GL_QUADS

            if p.anti_aliasing: 
                if not self._gave_alpha_warning: 
                    if len(p.color) > 3 and p.color[3] != 1.0: 
                        logger = logging.getLogger('VisionEgg.MoreStimuli') 
                        logger.warning("The parameter anti_aliasing is " 
                                       "set to true in the Target2D " 
                                       "stimulus class, but the color " 
                                       "parameter specifies an alpha " 
                                       "value other than 1.0.  To " 
                                       "acheive anti-aliasing, ensure " 
                                       "that the alpha value for the " 
                                       "color parameter is 1.0.") 
                        self._gave_alpha_warning = 1 

                # We've already drawn a filled polygon (aliased), now 
                # redraw the outline of the polygon (with 
                # anti-aliasing).  (Using GL_POLYGON_SMOOTH results in 
                # artifactual lines where triangles were joined to 
                # create quad, at least on some OpenGL 
                # implementations.) 

                # Calculate coverage value for each pixel of outline 
                # and store as alpha 
                gl.glEnable(gl.GL_LINE_SMOOTH) 
                # Now specify how to use the alpha value 
                gl.glBlendFunc(gl.GL_SRC_ALPHA,gl.GL_ONE_MINUS_SRC_ALPHA) 
                gl.glEnable(gl.GL_BLEND) 

                # Draw a second polygon in line mode, so the edges are anti-aliased 
                gl.glPolygonMode(gl.GL_FRONT_AND_BACK,gl.GL_LINE)

                gl.glBegin(gl.GL_QUADS)
                gl.glVertex3f(w, w, 0.0)
                gl.glVertex3f(w, -w, 0.0)
                gl.glVertex3f(w+h, -w, 0.0)
                gl.glVertex3f(w+h, w, 0.0)
                gl.glEnd() # GL_QUADS

                gl.glBegin(gl.GL_QUADS)
                gl.glVertex3f(-w, w, 0.0)
                gl.glVertex3f(w, w, 0.0)
                gl.glVertex3f(w, w+h, 0.0)
                gl.glVertex3f(-w, w+h, 0.0)
                gl.glEnd() # GL_QUADS

                gl.glBegin(gl.GL_QUADS)
                gl.glVertex3f(-w, -w, 0.0)
                gl.glVertex3f(-w, w, 0.0)
                gl.glVertex3f(-(w+h), w, 0.0)
                gl.glVertex3f(-(w+h), -w, 0.0)
                gl.glEnd() # GL_QUADS

                gl.glBegin(gl.GL_QUADS)
                gl.glVertex3f(w, -w, 0.0)
                gl.glVertex3f(-w, -w, 0.0)
                gl.glVertex3f(-w, -(w+h), 0.0)
                gl.glVertex3f(w, -(w+h), 0.0)
                gl.glEnd() # GL_QUADS

                # Set the polygon mode back to fill mode 
                gl.glPolygonMode(gl.GL_FRONT_AND_BACK,gl.GL_FILL) 
                gl.glDisable(gl.GL_LINE_SMOOTH) 
            gl.glPopMatrix()
