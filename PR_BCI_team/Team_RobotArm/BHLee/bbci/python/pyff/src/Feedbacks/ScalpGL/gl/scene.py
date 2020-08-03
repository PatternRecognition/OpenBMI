__copyright__ = """ Copyright (c) 2008, 2009 Torsten Schmits

This file is part of the pyff framework. pyff is free software;
you can redistribute it and/or modify it under the terms of the GNU General
Public License version 2, as published by the Free Software Foundation.

pyff is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
details.

You should have received a copy of the GNU General Public License along with
this program; if not, write to the Free Software Foundation, Inc., 59 Temple
Place, Suite 330, Boston, MA  02111-1307  USA

"""

from PyQt4 import QtCore, QtGui, QtOpenGL

from OpenGL import GL

from .display_list import DisplayList

class Scene(object):
    def draw(self):
        return
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadIdentity()

class HeadScene(Scene):
    def __init__(self, head, cap):
        self.head = head
        self.cap = cap
        self.xRot = 1
        self.yRot = 1
        self.zRot = 1

    def init(self):
        light_position = (10.0, 10.0, 10.0, 1.0)
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_POSITION, light_position)
        GL.glEnable(GL.GL_LIGHT0)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_BLEND)
        #GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE)
        #self.head.generate()
        #self.cap.generate()
        self.head.init()
        self.cap.init()

    def draw(self):
        Scene.draw(self)
        GL.glPushMatrix()
        GL.glRotated(self.xRot / 16.0, 1.0, 0.0, 0.0)
        GL.glRotated(self.yRot / 16.0, 0.0, 1.0, 0.0)
        GL.glRotated(self.zRot / 16.0, 0.0, 0.0, 1.0)
        GL.glRotated(-90., 1.0, 0.0, 0.0)
        GL.glBlendFunc(GL.GL_ONE, GL.GL_ZERO)
        GL.glEnable(GL.GL_LIGHTING)
        self.head.call()
        GL.glDisable(GL.GL_LIGHTING)
        GL.glTranslate(0., 0.5, 0.7)
        GL.glRotated(90., 1.0, 0.0, 0.0)
        GL.glScale(1.5, 1.5, 1.5)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        self.cap.call()
        GL.glPopMatrix()

class HeadScene61(Scene):
    def __init__(self, head, cap):
        self.head = head
        self.cap = cap
        self.xRot = 1
        self.yRot = 1
        self.zRot = 1

    def init(self):
        light_position = (10.0, 10.0, 10.0, 1.0)
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_POSITION, light_position)
        GL.glEnable(GL.GL_LIGHT0)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_BLEND)
        #GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE)
        #self.head.generate()
        #self.cap.generate()
        self.head.init()
        self.cap.init()

    def draw(self):
        Scene.draw(self)
        GL.glPushMatrix()
        GL.glRotated(self.xRot / 16.0, 1.0, 0.0, 0.0)
        GL.glRotated(self.yRot / 16.0, 0.0, 1.0, 0.0)
        GL.glRotated(self.zRot / 16.0, 0.0, 0.0, 1.0)
        GL.glRotated(-90., 1.0, 0.0, 0.0)
        GL.glBlendFunc(GL.GL_ONE, GL.GL_ZERO)
        GL.glEnable(GL.GL_LIGHTING)
        self.head.call()
        GL.glDisable(GL.GL_LIGHTING)
        GL.glTranslate(0., 0.5, 0.7)
        #GL.glRotated(90., 1.0, 0.0, 0.0)
        GL.glScale(1.5, 1.5, 1.5)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        self.cap.call()
        GL.glPopMatrix()
