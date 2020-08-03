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

from logging import debug

from OpenGL import GL, GLU
from camera import Camera

class CameraController(object):
    def __init__(self, camera=None):
        self.__init_attributes()
        if camera is not None:
            self.assume_camera(camera)

    def __init_attributes(self):
        self.camera = None
        self.has_camera = False
        self.width = 0
        self.height = 0
        self.aspect_ratio = 1

    def assume_camera(self, camera):
        assert(isinstance(camera, Camera))
        self.camera = camera
        self._apply_projection()
        self.has_camera = True

    def apply_camera(self):
        if self.has_camera:
            c = self.camera
            p, l, u = c.position, c.look_at, c.up
            GL.glMatrixMode(GL.GL_MODELVIEW)
            GL.glLoadIdentity()
            GLU.gluLookAt(p[0], p[1], p[2], l[0], l[1], l[2], u[0], u[1], u[2])
        else:
            debug('CameraController.apply_camera: No camera set yet')

    def resize(self, width, height):
        self.width, self.height = width, height
        self.aspect_ratio = float(width)/height
        self._apply_projection()

    def _apply_projection(self):
        c = self.camera
        side = min(self.width, self.height)
        #GL.glViewport((self.width - side) / 2, (self.height - side) / 2, side,
                      #side)
        GL.glViewport(0, 0, self.width, self.height)
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        #GL.glOrtho(-0.5, +0.5, +0.5, -0.5, 0.1, 1000.0)
        GLU.gluPerspective(self.camera.vertical_angle, self.aspect_ratio,
                           0.1, 1000)
