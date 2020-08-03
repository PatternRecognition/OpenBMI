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

from PyQt4 import QtCore, QtGui
from PyQt4.QtOpenGL import QGLWidget

from ..gl import GLSetup, Scene, Camera, CameraController
from ..gl.scene import HeadScene

class GLWidget(QGLWidget):
    def __init__(self, scene, shared=None):
        QGLWidget.__init__(self, None, shared)
        self.scene = scene
        self.type = '3dscalp'
        self.__init_attributes()

    def __init_attributes(self):
        self.gl_setup = GLSetup()
        #self.scene = HeadScene(self)
        #self.scene = TestScene()
        self.camera_ctl = CameraController()
        self.cameras = { 'std': Camera() }

        self.lastPos = QtCore.QPoint()

        self.trolltechGreen = QtGui.QColor.fromCmykF(0.40, 0.0, 1.0, 0.0)
        self.trolltechPurple = QtGui.QColor.fromCmykF(0.39, 0.39, 0.0, 0.0)

    def xRotation(self):
        return self.scene.xRot

    def yRotation(self):
        return self.scene.yRot

    def zRotation(self):
        return self.scene.zRot

    def setXRotation(self, angle):
        angle = self.normalizeAngle(angle)
        if angle != self.scene.xRot:
            self.scene.xRot = angle
            #self.emit(QtCore.SIGNAL("xRotationChanged(int)"), angle)
            self.updateGL()

    def setYRotation(self, angle):
        angle = self.normalizeAngle(angle)
        if angle != self.scene.yRot:
            self.scene.yRot = angle
            #self.emit(QtCore.SIGNAL("yRotationChanged(int)"), angle)
            self.updateGL()

    def setZRotation(self, angle):
        angle = self.normalizeAngle(angle)
        if angle != self.scene.zRot:
            self.scene.zRot = angle
            #self.emit(QtCore.SIGNAL("zRotationChanged(int)"), angle)
            self.updateGL()

    def initializeGL(self):
        self.qglClearColor(self.trolltechPurple.dark())
        self.camera_ctl.assume_camera(self.cameras['std'])
        self.scene.init()
        self.gl_setup.reinit()

    def paintGL(self):
        self.gl_setup.prepare_paint()
        self.camera_ctl.apply_camera()
        self.scene.draw()

    def resizeGL(self, width, height):
        self.camera_ctl.resize(width, height)

    def update_view(self):
        self.updateGL()

    def mousePressEvent(self, event):
        self.lastPos = QtCore.QPoint(event.pos())
        event.ignore()

    def mouseMoveEvent(self, event):
        dx = event.x() - self.lastPos.x()
        dy = event.y() - self.lastPos.y()

        if event.buttons() & QtCore.Qt.LeftButton:
            self.setXRotation(self.scene.xRot + 8 * dy)
            self.setYRotation(self.scene.yRot + 8 * dx)
        elif event.buttons() & QtCore.Qt.RightButton:
            self.setXRotation(self.scene.xRot + 8 * dy)
            self.setZRotation(self.scene.zRot + 8 * dx)

        self.lastPos = QtCore.QPoint(event.pos())

    def normalizeAngle(self, angle):
        while angle < 0:
            angle += 360 * 16
        while angle > 360 * 16:
            angle -= 360 * 16
        return angle

