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

from __future__ import with_statement

from itertools import izip

from numpy import cross

from OpenGL import GL

class Painting(object):
    """ Context manager that handles GL parameters.

    """
    def __init__(self, poly_type):
        self.poly_type = poly_type

    def __enter__(self):
        """ Start drawing vertices with the given parameters.
        @param poly_type: The value to pass to glBegin()

        """
        GL.glBegin(self.poly_type)

    def __exit__(self, exc_type, exc_val, exc_tb):
        GL.glEnd()

class Painter(object):
    @classmethod
    def vertex(self, v):
        GL.glVertex3d(*v.coords)

    @classmethod
    def normal(self, n):
        GL.glNormal(*n.normal)

    @classmethod
    def color(self, n):
        GL.glColor4f(*n.color)

    @classmethod
    def triangle_mesh(self, mesh):
        with Painting(GL.GL_TRIANGLES) as p:
            mat_ambient = (0.6, 0.5, 0.4)
            mat_diffuse = (0.77, 0.70, 0.48)
            mat_specular = (0.1, 0.1, 0.1)
            mat_shininess = 0.25
            GL.glMaterial(GL.GL_FRONT, GL.GL_DIFFUSE, mat_diffuse)
            GL.glMaterial(GL.GL_FRONT, GL.GL_AMBIENT, mat_ambient)
            GL.glMaterial(GL.GL_FRONT, GL.GL_SPECULAR, mat_specular)
            GL.glMaterial(GL.GL_FRONT, GL.GL_SHININESS, mat_shininess)
            v = mesh.vertices 
            for face in mesh.faces:
                for i in face:
                    vertex = mesh.vertices[i]
                    Painter.normal(vertex)
                    Painter.vertex(vertex)

    @classmethod
    def triangle_strip_mesh(self, mesh):
        v = mesh.vertices
        for lrows, urows in izip(v[:-1], v[1:]):
            with Painting(GL.GL_TRIANGLE_STRIP) as p:
                for lower, upper in izip(lrows, urows):
                    #Painter.normal(lower)
                    Painter.color(lower)
                    Painter.vertex(lower)
                    #Painter.normal(upper)
                    Painter.color(upper)
                    Painter.vertex(upper)

    @classmethod
    def triangle_strip_mesh_padded(self, mesh):
        v = mesh.vertices
        for lrows, urows in izip(v[:-1], v[1:]):
            with Painting(GL.GL_TRIANGLE_STRIP) as p:
                for lower, upper in map(None, lrows, urows):
                    if lower is not None:
                        Painter.color(lower)
                        Painter.vertex(lower)
                    if upper is not None:
                        Painter.color(upper)
                        Painter.vertex(upper)

    @classmethod
    def points(self, mesh):
        GL.glPointSize(3.)
        with Painting(GL.GL_POINTS) as p:
            for row in mesh.vertices:
                for v in row:
                    GL.glColor4f(0., 0., 0., 1.)
                    Painter.vertex(v)
