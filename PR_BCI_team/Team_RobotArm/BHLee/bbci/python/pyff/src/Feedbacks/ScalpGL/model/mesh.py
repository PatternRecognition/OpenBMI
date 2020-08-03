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

from itertools import izip
from numpy import array, asarray

from ..gl.painter import Painter

def a2s(coords):
    """ Return a perenthesized coordinate list out of an array, because
    numpy array string representation is ugly.

    """
    coords = ', '.join(str(c) for c in coords)
    return '(%s)' % coords

class Vertex(object):
    def __init__(self, coords, normal=None, color=None):
        self.coords = coords
        self.normal = normal
        self.color = color

    def __asarray__(self):
        return self.coords

    @property
    def has_color(self):
        return self.color is not None

    @property
    def has_normal(self):
        return self.normal is not None

    def __str__(self):
        string = 'V%s' % a2s(self.coords)
        if self.has_normal:
            string += ', N%s' % a2s(self.normal)
        if self.has_color:
            string += ', C%s' % a2s(self.color)
        return string

class Polygon(object):
    def __init__(self, indices, normal=None):
        self.indices = indices
        self.normal = normal

    def __array__(self):
        return self.indices

    def __getitem__(self, index):
        return self.indices.__getitem__(index)

    @property
    def has_normal(self):
        return self.normal is not None
        
class Mesh(object):
    def __init__(self, vertices, faces):
        self.vertices = vertices
        self.faces = faces

    def draw(self):
        Painter.triangle_mesh(self)

class TriangleStripMesh(object):
    opacity = 0.7

    def __init__(self, vertices):
        self.vertices = vertices

    def draw(self):
        Painter.triangle_strip_mesh(self)

    def set_colors(self, colors):
        for row in colors:
            asarray(row)[:,3] = self.opacity
        for crow, vrow in izip(colors, self.vertices):
            for c, v in izip(crow, vrow):
                v.color = c

class TriangleStripMesh61(TriangleStripMesh):
    def set_colors(self, colors):
        colors = [colors[0:9], colors[9:17], colors[17:24], colors[24:32],
                        colors[32:39], colors[39:47], colors[47:54],
                        colors[54:]]
        TriangleStripMesh.set_colors(self, colors)

    def draw(self):
        Painter.triangle_strip_mesh_padded(self)
        Painter.points(self)

