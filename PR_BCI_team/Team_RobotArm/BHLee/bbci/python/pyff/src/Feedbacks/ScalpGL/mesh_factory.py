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

from os import path

import numpy as n
from numpy import array, asarray

from .model.errors import MeshParseError
from .model import Mesh, Vertex, Polygon, TriangleStripMesh, TriangleStripMesh61
from .util.metadata import datadir

def sphere_vertex(theta, phi):
    """ Given two radian angles, return cartesian coordinates of the
    corresponding unit sphere location as Vertex instance with normal
    set to its coordinates.
    
    """
    coords = array([n.sin(theta) * n.sin(phi), 
                    n.sin(theta) * n.cos(phi),
                    n.cos(theta)])
    return Vertex(coords, normal=coords)

def sphere_coords(theta, phi):
    """ Given two radian angles, return cartesian coordinates of the
    corresponding unit sphere location as Vertex instance with normal
    set to its coordinates.
    
    """
    return array([n.sin(theta) * n.sin(phi), 
                  n.sin(theta) * n.cos(phi),
                  n.cos(theta)])

class MeshFactory(object):
    meshdir = path.join(datadir, 'meshes')

    def __init__(self, data):
        self.data = data

    @classmethod
    def normal(self, vertices, face):
        """ Conveniently calculate a face normal.
        vertices is an array of coordinates into which face's indices
        point.

        """
        v = vertices
        f = face
        a = n.asarray
        return n.cross(a(v[f[2]]) - a(v[f[0]]), a(v[f[1]]) - a(v[f[0]]))

    @classmethod
    def mesh(self, vertices, faces):
        """ Turn an array of coordinates and indices into a Mesh
        instance, calculating normals for each vertex.

        """
        faces = [Polygon(f, MeshFactory.normal(vertices, f)) for f in faces]
        vertices = [Vertex(v, n.zeros((1, 3))) for v in vertices]
        for face in faces:
            for f in array(face):
                vertices[f].normal += face.normal
        return Mesh(vertices, faces)

    @classmethod
    def mesh_from_file(self, name):
        filename = name + '.mesh'
        filepath = path.abspath(path.join(MeshFactory.meshdir, filename))
        with open(filepath) as file:
            try:
                content = [l.rstrip('\n') for l in file.readlines()]
                vertices_begin = content.index('vertices')
                faces_begin = content.index('faces')
            except ValueError, e:
                raise MeshParseError()
        vertices = array([array([float(n) for n in l.split()]) for l in
                    content[vertices_begin + 1:faces_begin]])
        faces = [array([int(n) for n in l.split()]) for l in content[faces_begin
                                                                     + 1:]]
        return MeshFactory.mesh(vertices, faces)

    @classmethod
    def mesh_from_file_3ds(self, name):
        filename = name + '.mesh'
        filepath = path.abspath(path.join(MeshFactory.meshdir, filename))
        with open(filepath) as file:
            try:
                content = [l.rstrip('\n') for l in file.readlines()]
                matrix_begin = content.index('matrix')
                vertices_begin = content.index('vertices')
                faces_begin = content.index('faces')
            except ValueError, e:
                raise MeshParseError()
        matrix = array([[float(x) for x in l.split()] for l in
                        content[matrix_begin + 1:vertices_begin]])
        vertices = array([array([float(x) for x in l.split()]) for l in
                    content[vertices_begin + 1:faces_begin]])
        faces = [array([int(x) for x in l.split()]) for l in content[faces_begin
                                                                     + 1:]]
        v2 = array([n.dot(n.linalg.inv(matrix), n.append(v, 1.)) for v in vertices])[:,:3]
        print vertices[:10]
        print v2
        m = Mesh(vertices, faces)
        return m

    @classmethod
    def write_mesh(self, mesh, name):
        filename = name + '.mesh'
        filepath = path.abspath(path.join(MeshFactory.meshdir, filename))
        with open(filepath, 'w') as file:
            try:
                file.write('vertices\n')
                for v in mesh.vertices:
                    file.write(' '.join(str(c) for c in array(v)) + '\n')
                file.write('faces\n')
                for f in mesh.faces:
                    file.write(' '.join(str(c) for c in array(f)) + '\n')
            except:
                raise

    @classmethod
    def hemisphere(self, complexity=100):
        step = n.pi / (complexity - 1)
        trange = n.arange(0., n.pi + step, step)
        prange = n.arange(-n.pi / 2., n.pi / 2. + step, step)
        vertices = [[sphere_vertex(theta, phi) for phi in prange] 
                    for theta in trange]
        return TriangleStripMesh(vertices)

    @classmethod
    def cap_61(self):
        # coords {{{
        coords = [
            [[-0.70711, -0.65328, -0.50000, -0.27060, 0.00000, 0.27060, 0.50000, 0.65328, 0.70711],
            [0.70711, 0.70711, 0.70711, 0.70711, 0.70711, 0.70711, 0.70711, 0.70711, 0.70711],
            [0.00000, 0.27060, 0.50000, 0.65328, 0.70711, 0.65328, 0.50000, 0.27060, 0.00000]],

            [[-0.81549, -0.69134, -0.46194, -0.16221, 0.16221, 0.46194, 0.69134, 0.81549],
            [0.55557, 0.55557, 0.55557, 0.55557, 0.55557, 0.55557, 0.55557, 0.55557],
            [0.16221, 0.46194, 0.69134, 0.81549, 0.81549, 0.69134, 0.46194, 0.16221]],

            [[-0.85355, -0.65328, -0.35355, 0.00000, 0.35355, 0.65328, 0.85355],
            [0.38268, 0.38268, 0.38268, 0.38268, 0.38268, 0.38268, 0.38268],
            [0.35355, 0.65328, 0.85355, 0.92388, 0.85355, 0.65328, 0.35355]],

            [[-0.96194, -0.81549, -0.54490, -0.19134, 0.19134, 0.54490, 0.81549, 0.96194],
            [0.19509, 0.19509, 0.19509, 0.19509, 0.19509, 0.19509, 0.19509, 0.19509],
            [0.19134, 0.54490, 0.81549, 0.96194, 0.96194, 0.81549, 0.54490, 0.19134]],

            [[-0.92388, -0.70711, -0.38268, 0.00000, 0.38268, 0.70711, 0.92388],
            [0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000],
            [0.38268, 0.70711, 0.92388, 1.00000, 0.92388, 0.70711, 0.38268]],

            [[-0.96194, -0.81549, -0.54490, -0.19134, 0.19134, 0.54490, 0.81549, 0.96194],
            [-0.19509, -0.19509, -0.19509, -0.19509, -0.19509, -0.19509, -0.19509, -0.19509],
            [0.19134, 0.54490, 0.81549, 0.96194, 0.96194, 0.81549, 0.54490, 0.19134]],

            [[-0.85355, -0.65328, -0.35355, 0.00000, 0.35355, 0.65328, 0.85355],
            [-0.38268, -0.38268, -0.38268, -0.38268, -0.38268, -0.38268, -0.38268],
            [0.35355, 0.65328, 0.85355, 0.92388, 0.85355, 0.65328, 0.35355]],

            [[-0.65328, -0.50000, -0.27060, 0.00000, 0.27060, 0.50000, 0.65328],
            [-0.70711, -0.70711, -0.70711, -0.70711, -0.70711, -0.70711, -0.70711],
            [0.27060, 0.50000, 0.65328, 0.70711, 0.65328, 0.50000, 0.27060]]] # }}}
        vertices = [[Vertex(asarray(row)[:, i]) for i in range(len(row[0]))] for row in coords]
        return TriangleStripMesh61(vertices)
