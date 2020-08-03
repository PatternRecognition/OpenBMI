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

from copy import deepcopy

from numpy import array

class Camera(object):
    params = { 'position': array([0, 0, 10]), 'look_at': array([0, 0, 0]),
              'up': array([0, 1, 0]), 'vertical_angle': 45. }

    def __init__(self, **params):
        for p in Camera.params.iteritems():
            setattr(self, p[0], params.get(p[0], deepcopy(p[1])))

    def __str__(self):
        return 'Camera: { %s }' % ', '.join([p + ': ' + str(getattr(self, p)) for p, v
                                     in Camera.params.iteritems()])

