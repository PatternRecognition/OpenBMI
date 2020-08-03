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

from numpy import pi, array, concatenate, random, mgrid, zeros, vstack
from scipy.interpolate import bisplrep, bisplev

from .electrode import Electrode

pt = pi / 10.

# TODO growing number of vertices per ring
class ElectrodeMesh(object):
    def __init__(self):
        self.progress = array([])
        self.backlog_thresh = 20
        self.construct()

class ElectrodeMeshESGrid(ElectrodeMesh):
    def __init__(self, shape=(8, 8)):
        self.shape = shape
        ElectrodeMesh.__init__(self)

    def construct(self):
        self.x, self.y = mgrid[0:1.0000001:1./(self.shape[0] - 1),
                               0:1.0000001:1./(self.shape[1] - 1)]
        self.electrodes = zeros(self.shape)

    def set_random_voltage(self):
        self.electrodes = random.ranf(self.shape)

    def add_noise(self):
        self.electrodes += (random.ranf(self.shape) - 0.5) * 0.05

    def step(self):
        self.progress.append(self.electrodes.copy())
        overflow = len(self.progress) - self.backlog_thresh
        if overflow > 0:
            self.progress = self.progress[1:]

    def interpolate(self, shape):
        xnew, ynew = mgrid[0:1.0000001:1./shape[0], 0:1.0000001:1./shape[1]]
        tck = bisplrep(self.x, self.y, self.electrodes, s=0)
        return bisplev(xnew[:,0], ynew[0,:], tck)

class ElectrodeMesh61(ElectrodeMesh):
    def __init__(self):
        self.shape = (61,)
        ElectrodeMesh.__init__(self)

    def construct(self):
        self.x = array([
            -0.54393,
            -0.43094,
            -0.29608,
            -0.15033,
            0.00000,
            0.15033,
            0.29608,
            0.43094,
            0.54393,
            -0.56978,
            -0.41631,
            -0.25280,
            -0.08471,
            0.08471,
            0.25280,
            0.41631,
            0.56978,
            -0.54044,
            -0.36292,
            -0.18212,
            0.00000,
            0.18212,
            0.36292,
            0.54044,
            -0.66148,
            -0.47366,
            -0.28457,
            -0.09491,
            0.09491,
            0.28457,
            0.47366,
            0.66148,
            -0.57692,
            -0.38462,
            -0.19231,
            0.00000,
            0.19231,
            0.38462,
            0.57692,
            -0.66148,
            -0.47366,
            -0.28457,
            -0.09491,
            0.09491,
            0.28457,
            0.47366,
            0.66148,
            -0.54044,
            -0.36292,
            -0.18212,
            0.00000,
            0.18212,
            0.36292,
            0.54044,
            -0.43094,
            -0.29608,
            -0.15033,
            0.00000,
            0.15033,
            0.29608,
            0.43094
        ])
        self.y = array([
            0.54393,
            0.46645,
            0.41872,
            0.39282,
            0.38462,
            0.39282,
            0.41872,
            0.46645,
            0.54393,
            0.38817,
            0.33455,
            0.30404,
            0.29014,
            0.29014,
            0.30404,
            0.33455,
            0.38817,
            0.24230,
            0.21259,
            0.19712,
            0.19231,
            0.19712,
            0.21259,
            0.24230,
            0.13415,
            0.11331,
            0.10188,
            0.09677,
            0.09677,
            0.10188,
            0.11331,
            0.13415,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            0.00000,
            -0.13415,
            -0.11331,
            -0.10188,
            -0.09677,
            -0.09677,
            -0.10188,
            -0.11331,
            -0.13415,
            -0.24230,
            -0.21259,
            -0.19712,
            -0.19231,
            -0.19712,
            -0.21259,
            -0.24230,
            -0.46645,
            -0.41872,
            -0.39282,
            -0.38462,
            -0.39282,
            -0.41872,
            -0.46645
        ])
        self.names = [
            'F7',
            'F5',
            'F3',
            'F1',
            'Fz',
            'F2',
            'F4',
            'F6',
            'F8',
            'FFC7',
            'FFC5',
            'FFC3',
            'FFC1',
            'FFC2',
            'FFC4',
            'FFC6',
            'FFC8',
            'FC5',
            'FC3',
            'FC1',
            'FCz',
            'FC2',
            'FC4',
            'FC6',
            'CFC7',
            'CFC5',
            'CFC3',
            'CFC1',
            'CFC2',
            'CFC4',
            'CFC6',
            'CFC8',
            'C5',
            'C3',
            'C1',
            'Cz',
            'C2',
            'C4',
            'C6',
            'CCP7',
            'CCP5',
            'CCP3',
            'CCP1',
            'CCP2',
            'CCP4',
            'CP6',
            'CCP8',
            'CP5',
            'CP3',
            'CP1',
            'CPz',
            'CP2',
            'CP4',
            'CP6',
            'P5',
            'P3',
            'P1',
            'Pz',
            'P2',
            'P4',
            'P6']

    def set_random_voltage(self):
        self.electrodes = random.ranf(size=self.shape)

    def add_noise(self):
        self.electrodes += (random.ranf(self.electrodes.shape) - 0.5) * 0.05

    def step(self):
        # TODO initialize progress with zeros
        if not len(self.progress):
            self.progress = array([self.electrodes.copy()])
        else:
            self.progress = vstack((self.progress, self.electrodes.copy()))
        overflow = self.progress.shape[0] - self.backlog_thresh
        if overflow > 0:
            self.progress = self.progress[overflow:,:]

    @property
    def data(self):
        # FIXME remove .copy()
        return self.x, self.y, self.electrodes.copy()
