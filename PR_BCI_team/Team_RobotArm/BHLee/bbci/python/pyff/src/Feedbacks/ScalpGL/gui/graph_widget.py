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

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.mlab import griddata
from pylab import figure, contourf, show, colorbar, scatter, plot
from numpy import linspace, min, max

class ScalpPlot2D(FigureCanvasQTAgg):
    def __init__(self, electrode_mesh):
        self.type = '2dscalp'
        self.figure = Figure(figsize=[2, 2])
        self.electrode_mesh = electrode_mesh
        FigureCanvasQTAgg.__init__(self, self.figure)

    def update_view(self):
        self.plot()

    def plot(self):
        self.figure.clear()
        self.axes = self.figure.add_subplot(111)
        x, y, z = self.electrode_mesh.data
        xmin = min(x)
        ymin = min(y)
        xmax = max(x)
        ymax = max(y)
        xi = linspace(xmin, xmax, 100)
        yi = linspace(ymin, ymax, 100)
        zi = griddata(x, y, z, xi, yi)
        cax = self.axes.pcolor(xi, yi, zi)
        self.axes.scatter(x, y, marker='o', c='b', s=5)
        self.figure.colorbar(cax)
        self.draw()

class SignalPlot(FigureCanvasQTAgg):
    def __init__(self, electrode_mesh):
        self.type = 'signal'
        self.config = {}
        self.config['name'] = 'Cz'
        self.config['fixed'] = True
        self.figure = Figure(figsize=[2, 2])
        self.electrode_mesh = electrode_mesh
        FigureCanvasQTAgg.__init__(self, self.figure)

    def update_view(self):
        self.plot()

    def plot(self):
        name = self.config['name']
        self.figure.clear()
        self.axes = self.figure.add_subplot(111, title=name)
        i = self.electrode_mesh.names.index(name)
        y = self.electrode_mesh.progress[:, i]
        self.axes.plot(y)
        if self.config['fixed']:
            self.axes.set_xlim(0., 1.)
            self.axes.set_ylim(0., 1.)
        self.draw()

    def mousePressEvent(self, event):
        event.ignore()
