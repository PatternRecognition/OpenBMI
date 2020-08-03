__copyright__ = """ Copyright (c) 2012 Torsten Schmits

This program is free software; you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by the
Free Software Foundation; either version 3 of the License, or (at your
option) any later version.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np, itertools, os

import pygame

from FeedbackBase.VisionEggFeedback import VisionEggFeedback
from lib.vision_egg.model.stimulus import CircleSector, Line, Triangle

__all__ = ['MapNavi']

def make_color(spec):
    return (pygame.Color(spec) if isinstance(spec, basestring) else
            pygame.Color(*spec)).normalize()

def rotation_matrix(angle):
    """ Angle in degrees. """
    angle = np.deg2rad(angle)
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array(((c, -s), (s, c)))

class Map(object):
    """ This class encapsules a texture object that serves as the game's
    map. It keeps track of its position and orientation as it changes
    them and provides functions for transforming map coordinates to
    screen coordinates.
    """

    def __init__(self, texture, zoom_factor, start_position):
        self._texture = texture
        self._texture.set_height(self._texture.height * zoom_factor)
        self._position = np.array((0, 0))
        self._angle = 0.0
        self._rotation = np.eye(2)
        self.move_to((0., 0.))

    def move_by(self, diff):
        self._position += diff
        self._texture.set(position=self._position)

    def move_to(self, position):
        self._position = position
        self._texture.set(position=self._position)

    def rotate(self, angle):
        self._angle += angle
        self._rotation = rotation_matrix(self._angle)
        self._texture.set(angle=self._angle)

    def coords_to_global(self, position):
        glbl = self._position + self.direction_to_global(position)
        return glbl

    def direction_to_local(self, direction):
        return np.dot(self._rotation.T, direction)

    def direction_to_global(self, direction):
        return np.dot(self._rotation, direction)

    def rotate_around(self, position, angle):
        diff = np.dot(rotation_matrix(angle), self._position - position)
        old_position = position + diff
        self.move_to(position)
        self.rotate(angle)
        self.move_to(old_position)

class HUD(list):
    """ This class contains the Stimulus instances for the player HUD.
    It also serves as a transformation proxy for movement and rotation
    relative to the player's position.
    Using the (de)select() methods, the direction circles can be
    activated, i.e. their color changed to red.
    """

    def __init__(self, screen, map, color_center, color_boundary,
                 color_selected, color_hud, position, radius, element_radius,
                 start_position, circle_count, line_width):
        self._map = map
        self._color_center = make_color(color_center)
        self._color_boundary = make_color(color_boundary)
        self._color_selected = make_color(color_selected)
        self._color_hud = make_color(color_hud)
        self._position = position
        self._radius = radius
        self._element_radius = element_radius
        self._circle_count = circle_count
        self._line_width = line_width
        self._selected = None
        self._head_size = .3
        super(HUD, self).__init__()
        self._setup_stimuli()
        self.move_to_map_coords(start_position)

    def _setup_stimuli(self):
        self._setup_lines()
        self._setup_head()
        self._setup_circles()

    def _setup_lines(self):
        circle = CircleSector(position=self._position,
                              radius=self._radius,
                              disk=False, end=180.,
                              color_edge=self._color_hud)
        self.append(circle)
        start = self._position - (self._radius, 0.)
        end = self._position + (self._radius, 0.)
        line = Line(color=self._color_hud, position=start, end=end)
        self.append(line)

    def _setup_head(self):
        head_radius = self._radius*self._head_size
        head = CircleSector(color=make_color('white'),
                            position=self._position,
                            radius=head_radius,
                            color_edge=self._color_hud,
                            circle_width=self._line_width,
                            end=180.)
        self.append(head)
        start = self._position - (head_radius, 0.)
        end = self._position + (head_radius, 0.)
        line = Line(color=self._color_hud, position=start, end=end, width=self._line_width)
        self.append(line)
        circle = CircleSector(color=self._color_center,
                              position=self._position,
                              radius=self._element_radius,
                              circle_width=self._line_width,
                              color_edge=self._color_hud)
        self.append(circle)
        nose = Triangle(position=self._position+(0., head_radius),
                        anchor='bottom', width=self._line_width, color=make_color('white'),
                        color_edge=self._color_hud, side=15.)
        self.append(nose)

    def _setup_circles(self):
        self._boundary = []
        angles = np.linspace(-np.pi/2., np.pi/2., self._circle_count)
        for position in np.vstack((np.sin(angles), np.cos(angles))).T:
            position = position * self._radius
            position += self._position
            circle = CircleSector(color=self._color_boundary, position=position,
                                  radius=self._element_radius,
                                  circle_width=self._line_width,
                                  color_edge=self._color_hud)
            self.append(circle)
            self._boundary.append(circle)

    def move_to_map_coords(self, position):
        glbl = self._map.coords_to_global(position)
        effective = self._position - glbl
        self._map.move_to(effective)

    def move_by_map_direction(self, direction):
        direction = self._map.direction_to_global(direction)
        self._map.move_by(-direction)

    def rotate(self, angle):
        self._map.rotate_around(self._position, -angle)

    def select(self, angle):
        angle = -angle + 90.
        index = int(round((angle * (self._circle_count - 1)) / 180.))
        if self._selected is not None:
            self._selected.set(color=self._color_boundary)
        self._selected = self._boundary[index]
        self._selected.set(color=self._color_selected)

    def deselect(self):
        self._selected.set(color=self._color_boundary)
        self._selected = None

class InputHandler(object):
    """ This class should be called with information about a movement,
    which consists of an angle and a step size. It caches this
    information and executes the movement in small fractions each time
    the step() method is called.
    """

    class _MovementFinished(Exception):
        pass

    class _Movement(object):
        def __init__(self, angle, direction, step_count):
            self.angle = angle
            step = (angle/step_count, direction/step_count)
            self._steps = itertools.repeat(step, step_count)

        @property
        def step(self):
            try:
                return next(self._steps)
            except StopIteration:
                raise InputHandler._MovementFinished()

    def __init__(self, hud, map, distance, step_count):
        self._hud = hud
        self._map = map
        self._distance = distance
        self._step_count = step_count
        self._movement = None
        self._movements = []

    def __call__(self, angle, stepsize):
        self._movements.append((angle, stepsize))

    def step(self):
        movement = self._current_movement
        if movement:
            try:
                angle, direction = movement.step
                self._hud.rotate(angle)
                self._hud.move_by_map_direction(direction)
            except self._MovementFinished:
                self._movement = None
                self._hud.deselect()
                self.step()

    @property
    def _current_movement(self):
        if self._movement is None and self._movements:
            self._movement = self._next_movement
            self._hud.select(self._movement.angle)
        return self._movement

    @property
    def _next_movement(self):
        angle, stepsize = self._movements.pop(0)
        angle = float(angle)
        radians = np.deg2rad(angle)
        direction = np.array((-np.sin(radians), np.cos(radians)))
        direction = (self._map.direction_to_local(direction) * stepsize *
                     self._distance)
        return self._Movement(angle, direction, self._step_count)

class MapNavi(VisionEggFeedback):

    def __init__(self, *a, **kw):
        self.running = False
        super(MapNavi, self).__init__(*a, **kw)

    def init_parameters(self):
        """ Explanation of parameters:
        interval: step time
        start_position: initial player coordinates relative to the
                        map (origin in lower left corner)
        zoom_factor: map magnification
        map_dir: place where map files are stored
        map_file: file name in map_dir subdirectory
        color_center: color of the player circle in the HUD center
        color_boundary: color of inactive direction circles
        color_selected: color of active direction circle
        color_hud: line color of the other HUD parts
        line_width: line width of the other HUD parts
        circle_count: number of direction circles
        hud_position: player HUD on-screen position in percent
        hud_radius: radius of the direction circle half-circle
        hud_element_radius: radius of direction circles
        transition_time: animation time for one movement
        """
        self.geometry = (100, 100, 1000, 675)
        self.interval = .02
        self.start_position = (500., 75.)
        self.zoom_factor = 2.
        self.map_dir = os.path.join(os.path.dirname(__file__), 'maps')
        self.map_file = 'test.png'
        self.color_center = 'cyan'
        self.color_boundary = 'orange'
        self.color_selected = 'red'
        self.color_hud = 'black'
        self.line_width = 3.
        self.circle_count = 7
        self.hud_position = (50, 10)
        self.hud_radius = 150
        self.hud_element_radius = 15
        self.transition_time = 5.
        self.wait_style_fixed = False

    def run(self):
        self._rescale()
        self._setup_map()
        self._setup_hud()
        self._setup_input_handler()
        self.stimulus_sequence(self._main(), self.interval).run()

    def _rescale(self):
        size = np.array(self.screen_size) / 100.
        self._hud_position = size * np.array(self.hud_position)

    def _setup_map(self):
        map_file = os.path.join(self.map_dir, self.map_file)
        texture = self.add_image_stimulus(anchor='lowerleft')
        texture.set_file(map_file)
        self._map = Map(texture, self.zoom_factor, self.start_position)

    def _setup_hud(self):
        self._hud = HUD(self._view.screen, self._map, self.color_center,
                        self.color_boundary, self.color_selected,
                        self.color_hud, self._hud_position, self.hud_radius,
                        self.hud_element_radius, self.start_position,
                        self.circle_count, self.line_width)
        self.add_stimuli(*self._hud)

    def _setup_input_handler(self):
        step_count = int(self.transition_time / self.interval)
        self._input_handler = InputHandler(self._hud, self._map,
                                           self.hud_radius, step_count)

    def _main(self):
        self.running = True
        while self._flag:
            self._input_handler.step()
            yield
        self.running = False

    def on_control_event(self, data):
        if data.has_key('cl_output'):
            angle = data['cl_output']
            stepsize = data.get('stepsize', 1)
            self._input_handler(angle, stepsize)

if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.DEBUG)
    feedback = MapNavi()
    feedback.on_init()
    feedback.on_play()
