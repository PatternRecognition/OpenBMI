__copyright__ = """ Copyright (c) 2011-2012 Torsten Schmits

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

import os, logging, shutil, copy, threading, time
from datetime import datetime
from itertools import cycle

import pygame
from pygame import joystick

import scipy.misc
import numpy as np
from numpy import array

from VisionEgg.MoreStimuli import FilledCircle

from FeedbackBase.VisionEggFeedback import VisionEggFeedback

__all__ = ['ZweiHand']

def make_color(spec):
    return (pygame.Color(spec) if isinstance(spec, basestring) else
            pygame.Color(*spec))

def bresenham(a, b):
    sign = lambda x: (x > 0) * 2 - 1
    x0, y0 = a
    x1, y1 = b
    steep = abs(y1 - y0) > abs(x1 - x0)
    if steep:
        x0, y0 = y0, x0
        x1, y1 = y1, x1
    deltax, deltay = x1 - x0, y1 - y0
    xstep, ystep = sign(deltax), sign(deltay)
    error = 0.
    deltaerror = abs(deltay / deltax)
    y = y0
    for x in xrange(int(x0), int(x1+xstep), xstep):
        yield np.array(map(int, ((y, x) if steep else (x, y))))
        error += deltaerror
        if error >= .5:
            y += ystep
            error -= 1
    
class ZweiHandError(Exception):
    pass

class Grid(object):
    """ This class holds the image data of a track.
    Calling resize() creates a copy with the desired dimensions and
    summed up color values for more efficient and convenient testing.
    value() returns the color sum at the given coordinates.
    in_a() and in_b() tell if a pixel is inside the start/finish area.
    """

    def __init__(self, values, start_value, finish_value, initial_position_value, size, keep_aspect_ratio):
        self._original_values = values
        self._start_value = start_value
        self._finish_value = finish_value
        self._initial_position_value = initial_position_value
        self._keep_aspect_ratio = keep_aspect_ratio
        self.scale_coords = np.array([1., 1.])
        self.scale = 1.
        self.resize(size)

    def resize(self, size):
        dshape = self._original_values.shape[:2]
        if size != dshape:
            if self._keep_aspect_ratio:
                frac = min(np.array(map(float, size)) / dshape)
                size = (np.array(dshape) * frac).astype(int)
            self.scale_coords = size.astype(float) / dshape
            self.scale = self.scale_coords[0]
            data = scipy.misc.imresize(self._original_values, size, interp='nearest')
            #data = scipy.misc.imresize(self._original_values, size)
        else:
            data = self._original_values
        self.size = size
        self.aspect_ratio = float(size[0]) / size[1]
        self._values = data[:,:,:3].sum(axis=2).transpose()
        self._detect_start_finish()

    def _detect_start_finish(self):
        def corners_for_value(value):
            x, y = np.nonzero(self._values == value)
            return array((min(x), min(y))), array((max(x), max(y)))
        self._a = corners_for_value(self._start_value)
        self._b = corners_for_value(self._finish_value)
        self._initial_position = corners_for_value(self._initial_position_value)

    @property
    def height(self):
        return self._values.shape[1]

    @property
    def a(self):
        return (self._a[0] + self._a[1]) / 2.

    @property
    def b(self):
        return (self._b[0] + self._b[1]) / 2.

    @property
    def initial_position(self):
        return (self._initial_position[0] + self._initial_position[1]) / 2.

    def _in_area(self, ul, lr, position):
        return (ul[0] < position[0] < lr[0] and
                ul[1] < position[1] < lr[1])

    def in_a(self, position):
        return self._in_area(*self._a, position=position)

    def in_b(self, position):
        return self._in_area(*self._b, position=position)

    def value(self, position):
        return self._values[position[0],position[1]]

class Cursor(FilledCircle):
    def __init__(self, speed, color_inactive, color_valid, color_critical,
                 radius=1., map_to_course=False, **kw):
        super(Cursor, self).__init__(radius=radius, **kw)
        self._absolute_speed = self._speed = speed
        self._color_inactive = color_inactive
        self._color_valid = color_valid
        self._color_critical = color_critical
        self.radius = radius
        self._map_to_course = map_to_course
        self._position = array(self.parameters.position)
        self._moved = False

    def set_grid(self, grid):
        self._grid = grid
        self.resize()
        self.reset()

    def resize(self):
        new_radius = self.radius * self._grid.scale
        self.set(radius=new_radius)
        self._speed = self._absolute_speed * self._grid.scale

    def reset(self):
        self._set_color(self._color_inactive)
        if self._map_to_course:
            self.move(self._grid.initial_position)
        self._last_position = self._position
        self._moved = False

    def move(self, position):
        self._last_position = self._position
        position = array(position)
        if not np.allclose(position, self._position):
            self._moved = True
        position = np.maximum(np.zeros(2), position)
        position = np.minimum(self._grid.size[::-1], position)
        self._position = position
        mapped = (position[0], self._grid.height - position[1])
        self.set(position=mapped)

    def move_diff(self, x, y):
        self.move(self._position + array((x, y)) * self._speed)

    def move_absolute(self, x, y):
        ratio = self._grid.aspect_ratio
        if ratio > 1:
            x /= ratio
        else:
            y /= ratio
        x = (x/1.3 + 1.) / 2.
        y = (y/1.6 + 1.) / 2.
        self.move(self._grid.size[::-1] * np.array((x, y)))

    @property
    def position(self):
        return map(int, self._position)

    @property
    def has_moved(self):
        moved = self._moved
        self._moved = False
        return moved

    @property
    def last_trajectory(self):
        return bresenham(self._last_position, self._position)

    def show(self):
        self.set(on=True)

    def hide(self):
        self.set(on=False)

    def _set_color(self, color):
        self.set(color=make_color(color).normalize())
        
    def set_valid(self):
        self._set_color(self._color_valid)

    def set_critical(self):
        self._set_color(self._color_critical)

def total_seconds(td):
    return ((td.microseconds + (td.seconds + td.days * 24 * 3600) * 10**6) /
            float(10**6))
    
class TimeTracker(object):
    def __init__(self):
        self.reset()

    def start(self):
        self._start = datetime.now()

    def stop(self):
        if self._start is not None:
            self.times.append((self._start, datetime.now()))
        self._start = None

    @property
    def durations(self):
        return [total_seconds(d[1] - d[0]) for d in self.times]

    @property
    def overall(self):
        return sum(self.durations, 0.)

    def reset(self):
        self.times = []
        self._start = None

class PositionTracker(object):
    class PositionConfirmed(Exception):
        pass

    class TrialStopped(Exception):
        pass

    def __init__(self, view, cursor, critical_value, fatal_value,
                 track_value, synthie, reset_pause):
        self._view = view
        self._cursor = cursor
        self._critical_value = critical_value
        self._fatal_value = fatal_value
        self._track_value = track_value
        self._synthie = synthie
        self._reset_pause = reset_pause
        self._grid = None
        self._fatal_pause = False
        self.global_timer = TimeTracker()
        self.critical_timer = TimeTracker()
        self.idle_timer = TimeTracker()

    def set_grid(self, grid):
        self._grid = grid
        self.global_timer.reset()
        self.critical_timer.reset()
        self.idle_timer.reset()
        self.reset()

    def moving(self):
        if self._grid is None:
            raise ZweiHandError('No grid set when using PositionTracker!')
        elif not self._fatal_pause:
            self._unidle()
            try:
                for position in self._cursor.last_trajectory:
                    self._current_position = position
                    self._test_position()
            except self.TrialStopped:
                pass

    def _test_position(self):
        try:
            self._test_start()
            self._test_finish()
            self._test_fatal()
            self._test_critical()
            self._test_track()
        except self.PositionConfirmed:
            pass

    def _test_start(self):
        if self._grid.in_a(self._current_position):
            if not self._entered_start:
                self.global_timer.start()
                self._entered_start = True
                self._synthie.play('start')
            raise self.PositionConfirmed()
        elif not self._entered_start:
            raise self.PositionConfirmed()

    def _test_finish(self):
        if self._grid.in_b(self._current_position):
            self.finished = True
            self._synthie.play('finish')
            self.global_timer.stop()
            self._cursor.set_valid()
            raise self.TrialStopped()

    def _test_critical(self):
        if self._current_value == self._critical_value:
            if self._entered_course:
                if not self._critical:
                    self._enter_critical()
            raise self.PositionConfirmed()
        else:
            if self._critical:
                self._leave_critical()

    def _enter_critical(self):
        if self._entered_course:
            self._synthie.play('critical')
            self.critical_timer.start()
            self._cursor.set_critical()
        self._critical = True

    def _leave_critical(self):
        if self._entered_course:
            self.critical_timer.stop()
            self._cursor.set_valid()
        self._critical = False

    def _test_fatal(self):
        if self._entered_start and self._current_value == self._fatal_value:
            self._fatal()
            raise self.TrialStopped()

    def _fatal(self):
        self._cursor.hide()
        self.critical_timer.stop()
        self.global_timer.stop()
        self._synthie.play('fatal')
        self._fatal_pause = True
        threading.Timer(self._reset_pause, self._end_fatal_pause).start()

    def reset(self):
        self._cursor.reset()
        self._critical = False
        self._entered_course = False
        self._entered_start = False
        self.finished = False
        self._idle = False

    def _test_track(self):
        if (self._current_value == self._track_value and not
            self._entered_course):
                self._start_track()

    def _start_track(self):
        self._entered_course = True
        self._cursor.set_valid()

    @property
    def _current_value(self):
        return self._grid.value(self._current_position)

    def _end_fatal_pause(self):
        self._fatal_pause = False
        self.reset()
        self._cursor.show()

    def idle(self):
        if self._entered_start and not self._idle:
            self._idle = True
            self.idle_timer.start()

    def _unidle(self):
        if self._entered_start and self._idle:
            self._idle = False
            self.idle_timer.stop()

class InputHandler(object):
    def __init__(self, cursor, x_axis_id, y_axis_id, joystick_style='absolute',
                 movement_threshold=0, map_to_course=False):
        self._cursor = cursor
        self._x_axis_id = x_axis_id
        self._y_axis_id = y_axis_id
        self._x_axis = getattr(self, '_x_axis_{0}'.format(joystick_style))
        self._y_axis = getattr(self, '_y_axis_{0}'.format(joystick_style))
        self._button_values = {}
        self._movement_threshold = movement_threshold
        self._logger = logging.getLogger('ZweiHand')
        self._process_button_events = joystick_style == 'button_push'
        if self._process_button_events:
            self._setup_button_push_mode()
        self._map_to_course = joystick_style == 'absolute' and map_to_course
        self._setup_keyboard()
        self._init_joysticks()
        self._setup_axes()

    def _setup_button_push_mode(self):
        if len(self._x_axis_id) != 3 and len(self._y_axis_id) != 3:
            text = ('axis_id parameters must contain joystick and two button'
                    ' IDs!')
            raise ZweiHandError(text)
        x, y = self._x_axis_id, self._y_axis_id
        self._button_values = { (x[0], x[1]): (1, 0),
                                (x[0], x[2]): (-1, 0),
                                (y[0], y[1]): (0, 1),
                                (y[0], y[2]): (0, -1) }
        self._button_value = np.array((0., 0.))

    def _setup_keyboard(self):
        self._key_values = { pygame.K_UP: np.array((0, -1)),
                             pygame.K_DOWN: np.array((0, 1)),
                             pygame.K_LEFT: np.array((-1, 0)),
                             pygame.K_RIGHT: np.array((1, 0)) }
        self.keyboard_value = np.array((0., 0.))

    def _init_joysticks(self):
        joystick.init()
        num_sticks = joystick.get_count()
        text = 'ID {0} with {1} axes'
        if not num_sticks:
            self._logger.warn('No joysticks found!')
        else:
            self._logger.info('Found {0} joysticks:'.format(num_sticks))
            for id in xrange(num_sticks):
                stick = joystick.Joystick(id)
                stick.init()
                self._logger.info(text.format(id, stick.get_numaxes()))
        self._has_joystick = all(ax[0] < num_sticks for ax in (self._x_axis_id,
                                                               self._y_axis_id))
        self._x_axis_old = 0.
        self._y_axis_old = 0.

    def _setup_axes(self):
        if self._has_joystick:
            self._stick_x = joystick.Joystick(self._x_axis_id[0])
            self._stick_y = joystick.Joystick(self._y_axis_id[0])
            self._process = self._process_joystick
        else:
            self._logger.warn('Invalid joystick IDs! Using keyboard input.')
            self._process = self._process_keyboard

    def keyboard_input(self, event):
        if event.key in self._key_values:
            value = copy.copy(self._key_values[event.key])
            if event.type == pygame.KEYUP:
                value = -value
            self.keyboard_value += value

    def joystick_button(self, event):
        if self._process_button_events:
            diff = self._button_values.get((event.joy, event.button), (0., 0.))
            self._button_value += diff

    def __call__(self):
        self._process()

    def _process_joystick(self):
        if self._map_to_course:
            self._cursor.move_absolute(self._x_value, self._y_value)
        else:
            self._cursor.move_diff(self._x_axis(), self._y_axis())

    def _process_keyboard(self):
        self._cursor.move_diff(*self.keyboard_value)

    @property
    def _x_value(self):
        return self._stick_x.get_axis(self._x_axis_id[1])

    @property
    def _y_value(self):
        return self._stick_y.get_axis(self._y_axis_id[1])

    def _x_axis_relative(self):
        """ The notions 'relative' and 'absolute' seem to be mismatched
        here, but they are not, because the cursor is moved by the
        difference of the x/y_axis values, see _process_joystick and
        move_diff.
        """
        diff = self._x_value
        if abs(diff) < self._movement_threshold:
            diff = 0
        return diff

    def _y_axis_relative(self):
        diff = self._y_value
        if abs(diff) < self._movement_threshold:
            diff = 0
        return diff

    def _x_axis_absolute(self):
        new = self._x_value
        diff = new - self._x_axis_old
        self._x_axis_old = new
        return diff

    def _y_axis_absolute(self):
        new = self._y_value
        diff = new - self._y_axis_old
        self._y_axis_old = new
        return diff

    def _button(self, stick, button):
        stick_attr = lambda s: getattr(self, s.format(stick))
        id = stick_attr('_{0}_axis_id')[button]
        return stick_attr('_stick_{0}').get_button(id)

    def _x_axis_button_hold(self):
        return self._button('x', 1) - self._button('x', 2)

    def _y_axis_button_hold(self):
        return self._button('y', 1) - self._button('y', 2)

    def _x_axis_button_push(self):
        old = self._button_value[0]
        self._button_value[0] = 0.
        return old

    def _y_axis_button_push(self):
        old = self._button_value[1]
        self._button_value[1] = 0.
        return old

class Synthie(object):
    def __init__(self, dir, start, finish, critical, fatal, duration, recreate,
                 active):
        self._dir = dir
        self._active = active
        events = ['start', 'finish', 'critical', 'fatal']
        sound_file = lambda n: os.path.join(self._dir, '{0}.wav'.format(n))
        if recreate:
            from scikits import audiolab
            shutil.rmtree(self._dir, ignore_errors=True)
            os.mkdir(self._dir)
            for event in events:
                value = eval(event)
                sound = np.sin(np.arange(0, duration / value, 1. / value))
                audiolab.wavwrite(sound, sound_file(event))
        sound = lambda event: pygame.mixer.Sound(sound_file(event))
        self._sounds = dict([(event, sound(event)) for event in events])

    def play(self, event):
        if self._active:
            self._sounds[event].play()

class ZweiHand(VisionEggFeedback):
    """ This feedback recreates the experience of the 2HAND test, which
    is part of the "Wiener Testsystem" suite.
    The task is to navigate a target through a course as fast as
    possible. The constraints of the trial are:
    - Specimen is not allowed to deviate too far from the course
    boundaries
    - Game is to be controlled with two joysticks, each one only
    movable along a single axis
    """

    def __init__(self, *a, **kw):
        super(ZweiHand, self).__init__(*a, **kw)
        self._cursor_basic_size = 12
        self._track_number = 0
        self._run = 0

    def init_parameters(self):
        """ Explanation of parameters:
        number_of_runs: maximum number of tracks to be tried
        running_time: maximum experiment time in seconds
        joystick_style: can be one of these values:
            relative: standard method where the stick's disposition
                      determines the cursor speed
            absolute: stick's disposition determines the cursor position
            button_hold: fixed speed movement when button is held
            button_push: small movement each time button is pushed,
                         suitable for rotation knobs
        *_axis_id: First number is the pygame id of the desired
                   joystick, second one said joystick's pygame axis id.
                   If button input mode is active, second and third
                   number are button ids for left/right or up/down
        interval: How long to wait between position updates
        cursor_speed: Movement speed in pixel/second
        reset_pause: Time to wait when cursor is being reset after
                     entering the critical area
        *_value: Sum of color values for different area types
        *_sound: Tone pitch of sounds for different events
        recreate_sounds: Whether to rewrite the sounds to disk
                         (when changing frequencies)
        keep_aspect_ratio: If the screen sizes differs from the track
                           size, whether to keep proportions
        """
        from os.path import join, dirname
        self.track_sequence = []
        self.number_of_runs = 1000
        self.running_time = 1800
        self.track_dir = join(dirname(__file__), 'tracks')
        self.geometry = [0, 0, 800, 600]
        self.cursor_color_inactive = 'orange'
        self.cursor_color = 'green'
        self.cursor_color_critical = 'red'
        self.cursor_radius_scale = 1
        self.joystick_style = 'absolute'
        self.map_joystick_to_course = True
        self.y_axis_id = (0, 0, 3)
        self.interval = .03
        self.cursor_speed = 600.
        self.movement_threshold = 0
        self.reset_pause = 1
        self.inter_trial_pause = 1
        self.start_value = 1
        self.finish_value = 2
        self.initial_position_value = 763
        self.critical_value = 764
        self.fatal_value = 765
        self.track_value = 600
        self.sound = False
        self.sound_dir = join(dirname(__file__), 'snd')
        self.start_sound = .33
        self.finish_sound = .44
        self.critical_sound = .55
        self.fatal_sound = .66
        self.sound_duration = 1000
        self.recreate_sounds = False
        self.keep_aspect_ratio = True
        self.fullscreen = False
        self.fullscreen_resolution = [1920, 1200]
        self.wait_style_fixed = False
        date = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        self.log_dir = join(dirname(__file__), 'logs')
        self.log_file_name = 'trial_{0}.log'.format(date)

    def init(self):
        pygame.init()

    def update_parameters(self):
        self._setup_sounds()
        self._setup_dirs()

    def _setup_sounds(self):
        self._synthie = Synthie(self.sound_dir, self.start_sound,
                                self.finish_sound, self.critical_sound,
                                self.fatal_sound, self.sound_duration,
                                self.recreate_sounds, self.sound)

    def _setup_dirs(self):
        if not os.path.isdir(self.log_dir):
            os.mkdir(self.log_dir)
        if not os.path.isdir(self.sound_dir):
            os.mkdir(self.sound_dir)
        self._log_file = open(os.path.join(self.log_dir, self.log_file_name),
                              'w')
        if not self.track_sequence:
            track_count = len(os.listdir(self.track_dir))
            self.track_sequence = range(1, track_count+1)

    def run(self):
        self._start_time = datetime.now()
        self._setup_cursor()
        self._setup_position_tracker()
        self._setup_input_handler()
        self._track_texture = self.add_image_stimulus(anchor='lowerleft')
        self.add_stimuli(self._cursor)
        numbers = enumerate(cycle(self.track_sequence), 1)
        while self._flag and self._conditions_fulfilled:
            self._run, self._track_number = next(numbers)
            self._trial()
        self._log_file.close()

    def _setup_cursor(self):
        speed = self.cursor_speed * self.interval
        radius = self.cursor_radius_scale * self._cursor_basic_size
        self._cursor = Cursor(speed, color_inactive=self.cursor_color_inactive,
                              color_valid=self.cursor_color,
                              color_critical=self.cursor_color_critical,
                              radius=radius,
                              map_to_course=self.map_joystick_to_course)

    def _setup_position_tracker(self):
        self._position_tracker = PositionTracker(self._view, self._cursor,
                                                 self.critical_value,
                                                 self.fatal_value,
                                                 self.track_value,
                                                 self._synthie,
                                                 self.reset_pause)

    def _setup_input_handler(self):
        self._input_handler = InputHandler(self._cursor, self.x_axis_id,
                                           self.y_axis_id, self.joystick_style,
                                           self.movement_threshold,
                                           self.map_joystick_to_course)
        handlers = [(pygame.JOYBUTTONDOWN, self._input_handler.joystick_button)]
        self._view.add_event_handlers(handlers)

    def _trial(self):
        self._setup_grid()
        self._setup_stimuli()
        self.stimulus_sequence(self._main(), self.interval).run()
        self._write_log()
        time.sleep(self.inter_trial_pause)

    def _setup_grid(self):
        fname = 'track{0}.png'.format(self._track_number)
        self._track_file = os.path.join(self.track_dir, fname)
        data = scipy.misc.imread(self._track_file)
        self._grid = Grid(data, self.start_value, self.finish_value,
                          self.initial_position_value, self.screen_size[::-1],
                          self.keep_aspect_ratio)
        self._cursor.set_grid(self._grid)
        self._position_tracker.set_grid(self._grid)

    def _setup_stimuli(self):
        size = self._grid.size[::-1]
        self._track_texture.set_file(self._track_file)
        self._track_texture.set(size=size)

    def _main(self):
        while self._flag and not self._position_tracker.finished:
            self._input_handler()
            if self._cursor.has_moved:
                self._position_tracker.moving()
            else:
                self._position_tracker.idle()
            yield

    def _write_log(self):
        def format(label, overall, durations):
            line =  '{0} time: {1} ({2})\n'
            return line.format(label, overall, ' + '.join(map(str, durations)))

        title = 'Run {0} on track {1}:\n'.format(self._run, self._track_number)
        g_o = self._position_tracker.global_timer.overall
        g_d = self._position_tracker.global_timer.durations
        c_o = self._position_tracker.critical_timer.overall
        c_d = self._position_tracker.critical_timer.durations
        i_o = self._position_tracker.idle_timer.overall
        i_d = self._position_tracker.idle_timer.durations
        self._log_file.write(title)
        self._log_file.writelines(map(format, ('total', 'critical', 'idle'),
                                      (g_o, c_o, i_o), (g_d, c_d, i_d)))
        self._log_file.write('\n')

    def keyboard_input(self, event):
        super(ZweiHand, self).keyboard_input(event)
        self._input_handler.keyboard_input(event)

    def keyboard_input_up(self, event):
        super(ZweiHand, self).keyboard_input_up(event)
        self._input_handler.keyboard_input(event)

    @property
    def _elapsed_time(self):
        return total_seconds(datetime.now() - self._start_time)

    @property
    def _conditions_fulfilled(self):
        return (self._run < self.number_of_runs and
                self._elapsed_time < self.running_time)

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    e = ZweiHand()
    e.on_init()
    e.on_play()
