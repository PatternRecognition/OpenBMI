# HexoModel.py -
# Copyright (C) 2009-2010  Sven Daehne
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

""" 
This class is only the "back-end" of the HexoModel. 

For a given classifier output at a given time, it computes position and 
length of the arrow. It changes between states depending on classifier output.

The Hexagons are indexed 0 to 5, clockwise, the top one is number 0.
"""

import Utils

class HexoModel():
        
    # constants
    # dictionary that contains IDs for each _state
    states = {
              "do_nothing":0,
              "arrow_rotation": 10,
              "arrow_growth": 20,
              "arrow_shrinkage": 30,
           }
    
    
    def __init__(self, params):
        """ Initializes the variables and the visualization object. """
        self.params = params
        self.phi = self.params["initial_arraw_angle"] # angle is expected to be in degrees
        self.arrow_length = self.params["initial_arrow_length"] # must be less than 1
        self.set_elapsed_time(0)
        self.set_control_signal(0)
        self.determine_state()
        self.arrow_length_observer = [] # a list of observers that will be notified when the arrow reaches max length
        
                
    def play_tick(self):
        """
        Depending on the _state the model is in, either rotate the arrow or increase/decrease it size
        """
        self.determine_state()
        if self._state == self.states["arrow_rotation"]:
            self.rotate_arrow()
        if self._state == self.states["arrow_shrinkage"]:
            self.decrease_arrow_length()
        if self._state == self.states["arrow_growth"]:
            self.increase_arrow_length()
        
    def pause_tick(self):
        """ Do nothing. """
        pass
    
    def tick(self, dt):
        """ Sets the elapsed time between this tick and the previous one. """
        self.set_elapsed_time(dt)
        
    def reset_phi(self, phi=None):
        """ Resets the rotating arrow to the given orientation. If no angle is provided, the arrow will face the
        default direction. """
        if not phi==None:
            self.phi = phi
        else:
            self.phi = self.params["initial_arraw_angle"]
    
    def reset_arrow_length(self, length=None):
        """ Resets the rotating arrow to the given length. If no length is specified, the arrow length will set to 
        self.params["initial_arrow_length"], i.e. the default start length. """
        if not length==None:
            self.arrow_length = length
        else:
            self.arrow_length = self.params["initial_arrow_length"]
        
    def rotate_arrow(self, dt=None):
        """ Rotates the arrow with self.arrow_rotation_speed according to how much time has elapsed (given by dt). 
        If dt is not specified, self.dt is used. """
        if dt==None:
            dt = self.dt
        self.phi += (360.0 / self.params["arrow_rotation_time"]) * dt
        self.phi = self.phi % 360.0
        
    def increase_arrow_length(self, dt=None):
        """ Increases the length of the arrow, depending on elapsed time and growth speed. 
        The length will be set to 1 if it is already larger or the increment made it larger. """
        if dt==None:
            dt = self.dt
        self.arrow_length += (1.0 /self.params["arrow_growth_time"]) * dt
        if self.arrow_length >= 1:
            self.arrow_length = 1
            self.signal_max_length_reached()
            
    
    def decrease_arrow_length(self, dt=None):
        """ Decreases the length of the arrow, depending on elapsed time and growth speed. 
        The length will be set to initial_length if it is already smaller that that or the decrement made it smaller. """
        if dt==None:
            dt = self.dt
        self.arrow_length -= (1.0 / self.params["arrow_growth_time"]) * dt
        if self.arrow_length < self.params["initial_arrow_length"]:
            self.arrow_length = self.params["initial_arrow_length"]
        

    def set_control_signal(self, value):
        """ Sets the control signal to the given value. The control signal is used to determine the current _state
        of the speller, i.e. whether it the arrow is rotating or being scaled for example. VALUE must be between -1 and 1. """
        if not self.is_in_range(value, lower_bound=-1, upper_bound=1.00001):
            if value > 1:
                value = 1
            if value < -1:
                value = -1
            #raise Exception("control signal value must be between -1 and 1!")
        self.control_signal = value
    
    def set_elapsed_time(self, dt):
        """ Sets the time that has passed (or passes) between consecutive frames. 
        The elapsed time is used to update the position or scale of the arrow. """
        self.dt = dt
    
    def get_control_signal(self):
        """ Returns the current value of the control signal. """
        return self.control_signal

    def get_selected_hexagon_index(self, phi=None):
        """ Returns the index of the hexagon that the arrow currently points at. """
        if phi==None:
            phi = self.get_phi_degrees()
        if not self.is_in_range(phi, 0, 361):
            raise Exception("Value of phi is not between 0 and 360!")
        for hex_idx, hex_angle in enumerate(range(0,360,60)):
            left = (hex_angle - 30)%360
            right = hex_angle + 30
            if self.is_in_range(phi, left, right):
                return hex_idx
            if left > right: # special case: left is 330 and right is 30, 
                            # therefore the value of phi can't be higher then 
                            # left and smaller then right. Hence it needs special treatment
                if phi > left or phi <=right:
                    return hex_idx
        return None
        
    def get_phi_degrees(self):
        """ Returns the current angle of the arrow in degrees (between 0 and 360). """
        return self.phi
    
    def get_phi_radians(self):
        """ Returns the current angle of the arrow in radians (between 0 and 2*pi). """
        return Utils.degrees_to_radians(self.get_phi_degrees())
        
    def get_arrow_length(self):
        """ Returns the current length of the arrow. The length will be a value between 0 and 1. """
        return self.arrow_length
    
    def is_in_range(self, x, lower_bound, upper_bound):
        """ Checks whether x is in the range defined by the given lower and upper bound.
            Specifically, this function returns   lower_bound <= x and x < upper_bound """
        return lower_bound <= x and x < upper_bound

    def determine_state(self):
        """ Determine the _state of the speller, based on the control signal output. """
        if self.is_in_range(self.control_signal, -1, self.params["control_signal_arrow_rotation_threshold"]):
            self._state = self.states["arrow_rotation"]
        if self.is_in_range(self.control_signal, 
                            self.params["control_signal_arrow_rotation_threshold"],
                            self.params["control_signal_arrow_growth_threshold"]):
            self._state = self.states["arrow_shrinkage"]
        if self.is_in_range(self.control_signal, self.params["control_signal_arrow_growth_threshold"], 1):
            self._state = self.states["arrow_growth"]
    
    def add_arrow_length_observer(self, observer):
        """ Appends the given arrow length observer to the list of observers. """
        self.arrow_length_observer.append(observer)
        
    def signal_max_length_reached(self):
        """ Goes through the list of arrow_length_observer and calls the arrow_at_max_length method. """
        for observer in self.arrow_length_observer:
            observer.arrow_at_max_length()
