# Utils.py -
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
Utility methods for the classes involved in the Hexospeller feedback. 
"""
from math import sin, cos, pi

def rotate_phi_degrees_clockwise(phi, x_y):
    """ Rotates the point given by the tuple x_y for phi degrees clockwise and returns the 
    resulting coordinates in a tuple. """
    phi_rad = degrees_to_radians(phi)
    return rotate_phi_radians_clockwise(phi_rad, x_y)

def rotate_phi_radians_clockwise(phi, x_y):
    """ Rotates the point given by the tuple x_y for phi radians clockwise and returns the 
    resulting coordinates in a tuple. """
    x, y = x_y
    x_new = cos(phi)*x + sin(phi)*y
    y_new = -sin(phi)*x + cos(phi)*y
    return (x_new, y_new)        
    
def rotate_phi_degrees_counter_clockwise(phi, x_y):
    """ Rotates the point given by the tuple x_y for phi degrees counter clockwise and returns the 
    resulting coordinates in a tuple. """
    return rotate_phi_degrees_clockwise(-phi, x_y)
    
def rotate_phi_radians_counter_clockwise(phi, x_y):
    """ Rotates the point given by the tuple x_y for phi radians counter clockwise and returns the 
    resulting coordinates in a tuple. """
    return rotate_phi_radians_clockwise(-phi, x_y)

def degrees_to_radians(phi_degrees):
    """ Converts the angle phi given in degrees to radians. """
    return (phi_degrees/360.0) * 2.0*pi

def radians_to_degrees(phi_radians):
    """ Converts the angle phi given in radians to degrees. """
    return (phi_radians/(2.0*pi)) * 360.0

def copy_list(orig_list):
    """ Creates a shallow copy of the given list and all its sublists. """
    c_list = []
    for elem in orig_list:
        if type(elem) == type([]):
            c_list.append(copy_list(elem))
        else:
            c_list.append(elem)
    return c_list

def array_to_list(array):
    li = []
    for elem in array:
        li.append(elem)
    return li

def max_with_idx(iter):
    max_value = max(iter)
    idx = 0
    for elem in iter:
        if max_value == elem:
            return max_value, idx
        idx += 1
    return max_value, None

def sort_list_according_to_values(list, values):
    sorted_list = []
    while len(list) > 0:
        # find the position of the max value in values
        idx = max_with_idx(values)[1]
        sorted_list.append(list.pop(idx))
        values.pop(idx)
    return sorted_list
    

