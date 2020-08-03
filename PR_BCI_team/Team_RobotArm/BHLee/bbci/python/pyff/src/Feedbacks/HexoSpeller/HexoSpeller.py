# HexoSpeller.py -
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


import time, os
from FeedbackBase.MainloopFeedback import MainloopFeedback
from HexoModel import HexoModel
from HexoViz import HexoViz
from LanguageModel import LanguageModel
import GraphicComponents.ColorSchemes as ColorSchemes

PARAMS = {
          "initial_arraw_angle" : 0, # initial orientation of the arrow, in degrees
          "initial_arrow_length": 0.35, # the starting length of the arrow. The maximum length is 1, so this value must be smaller than 1!
          "arrow_rotation_time": 5, # time it takes for a full 360 degree rotation of the arrow, given in seconds
          "arrow_growth_time": 2, # time it takes to grow from length zero to full length (1), given in seconds
          "arrow_shrinkage_time": 1.5, # time it takes to shrink from full length (1) to length zero, given in seconds
          "control_signal_arrow_rotation_threshold": -0.3, # control signal interval in which the arrow will be rotating will be between -1 and this threshold
          "control_signal_arrow_growth_threshold": 0.3, # interval in which the arrow length will be increasing will be this threshold and 1
          "hex_pre_select_bias": -20,
          "arrow_locked_duration": 0.7, # length of time period (in seconds) after hexagon selection during which the arrow is locked, i.e. cannot be moved.
                                    # This "locking period" is useful to prevent interaction during hex rotation animation
          # language model params
          "language_model_file" : "english_all.pckl", #None,# "lm1to8.pckl", # one of 'lm1to8.pckl' or 'german.pckl', must lie in the folder LanguageModels
          "lm_head_factors" : [1.0, 0.9, 0.8, 0.7, 0.6, 0.5],
          "lm_letter_factor" : 0.01,
          "lm_n_pred" : 2,
          }

VIZ_PARAMS = {
              "manual_control_signal_increment" : 0.1, # how the control signal gets changed per manual control (keyboard) interaction
              "center_position": (0,-0.2), # position of the center point around which the arrow rotates and also the center reference point for the hexagons
              "hex_distance_to_middle": 0.6, # distance of the center of each hexagon to the center position
              "hex_depth": 0.1, 
              "gap_width_between_hexagons":0.01, # a little gap between adjacent hexagons
              "arrow_scale": 0.28,
              "state_change_animation_duration": 0.5, # length of the _state change animation in seconds
              "control_signal_bar_position": (1.3, -1.05), # position of the lower end of the bar, not relative to center point (!)
              "control_signal_bar_width": 0.15, # the width of the bar relative to a length of 1
              "control_signal_bar_scaling": 1.5, # controls the size of the control signal bar
              "control_signal_bar_padding": 0.02, # distance between frame and bar
              
              "color_scheme": "TREE", # one of "", "ICE", "DESERT", "TREE"
              # if color_scheme is not None, then this values will be overwritten by the color scheme
              "hexagon_default_color": (1,1,1), 
              "hexagon_highlight_color": (1,0,0),
              "hexagon_text_color": (1,1,1),
              "arrow_color": (0,1,0),
              "arrow_locked_color": (0.5,0.5,0.5),
              "control_signal_bar_color": (1 ,1 ,0),
              "control_signal_bar_frame_color": (0,0,0),
              "textboard_background_color": (1,1,0),
              "textboard_frame_color": (0,0,0),
              "textboard_text_color": (0,0,0),
              "background_color": (0.4,0.4,0.4), 
              }

# Markers written to the parallel port
class Marker():
    # hex state changes
    arrow_free = 31
    arrow_locked = 39
    hex_selected_level_one = 32
    state_change_animation_level_one_to_two_start = 34
    state_change_animation_level_one_to_two_end = 36
    hex_selected_level_two = 40
    state_change_animation_level_two_to_one_start = 41
    state_change_animation_level_two_to_one_end = 42
    selected_hex_level_one = [11,12,13,14,15,16] # hex 1-6 was selected at level 1
    selected_hex_level_two = [21,22,23,24,25,26] # hex 1-6 was selected at level 2
    selected_letter = range(61,61+30)   # code for the selected symbols, the list will be indexed according to the 
                                        # symbol list that comes with the language model
    feedback_init = 200
    status_change_to_play = 210
    status_change_to_pause = 211
    status_change_to_stop = 212
           

class HexoSpeller(MainloopFeedback):
    
    states = {
              "level_one": 1, # the hexagons contain groups of symbols, the group has to be picked first
              "level_two": 2, # the hexagons contain individual symbols
              }
    
    
    def init(self):
        self.send_parallel(Marker.feedback_init)
        self.logger.debug("HexoSpeller::init")
        self._last_tick_time = time.clock()
        self._state = self.states["level_one"]
        language_model_folder_path = self._create_language_model_folder_path()
        #print language_model_folder_path
        print '!!!!!!!!SVN Version!!!!!'
        self.load_language_model(os.path.join(language_model_folder_path, PARAMS["language_model_file"]))
        self.spelled_text = []
        self._sub_list_probs = [] # probability values for each symbol sublist
        self._selected_symbol_idx = 0
        self._selected_symbol_sublist_idx = 0
        self._arrow_locked = False
        self._arrow_locked_time = None
        self.lock_arrow()
        self._control_signal = 0
        self._viz = None
        self._model = None
        
        
        
    def pre_mainloop(self):
        #print "HexoSpeller::pre_main_loop"
        self._model = HexoModel(PARAMS)
        self._model.add_arrow_length_observer(self)
        self._viz = HexoViz(self, VIZ_PARAMS)
        self._viz.hexo_controller = self
        self._viz.set_symbol_lists(self.symbol_list)
        if hasattr(ColorSchemes, VIZ_PARAMS["color_scheme"]):
            scheme_dictionary = getattr(ColorSchemes, VIZ_PARAMS["color_scheme"])
            VIZ_PARAMS.update(scheme_dictionary)
        # set some public variable that can be modified from the feedback controller GUI
        # set all variables for which there is a setter with the corresponding name
        for dict in [PARAMS, VIZ_PARAMS]:
            for key in dict.keys():
                if hasattr(self, 'set_'+key):
                    set_method = getattr(self, 'set_'+key)
                    set_method(dict[key])
    
    def post_mainloop(self):
        """ Tries to shut down the visualization. """
        self._viz.shut_down()
        
        
    def tick(self):
        """ Is called in each iteration of the main loop. This method determines how much time has passed
        between the current and the previous tick, and then delegates that information to the _model and the view
        via their tick(dt) methods. """
        if self._viz==None or self._model==None:
            return
        # determine how much time (in seconds) has passed between this and the previous tick
        current_time = time.clock()
        dt = current_time - self._last_tick_time
        self._last_tick_time = current_time
        # delegate the tick to the back end and the front end
        self._viz.tick(dt)
        self._model.tick(dt)
        self._model.set_control_signal(self.get_control_signal())
        # if the arrow is locked and the locking period is over, unlock it
        if self.is_arrow_locked():
            if current_time - self._arrow_locked_time > self.arrow_locked_duration:
                self.unlock_arrow()
        
    def play_tick(self):
        if not self.is_arrow_locked():
            self._model.play_tick()
        self._viz.play_tick()
    
    def pause_tick(self):
        self._viz.pause_tick()
        self._model.pause_tick()
        
    def on_control_event(self, data):
        self.logger.debug('on_control_event')               
        self.set_control_signal(data["cl_output"])
        
    def on_interaction_event(self, data):
        self.logger.debug("on_interaction_event") 
        if type(data)==type({}):   
            # try to set the modified attributes
            for name in data.keys():
                # if we have the attribute and the respective setter
                if hasattr(self, name) and hasattr(self, "set_"+name):
                    set_method = getattr(self, "set_"+name)
                    new_value = data[name]
                    set_method(new_value)
    
    def on_play(self):
        self.send_parallel(Marker.status_change_to_play)
        MainloopFeedback.on_play(self)
    
    def on_pause(self):
        if self._MainloopFeedback__running and self._MainloopFeedback__paused:
            self.send_parallel(Marker.status_change_to_play)
        if self._MainloopFeedback__running and not self._MainloopFeedback__paused:
            self.send_parallel(Marker.status_change_to_pause)
        MainloopFeedback.on_pause(self)
        
    def on_stop(self):
        self.send_parallel(Marker.status_change_to_stop)
        MainloopFeedback.on_stop(self)
                    
        
    def get_selected_hexagon_index(self):
        """ Returns the hexagon that the arrow is currently pointing at. """
        return self._model.get_selected_hexagon_index()
    
    def get_arrow_length(self):
        return self._model.get_arrow_length()
    
    def get_phi_degrees(self):
        return self._model.get_phi_degrees()
    
    def arrow_at_max_length(self):
        """ To be called by the _model when the arrow has reached maximum length. """
        self.logger.debug("HexoFeedback::arrow_at_max_length")
        if self._state == self.states["level_one"]:
            self.send_parallel(Marker.hex_selected_level_one)
            # signal the GUI to change the content of the hexagons to single symbols
            selected_idx = self.get_selected_hexagon_index()
            self.send_parallel(Marker.selected_hex_level_one[selected_idx])
            self._viz.set_big_symbols(self.symbol_list[selected_idx], selected_idx)
            self._selected_symbol_sublist_idx = selected_idx
            # return the arrow to start length, but don't change the angle
            self.reset_arrow_model(reset_phi=False)
            # change to _state 'second selection'
            self._state = self.states["level_two"]
            self.lock_arrow()
            self._viz.start_state_change_animation()
        elif self._state == self.states["level_two"]:
            self.send_parallel(Marker.hex_selected_level_two)
            # get and store the selected symbol
            self.get_selected_symbol()
            self.update_symbol_list()
            # update the spelled word in the GUI
            self._viz.show_spelled_text(self.text_list_to_string(self.spelled_text))
            # signal the GUI to change the content of the hexagons back to multiple symbols
            self._viz.set_symbol_lists(self.symbol_list)
            # return the arrow to start angle and start length
            current_phi = self._model.get_phi_degrees()
            new_phi = (self.get_most_probable_hexagon_index()*60 + self.hex_pre_select_bias) % 360
            self.reset_arrow_model(reset_phi=True, phi=new_phi)
            self._state = self.states["level_one"]
            self.lock_arrow()
            self._viz.start_state_change_animation(rot_arrow=True, phi_start=current_phi, phi_end=new_phi)
        
    def reset_arrow_model(self, reset_phi=False, phi=0, control_signal=0):
        """ Resets the arrow length to initial length and the arrow angle and control signal value
         according to the given values. """
        self._model.reset_arrow_length()
        self._model.set_control_signal(control_signal)
        if reset_phi:
            self._model.reset_phi(phi)
        
    def get_selected_symbol(self):
        idx = self.get_selected_hexagon_index()
        self._selected_symbol_idx = idx
        self.send_parallel(Marker.selected_hex_level_two[idx])
        symbol = self._viz.get_selected_symbol(self._selected_symbol_sublist_idx, self._selected_symbol_idx)
        if symbol == self._language_model.delete_symbol:
            # if the delete symbol was selected and there is something to delete, pop the last character from the list
            if len(self.spelled_text) > 0:
                self.spelled_text.pop() 
        elif not symbol == None:
            # if the symbol is not None, attach it to the spelled Text
            self.spelled_text.append(symbol)
            # send a marker 
            idx = self._language_model.get_symbol_index(symbol)
            if not idx == None:
                self.send_parallel(Marker.selected_letter[idx])
            
    def get_most_probable_hexagon_index(self):
        """ Returns the index of the hexagon that contains the most probable next letter, based on what is already written. """
        return self._language_model.get_most_probable_symbol_sublist_index()
    
    def update_symbol_list(self):
        """ Update the order of symbols in the symbol list based on the spelled text. """
        spelled_text = self.text_list_to_string(self.spelled_text)
        self.symbol_list = self._language_model.update_symbol_list_sorting(spelled_text)

    def _create_language_model_folder_path(self):
        """ Creates a path that points to the folder that contains the language model file. 
        I assume that the lm files lie in a folder called "LanguageModels" which itself lies
        in the same folder as the HexoSpeller.py file, whose path is given by the __file__
        variable. """
        file_path = __file__ # file_path is now something like foo/bar/Feedbacks/HexoSpeller/HexoSpeller.py
        # remove the actual file name from the path by first reversing the string, then partitioning at the
        # last occourence of the path separator end reversing the tail of the partitioning 
        reversed_file_path = file_path[::-1]
        (_file_name, _sep, hexospeller_dir) = reversed_file_path.partition(os.path.sep)
        hexospeller_dir = hexospeller_dir[::-1] # reverse it, now in correct order
        # now complete the path to point to the language model directory
        lm_path = os.path.join(hexospeller_dir,"LanguageModels")
        return lm_path
       
    def load_language_model(self, file_name):
        """ Get the language _model, preferably from file. The path should be specified in params... """
        self._language_model = LanguageModel(file_name)
        self.symbol_list = self._language_model.get_symbol_list()
        
    def text_list_to_string(self, text_list):
        text = ''
        for c in text_list:
            text = text + c
        return text
    
    def set_control_signal(self, value):
        self._control_signal = value
    
    def get_control_signal(self):
        return self._control_signal
    
    def lock_arrow(self):
        self._arrow_locked = True
        self._arrow_locked_time = time.clock()
    
    def unlock_arrow(self):
        self._arrow_locked = False
    
    def is_arrow_locked(self):
        return self._arrow_locked
    
#================================================================================
#  Setter for the variables that should be 
#  setable from the feedback controller GUI
#================================================================================
    
    def set_hexagon_default_color(self, rgb):
        self.hexagon_default_color = rgb
        if not self._viz == None:
            r,g,b, = rgb
            self._viz.set_hexagon_color(r,g,b)
    
    def set_hexagon_highlight_color(self, rgb):
        self.hexagon_highlight_color = rgb
        if not self._viz == None:
            r,g,b, = rgb
            self._viz.set_hexagon_highlight_color(r, g, b)
        
    def set_hexagon_text_color(self, rgb):
        self.hexagon_text_color = rgb
        if not self._viz == None:
            r,g,b, = rgb
            self._viz.set_hexagon_text_color(r, g, b, alpha=1)
        
    def set_arrow_color(self, rgb):
        self.arrow_color = rgb
        if not self._viz == None:
            r,g,b, = rgb
            self._viz.set_arrow_color(r,g,b)
    
    def set_state_change_animation_duration(self, dur):
        self.state_change_animation_duration = dur
        if not self._viz == None:
            self._viz.params['state_change_animation_duration'] = dur
    
    def set_arrow_growth_time(self, time):
        self.arrow_growth_time = time
        if not self._model == None:
            self._model.params['arrow_growth_time'] = time
        
    def set_arrow_shrinkage_time(self, time):
        self.arrow_shrinkage_time = time
        if not self._model == None:
            self._model.params['arrow_shrinkage_time'] = time
        
    def set_arrow_rotation_time(self, time):
        self.arrow_rotation_time = time
        if not self._model == None:
            self._model.params['arrow_rotation_time'] = time
        
    def set_arrow_locked_duration(self, duration):
        self.arrow_locked_duration = duration
    
    def set_control_signal_arrow_rotation_threshold(self, t):
        self.control_signal_arrow_rotation_threshold = t
        if not self._model == None:
            self._model.params["control_signal_arrow_rotation_threshold"] = t
        if not self._viz == None:
            self._viz.set_arrow_rotation_threshold(t)
    
    def set_control_signal_arrow_growth_threshold(self, t):
        self.control_signal_arrow_growth_threshold = t
        if not self._model == None:
            self._model.params["control_signal_arrow_growth_threshold"] = t
        if not self._viz == None:
            self._viz.set_arrow_growth_threshold(t)
        
    def set_control_signal_bar_frame_color(self, rgb):
        self.control_signal_bar_frame_color = rgb
        if not self._viz == None:
            r,g,b = rgb
            self._viz.set_control_signal_bar_frame_color(r,g,b)

    def set_control_signal_bar_color(self, rgb):
        self.control_signal_bar_color = rgb
        if not self._viz == None:
            r,g,b = rgb
            self._viz.set_control_signal_bar_color(r,g,b)
        
    def set_lm_head_factors(self, head_factors):
        self._language_model.head_factors = head_factors
        self.lm_head_factors = head_factors
    
    def set_lm_letter_factor(self, letter_factor):
        self._language_model.letter_factor = letter_factor
        self.lm_letter_factor = letter_factor
        
    def set_lm_n_pred(self, n_pred):
        self._language_model.n_pred = n_pred
        self.lm_n_pred = n_pred
        
    def set_textboard_background_color(self, rgb):
        self.textboard_background_color = rgb
        if not self._viz == None:
            r,g,b = rgb
            self._viz.set_textboard_background_color(r, g, b)
        
    def set_textboard_frame_color(self, rgb):
        self.textboard_frame_color = rgb
        if not self._viz == None:
            r,g,b = rgb
            self._viz.set_textboard_frame_color(r, g, b)
    
    def set_textboard_text_color(self, rgb):
        self.textboard_text_color = rgb
        if not self._viz == None:
            r,g,b = rgb
            self._viz.set_textboard_text_color(r, g, b)
        
    def set_background_color(self, rgb):
        self.background_color = rgb
        if not self._viz == None:
            r,g,b = rgb
            self._viz.set_background_color(r,g,b)
        
    def set_hex_pre_select_bias(self, v):
        self.hex_pre_select_bias = v
        
    
            
if __name__ == "__main__":
    speller = HexoSpeller()
    speller.on_init()
    speller.on_play()
    
