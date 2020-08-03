# HexoViz.py -
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

import Utils
import sys
from math import sqrt
from GraphicComponents.Hexagon import Hexagon
from GraphicComponents.Arrow import Arrow
from GraphicComponents.ControlSignalBar import ControlSignalBar
from GraphicComponents.TextBoard import TextBoard

# PANDA imports
from direct.showbase.DirectObject import DirectObject
import direct.gui.OnscreenText as ost
from direct.task import Task
from direct.gui.OnscreenText import OnscreenText
from pandac.PandaModules import TextNode, PointLight, VBase4, AmbientLight
#from direct.VBase4_extensions import VBase4_extensions as VBase4
#from include import TextNode, PointLight, AmbientLight


class HexoViz(DirectObject):
    
    def __init__(self, hexo_controler, PARAMS):
        
        #if not sys.modules.has_key('direct.directbase.DirectStart'):
        from direct.directbase import DirectStart
#        if not hasattr(base, 'win'):
#            print "no window available - trying to reload DirectStart"
#            self._reinit_base()
#        mod_list = [sys.modules[key] for key in sys.modules.keys() if key.startswith("direct.") and not(sys.modules[key]==None)]
#        print mod_list
        self.set_hexo_controler(hexo_controler)
        self.params = PARAMS
        # create the visualization components
        self._create_control_signal_viz()
        self._create_spelled_text_viz()
        self.hexagons = [] # list of hexagon visual objects
        self._create_hexagons_viz()
        self._create_arrow_viz()
        self.hex_direction = 'front' # initially the hexagons are showing their front side
        # create the manual control elements
        self.manual_control_signal_increment = self.params["manual_control_signal_increment"]
        self._create_manual_control_elements()
        self._set_light_sources()
        base.disableMouse()
        base.camera.setPos(0,-5,0)
        
    def _reinit_base(self):
        
        for key in sys.modules.keys():
            if key.startswith("direct.") and not(sys.modules[key]==None) and not key.endswith('Messenger'):
                sys.modules.pop(key)
        import direct.showbase
        reload(sys.modules['direct.showbase.Messenger'])
        from direct.directbase import DirectStart        

        
    def tick(self, dt):
        # render the scene once       
        taskMgr.step()
        
    def pause_tick(self):
        """ The viz will show some pause indicator or pause animation. """ 
        pass
        
    def play_tick(self):
        self.viz_control_signal(self.hexo_controler.get_control_signal())
        self.viz_arrow()
        self.highlight_hexagon(self.hexo_controler.get_selected_hexagon_index())
    
    def shut_down(self):
        # try to remove all Panda3D stuff from working memory so that it has to be completely reloaded
        base.exitfunc()
        for key in sys.modules.keys():
            if key.startswith("direct.") and not(sys.modules[key]==None):# and not key.endswith('Messenger'):
                del sys.modules[key]
                #sys.modules.pop(key)
        pass
        
    def viz_control_signal(self, control_signal):
        """ Visualize the current output value of the classifier. """
        #print "HexoViz viz_control_signal"
        #self.control_signal_viz.setText("Control Signal = "+str(control_signal))
        # here I assume that the control signal is between -1 and 1
        v = self.normalize_control_signal(control_signal)        
        self.control_signal_viz.set_bar_height(v)         
        
    def viz_arrow(self):
        """ Visualizes the arrow. """
        phi = self.hexo_controler.get_phi_degrees()
        length = self.hexo_controler.get_arrow_length()
        self.arrow.set_angle_x_z_plane(phi)
        self.arrow.set_length(length)
        if self.hexo_controler.is_arrow_locked():
            r,g,b = self.params["arrow_locked_color"]
        else:
            r,g,b = self.params["arrow_color"]
        self.arrow.set_color(r, g, b)
        #self.arrow_text.setText("Arrow: angle = "+str(self.hexo_controler.get_phi_degrees())+", length = "+str(self.hexo_controler.get_arrow_length()))
    
    def set_arrow_color(self, r, g, b):
        self.arrow.set_color(r, g, b)
        self.params["arrow_color"] = (r,g,b)
        
    def set_arrow_locked_color(self, r, g, b):
        self.params["arrow_locked_color"] = (r,g,b)
    
    def set_arrow_rotation_threshold(self, t):
        t = self.normalize_control_signal(t)
        self.control_signal_viz.set_threshold_1(t)
        
    def set_arrow_growth_threshold(self, t):
        t = self.normalize_control_signal(t)
        self.control_signal_viz.set_threshold_2(t)
        
    def highlight_hexagon(self, index=0):
        """ Highlights the hexagon with the given index. """
        if not index==None:
            for i in range(6):
                if i==index:
                    r,g,b = self.params["hexagon_highlight_color"]
                    #self.hexagons[int(index)].setBg((r,g,b,1))
                    self.hexagons[int(index)].set_color(r,g,b)
                else:
                    r,g,b = self.params["hexagon_default_color"]
                    #self.hexagons[i].setBg((r,g,b,1))
                    self.hexagons[i].set_color(r,g,b)
            
    def set_symbol_lists(self, symbol_lists):
        """ The given list of symbol lists will be displayed by the hexagons. """
        for i, sub_list in enumerate(symbol_lists):
            self.hexagons[i].set_front_side_symbols(sub_list)
    
    def set_big_symbols(self, symbols, selected_hex):
        """ Sets the given symbols as big symbols on the hexagons, on symbol for each hexagon. """
        self.remove_big_symbols()
        pos_idx= self.hexagons[selected_hex].idx_to_pos
        for i, symbol in enumerate(symbols):
            self.hexagons[pos_idx[i]].set_back_side_symbol(symbol)
            
    def get_selected_symbol(self, hex_idx, pos_idx):
        return self.hexagons[hex_idx].get_symbol(pos_idx)
    
    def remove_big_symbols(self):
        """ Sets all back side symbols to '', thereby erasing the ones that were there previously. """
        for hex in self.hexagons:
            hex.set_back_side_symbol('')
        
    def show_spelled_text(self, text):
        """ Displays the given text in the spelled-text-field. """
        self.spelled_text_viz.set_text(text)
    
    def set_hexo_controler(self, hexo_controler):
        """ Sets self.hexo_controler to the given hexo_controler. """
        self.hexo_controler = hexo_controler
    
    def set_hexagon_color(self, r,g,b):
        self.params['hexagon_default_color'] = (r,g,b)
        for hex in self.hexagons:
            hex.set_color(r,g,b)
    
    def set_hexagon_highlight_color(self, r,g,b):
        self.params['hexagon_highlight_color'] = (r,g,b)
        
    def set_hexagon_text_color(self, r,g,b,alpha=1):
        for hex in self.hexagons:
            hex.set_text_color(r,g,b,alpha)
        
    def set_textboard_background_color(self, r,g,b):
        self.spelled_text_viz.set_background_color(r, g, b, 1)
        
    def set_textboard_frame_color(self, r,g,b):
        self.spelled_text_viz.set_frame_color(r, g, b, alpha=1)
    
    def set_textboard_text_color(self, r,g,b):
        self.spelled_text_viz.set_text_color(r, g, b, alpha=1)
        
    def set_background_color(self, r,g,b):
        base.setBackgroundColor((r,g,b))
        
    def set_control_signal_bar_frame_color(self, r,g,b):
        self.control_signal_viz.set_frame_color(r, g, b, alpha=1)
    
    def set_control_signal_bar_color(self, r,g,b):
        self.control_signal_viz.set_bar_color(r, g, b, alpha=1)
    
    def start_state_change_animation(self, rot_arrow=False, phi_start=0, phi_end=0):
        if self.hex_direction == 'front':
            start_angle = 0
            stop_angle = 180
            self.hex_direction = 'back'
        elif self.hex_direction == 'back':
            start_angle = 180
            stop_angle = 360
            self.hex_direction = 'front'
        else:
            raise Exception("HexoViz::start_state_change_animation - self.hex_direction has invalid value: "+str(self.hex_direction))
        rot_time = self.params["state_change_animation_duration"]
        hex_list = self.hexagons
        taskMgr.add(HexoViz.rotate_hexagons_task, 'rotate_hexagons', 
                    extraArgs=[hex_list, rot_time, start_angle, stop_angle], appendTask=True)
        if rot_arrow:
            taskMgr.add(HexoViz.rotate_arrow_task, 'rotate_arrow', 
                    extraArgs=[self.arrow, rot_time, phi_start, phi_end], appendTask=True)
        

    def normalize_control_signal(self, v):
        """ Assuming a given control signal range (e.g. -1 to 1), the given value is mapped to the range 0 to 1. """        
        return (v+1)/2.0
            
    def _create_control_signal_viz(self):
        """ Creates and initializes the visualization of the classifier output. """
        self.control_signal_viz = ControlSignalBar(1, self.params["control_signal_bar_width"],
                                                   (0.3, 0.7), # dummy thresholds, will be replaced by real ones via the set_tresholds method 
                                                   self.params["control_signal_bar_padding"])
        self.control_signal_viz.set_scale(self.params["control_signal_bar_scaling"])
        (x,z) = self.params["control_signal_bar_position"]
        self.control_signal_viz.set_pos(x, 0, z)
        (r,g,b) = self.params["control_signal_bar_color"]
        self.control_signal_viz.set_bar_color(r, g, b)
        (r,g,b) = self.params["control_signal_bar_frame_color"]
        self.control_signal_viz.set_frame_color(r, g, b)
        #self.control_signal_viz = OnscreenText(text="Control Signal = 0", pos=(0.9, -0.9), mayChange=True, scale=0.05)
        
    def _create_spelled_text_viz(self):
        """ Creates and initializes the visualization of the text field in which the already spelled text is shown. """
        #self.spelled_text_viz = OnscreenText(text="Spelled Text", pos=(0,0.85), scale=0.1, mayChange=True, fg=(0,0,1,1), frame=(0,1,0,1))
        self.spelled_text_viz = TextBoard()
        self.spelled_text_viz.set_text_color(0, 0, 0)
        self.spelled_text_viz.set_scale(0.1)
        self.spelled_text_viz.set_center_pos_xz(0, 1)
        
    
    def _create_hexagons_viz(self):
        """ Creates and positions the Hexagons. """
        # position them around the center point with radius r
        r = self.params["hex_distance_to_middle"] # distance from the center to the middle point of each hex
        hex_inner_r = r / sqrt(3) - self.params["gap_width_between_hexagons"]
        # create the hex objects first
        for i in range(6):
            hex = Hexagon(radius=hex_inner_r, width=self.params["hex_depth"], color=self.params["hexagon_default_color"], hex_index=i)
            #hex.set_front_side_symbols_scale(0.5)
            self.hexagons.append(hex)
            #self.hexagons.append(OnscreenText(text="Hex Nr "+str(i+1), pos=(0,0), scale=0.05, mayChange=True, align=TextNode.ACenter, style=ost.BlackOnWhite))
        # the first set of coordinates is (0,r) + center point
        # the next coordinates are rotated clockwise 60 degrees
        x, y = 0,r
        phi = 60 # angle in degrees
        for i in range(6):
            #self.hexagons[i].setPos(x+self.params["center_position"][0], y+self.params["center_position"][1])
            self.hexagons[i].set_pos(x+self.params["center_position"][0], 0, y+self.params["center_position"][1])
            x,y = Utils.rotate_phi_degrees_clockwise(phi, (x,y))
    
    def _create_arrow_viz(self):
        """ Creates the rotating arrow. """
        self.arrow = Arrow()
        self.arrow.get_node_path().setX(self.params['center_position'][0])
        self.arrow.get_node_path().setZ(self.params['center_position'][1])
        self.arrow.set_scale(self.params['arrow_scale'])
        r,g,b = self.params["arrow_color"]
        self.arrow.set_color(r, g, b)
#        self.arrow_text = OnscreenText(text="Arrow: angle = ?, length = ?", pos=self.params["center_position"], 
#                                       scale=0.05, mayChange=True, align=TextNode.ACenter)
    
    def _create_manual_control_elements(self):
        """ Creates keyboard listeners, such that manual interaction is possible. """
        self.accept('j', self._increase_control_signal)
        self.accept('f', self._decrease_control_signal)
        self.accept('space', self._toggle_control_signal_bar_visible)
        
    def _increase_control_signal(self):
        """ Callback that signals the speller backend to increase the control signal (classifier output). """
        current_value = self.hexo_controler.get_control_signal()
        increased_value = current_value + self.manual_control_signal_increment
        self.hexo_controler.set_control_signal(increased_value)
#        print "increased value = " + str(self.hexo_controler.get_control_signal())
        
    def _decrease_control_signal(self):
        """ Callback that signals the speller backend to decrease the control signal (classifier output). """
        current_value = self.hexo_controler.get_control_signal()
        decreased_value = current_value - self.manual_control_signal_increment
        self.hexo_controler.set_control_signal(decreased_value)
        #print "decreased value = " + str(self.hexo_controler.get_control_signal())
    
    def _toggle_control_signal_bar_visible(self):
        if self.control_signal_viz.get_node_path().isHidden():
            self.control_signal_viz.get_node_path().show()
        else:
            self.control_signal_viz.get_node_path().hide()
                    
    def _set_light_sources(self):
        light_positions = [(1,-1,1),(-1,-5,1)]
        intensity = 0.8
        for l_pos in light_positions:
            plight = PointLight('plight')
            plight.setColor(VBase4(intensity, intensity, intensity, 1))
            plnp = render.attachNewNode(plight)
            plnp.setPos(l_pos[0], l_pos[1], l_pos[2])
            render.setLight(plnp)
        light = AmbientLight('')
        light.setColor(VBase4(0.4,0.4,0.4,1))
        light_np = render.attachNewNode(light)
        light_np.setPos(0,0,0)
        render.setLight(light_np)
        
        
        pass
    
    
    
    @staticmethod
    def rotate_hexagons_task(hexagon_list, rotation_time, start_angle, stop_angle, task):
        current_angle = hexagon_list[0].get_node_path().getH() # here I assume all hexagons have the same angle!!!
        if current_angle >= stop_angle or current_angle < start_angle:
            return Task.done
        phi = start_angle + 180 * task.time/rotation_time
        phi = phi % 360
        for hexagon in hexagon_list:
            hexagon.get_node_path().setH(phi)
        return Task.cont
        
    @staticmethod
    def rotate_arrow_task(arrow, rotation_time, start_phi, stop_phi, task):
#        current_angle = arrow.get_node_path().getR()
        if task.time >= rotation_time:
            return Task.done
        angle_diff = (stop_phi - start_phi) % 360
        phi = start_phi + angle_diff * task.time/rotation_time
        arrow.get_node_path().setR(phi)
        return Task.cont
    
