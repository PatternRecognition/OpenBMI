from VisionEgg import *
from VisionEgg.WrappedText import WrappedText
import numpy as np
import pylab as p
from Figure_Many import *
from VisionEgg.FlowControl import Presentation, FunctionController
from VisionEgg.Core import *
from VisionEgg.MoreStimuli import *
from VisionEgg.Text import *
#from lib.eyetracker import EyeTracker
from FeedbackBase.MainloopFeedback import MainloopFeedback
#from VisionEgg.Gratings import SinGrating2D

class Hex_skeleton(MainloopFeedback):
    START_EXP, END_EXP = 252, 253
    
    def init(self):
        self.number_of_hexagons = 6
        self.same_screen = True
        self.color_of_background = (0.5, 0.5, 0.5)
        self.monitor_refresh_rate = 60
        self.screen_size = [800, 600] 
        self.full_screen = False
        self.radius_of_figure = 50
        self.distance_between_figures = 150
        self.flicker_rate = np.arange(1, self.number_of_hexagons + 1, 1)
    
    def pre_mainloop(self):
        self.init_graphics()
    
    def post_mainloop(self):
        pass

    def play_tick(self):
        self.screen.clear()
        self.viewport.draw()
        swap_buffers()
        for i in range(self.number_of_hexagons):
             self.stimulus[i].parameters.time_passed = time.clock()
    
    def init_graphics(self):
        VisionEgg.start_default_logging(); 
        VisionEgg.watch_exceptions()
        VisionEgg.config.VISIONEGG_FRAMELESS_WINDOW = 1
        VisionEgg.config.VISIONEGG_MONITOR_REFRESH_HZ = self.monitor_refresh_rate    
        VisionEgg.config.VISIONEGG_GUI_INIT = 0
        VisionEgg.config.VISIONEGG_SCREEN_W = self.screen_size[0]
        VisionEgg.config.VISIONEGG_SCREEN_H = self.screen_size[1]
        #if feedback on another screen - put it there
        if self.same_screen == False:
            os.environ['SDL_VIDEO_WINDOW_POS']="-800,1000"
        else:
            os.environ['SDL_VIDEO_CENTERED']="center"
        if self.full_screen == True:
            VisionEgg.config.VISIONEGG_FULLSCREEN = 1
        self.screen = get_default_screen()
        self.screen.parameters.bgcolor = self.color_of_background
        figure_positions = self.calculate_center_of_each_figure(
                self.number_of_hexagons, 
                self.radius_of_figure, 
                self.distance_between_figures,
                [self.screen_size[0]/2, self.screen_size[1]/2])
        self.stimulus = []
        for i in range(self.number_of_hexagons):
            now_letters_to_use = "abcdef"
            self.stimulus.append(Figure_Many(color_on = (0.0, 0.0, 0.0),
                                   color_off = (1.0, 1.0, 1.0),
                                   center_position = figure_positions[i],
                                   radius_size = self.radius_of_figure, 
                                   time_passed = 0,
                                   frequency_used = self.flicker_rate[i], 
                                   letters_to_use = now_letters_to_use,
                                   empty_letter_spot = i,
                                   screen_center = (self.screen.size[0]/2,self.screen.size[1]/2)
                                   ))
            
        self.viewport = Viewport(screen = self.screen , stimuli = self.stimulus)
            
    def calculate_center_of_each_figure(self, number_of_figures, radius_of_figure, 
                                        distance_between, center, start_at = np.pi/2):
        """calculates the center for the figures """
        all_positions = []
        if number_of_figures == 1:
            all_positions.append((self.screen_size[0]/2, self.screen_size[1]/2))
            return(all_positions)
        else:
            angle_between = 2 * np.pi / number_of_figures
            circle_size = (radius_of_figure + distance_between) * number_of_figures
            radius = circle_size/ (np.pi*2)
            a = center[0]
            b = center[1]
            for i in range(0, number_of_figures):
                all_positions.append((a + radius *p.cos(i * angle_between + start_at),
                                      b + radius *p.sin(i * angle_between + start_at))) 
            return all_positions       
    
if __name__ == '__main__':
    hs = Hex_skeleton()
    hs.on_init()
    hs.on_play()
    print "left all"        
