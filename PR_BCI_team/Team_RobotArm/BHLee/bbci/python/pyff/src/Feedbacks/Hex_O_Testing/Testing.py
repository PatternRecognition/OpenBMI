from VisionEgg import *
from VisionEgg.WrappedText import WrappedText
import numpy as np
import pylab as p
from Figure_Many import *
from VisionEgg.FlowControl import Presentation, FunctionController
from VisionEgg.Core import *
from VisionEgg.MoreStimuli import *
from VisionEgg.Text import *
try:
    from lib.eyetracker import EyeTracker
except:
    pass
from Top_Letters import *
from FeedbackBase.MainloopFeedback import MainloopFeedback
from VisionEgg.Gratings import SinGrating2D

class Testing(MainloopFeedback):
    START_EXP, END_EXP = 252, 253
    START_EXP, END_EXP = 252, 253
    COUNTDOWN_START = 0
    #different from pilot - here ALWAYS the start of trial animation is 36 
    START_ANIMATION = 36  
    START_TRIAL = 37
    SHORTPAUSE_START = 249
    INVALID = 66
    DISPLAY_START = 45
    WELCOME = 150
        
    def init(self):
        """ initializes all the settings of the figure(s)"""
        #variables to be set constantly by matlab
        self.classifier = []
        #frequencies to be used, calculated from training phase (given as indices)        
        self.f_index= [0, 2, 3, 4, 5, 6]

        self.use_class_update = 3 #also can choose 2
        #settings of different text displays
        self.top_word_main_color = (0.0,0.0,0.0)
        self.top_word_special_color = (1.0,0.0,0.0)  
        self.top_word_font_size = 18
        self.display_font_size = 75
                               
        #settings of the figures to be drawn
        self.number_of_hexagons = 6
        self.radius_of_figure = 50  
        self.distance_between_figures = 150 
        self.color_of_figure_on = (1.0,1.0,1.0)
        self.color_of_figure_off = (0.0,0.0,0.0)
        self.one_letter_font_size = 18 
        self.many_letters_font_size = 18
        self.size_of_change = 5    #if +/- is being used to adjust the hexagon size
        self.animate_hex = True
        self.number_of_trials = 10
                
        #settings about countdown
        self.color_countdown = (1.0,1.0,1.0)   
        self.round_count_down = 1    
        self.count_down_font_size = 200  
                        
        #other settings about display
        self.color_of_background = (0.5,0.5,0.5)  
        self.fixation_spot_color = (255,0,0,0) 
        self.fixation_spot_size = (8, 8)
        self.error_message = "Look in the middle"
        self.error_message2 = "Look at the target"
        self.overt_welcome = "Now you will see 6 hexagons. Each time the letter will be shown - you must LOOK at the hexagon displaying that letter, if not, error message will be shown and you will have to repeat the trial. When you are ready click any button and good luck"""
        self.covert_welcome = "Now you will see 6 hexagons. Each time the letter will be shown - you must look in the middle and CONCENTRATE at the hexagon displaying that letter, if not, error message will be shown and you will have to repeat the trial. When you are ready click any button and enjoy"""

        #settings of monitor  
        self.screen_size = [800, 600]  #[width, height]
        self.monitor_refresh_rate = 60  
        self.same_screen = True
        self.full_screen = False #not adequate if two screens used  
        self.platform = "windows" # "linux"

        #settings of the frequencies & timing
        self.time_of_trial = 1 #given in seconds
        self.time_of_count_down = 1
        self.time_word_display = 1
        self.flicker_rate = [4.0, 6.0, 7.5, 8.0, 10.0, 12.0]     
        self.time_animate = 4.5 
        
        # Eye tracker
        self.use_eyetracker = False
        self.et_fixate_center = True   # Whether center or fixdot 
                                       #has to be fixated   
        self.et_currentxy = (0, 0)      # Current fixation
        self.et_duration = 100
        self.et_targetxy = (100, 100)       # Target coordinates
        self.et_range = 200        # Maximum acceptable distance between target
                                   # and actual fixation (200)
        self.et_range_time = 200    # maximum acceptable fixation off the 
                                    #designated point 
        self.test_classify = True
    
    def pre_mainloop(self):
        """ prepares the feedback for being run """  
        #settings not to be changed normally
        if self.use_eyetracker:
            self.et = EyeTracker()
            self.et.start()
            self.presentation_stopped = False
        if self.et_fixate_center == True:
            self.fixation_spot = True 
        else:
            self.fixation_spot = False 
        if (len(self.overt_welcome)> 0 and self.et_fixate_center) or (len(self.covert_welcome) > 0 and (not self.et_fixate_center)): 
            self.welcome = True
        self.winning_class = [0, 0] #[for the hex idx, for the no.of winning pts]
        self.display_on = False    
        self.figure_to_be_used = 0
        self.classifier_reset = 0  
        self.spelled = ""
        self.number_of_trials = self.number_of_trials * 2
        self.use_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ><-&"
        self.look_in_the_middle = False
        if self.time_of_count_down > 0:
            self.countdown = True
        else:
            self.countdown = False  
        self.letters = "a"
        #initialize everything         
        self.init_graphics()
        self.init_figures()
        self.send_parallel(Testing.START_EXP)
        self.send_parallel(Testing.COUNTDOWN_START)
    
    def post_mainloop(self):
        self.send_parallel(Testing.END_EXP)
        self.on_quit()

    def play_tick(self):
        if self.welcome:
            self.send_parallel(Testing.DISPLAY_START)
            self.welcome_tick()
            self.welcome = False
        elif self.time_of_count_down > 0:
            self.letters = str(self.time_of_count_down)
            self.present_display.go()            
            self.time_of_count_down -= 1
        elif self.display_on:
            self.display_tick()
        elif self.number_of_trials > 0: 
            self.send_parallel(Testing.START_TRIAL)
            self.classifier = []
            if (self.number_of_trials % 2) == 0: 
                self.present_many_many.go()
            else:
                self.present_many_one.go()
            if self.passed:
                self.calc_average()
            self.number_of_trials = self.number_of_trials - 1
        else:
            #self.letters = "thank you, you spelled: " + self.spelled
            self.letters = self.spelled
            self.present_display.go()
            self.post_mainloop()

    def welcome_tick(self):
        """this will display the welcome message"""
        if self.et_fixate_center == True:
            msg = self.covert_welcome
        else:
            msg = self.overt_welcome
        text = WrappedText(text=msg,
                    color=self.color_countdown, # alpha is ignored (set with max_alpha_param)                    position=(self.screen.size[0]/2,self.screen.size[1]/2),
                    font_size= self.display_font_size,
                    position=(10,self.screen.size[1]-10))
        
        viewport = Viewport(screen=self.screen,
                    size=self.screen.size,
                    stimuli=[text])
        
        self.welcome_you = Presentation(go_duration=('forever',),viewports=[viewport])
        self.welcome_you.parameters.handle_event_callbacks = [(pygame.locals.KEYDOWN, self.keydown)] 
        self.welcome_you.go()
    
    def keydown(self, event):
        """controls the press of the keyboard keys"""
        global up, down, left, right
        self.welcome_you.parameters.go_duration = (0,'frames')
        if event.key == pygame.locals.K_ESCAPE:
            quit(event)
        elif event.key == pygame.locals.K_EQUALS:
            self.radius_of_figure = self.radius_of_figure + self.size_of_change
            self.many_letters_font_size = self.many_letters_font_size + self.size_of_change
            self.one_letter_font_size = self.one_letter_font_size + self.size_of_change
        elif event.key == pygame.locals.K_MINUS:
            self.radius_of_figure = self.radius_of_figure - self.size_of_change
            self.many_letters_font_size = self.many_letters_font_size - self.size_of_change
            self.one_letter_font_size = self.one_letter_font_size - self.size_of_change            
        else:
            print event.key
            
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

    def init_figures(self):
        top_letters = self.init_top_letters()
        self.present_many_many = self.init_many_many(self.use_letters, self.number_of_hexagons, self.flicker_rate, self.time_of_trial, add_letters = top_letters)
        #self.present_one_many = self.init_many_many("abcde", 1, [0], self.time_word_display)
        a = self.time_word_display
        self.present_many_stop = self.init_many_many(self.use_letters, self.number_of_hexagons, [a, a, a, a, a, a], a, add_letters = top_letters)
        self.present_many_one = self.init_many_many("abcdef", self.number_of_hexagons, self.flicker_rate, self.time_of_trial, add_letters = top_letters)
        self.present_display = self.init_text_to_display()
        self.present_one_stop = self.init_many_many("abcdef", self.number_of_hexagons, [a, a, a, a, a, a], a, add_letters = top_letters)
    def init_top_letters(self):
        word_on_the_top = Top_Letters(text=str(""),
            color=self.top_word_main_color, # alpha is ignored 
                                                        #(set with max_alpha_param)
            position = (10, (self.screen_size[1] - 20)))
        return word_on_the_top
        
    def init_text_to_display(self):
        """ initializes count down """
        text = Text(text=self.letters,
                    color=self.top_word_main_color, # alpha is ignored (set with max_alpha_param)
                    position=(self.screen.size[0]/2,self.screen.size[1]/2),
                    font_size= self.display_font_size,
                    anchor='center')           
#        text = WrappedText(text=self.letters,
#                    color=self.color_countdown, # alpha is ignored (set with max_alpha_param)                    position=(self.screen.size[0]/2,self.screen.size[1]/2),
#                    font_size= self.display_font_size,
#                    position=(10,self.screen.size[1]-10))
        
        viewport = Viewport(screen=self.screen,
                    size=self.screen.size,
                    stimuli=[text])
        present_display = Presentation(go_duration=(self.time_word_display,
                                                         'seconds'),viewports=[viewport])      
        
        text_controller = FunctionController(\
            during_go_func = self.control_display,\
            eval_frequency = Controller.EVERY_FRAME,\
            temporal_variables = \
            Controller.TIME_SEC_SINCE_GO)
    
        present_display.add_controller(text, 'text',\
               text_controller)                                  
        return present_display
        
    def init_many_many(self, letters_used, no_of_hex, flicker, timing, add_letters = None):
        self.figure_positions = self.calculate_center_of_each_figure(
                no_of_hex, 
                self.radius_of_figure, 
                self.distance_between_figures,
                [self.screen_size[0]/2, self.screen_size[1]/2])
        stimulus = []
        for i in range(no_of_hex):
            if (len(letters_used) > 6) or (no_of_hex == 1):
                use_letters = letters_used[i * 5:i*5 + 5]
                now_letters_to_use = use_letters[:(i + 3 ) % 6] + " " \
                                 + use_letters[(i + 3) % 6:]
            else:
                now_letters_to_use = letters_used[i]
            stimulus.append(Figure_Many(color_on = (0.0, 0.0, 0.0),
                                   color_off = (1.0, 1.0, 1.0),
                                   center_position = self.figure_positions[i],
                                   radius_size = self.radius_of_figure, 
                                   time_passed = 0,
                                   frequency_used = flicker[i], 
                                   letters_to_use = now_letters_to_use,
                                   empty_letter_spot = i,
                                   screen_center = (self.screen.size[0]/2,self.screen.size[1]/2),
                                   letters_number = len(now_letters_to_use)
                                   ))
        stimulus.append(self.define_fix_spot(no_of_hex))
        if not add_letters == None:
            additional = -2
            stimulus.append(add_letters)
        else:
            additional = -1
        viewport = Viewport(screen = self.screen , stimuli = stimulus)
        present_figure = Presentation(go_duration=(timing, 'seconds'), 
                                       viewports=[viewport])  
        
        figure_controller = FunctionController(\
            during_go_func = self.control_figure_timing,\
            eval_frequency = Controller.EVERY_FRAME,\
            temporal_variables = \
            Controller.TIME_SEC_SINCE_GO)
        
        [present_figure.add_controller(i, 'time_passed', figure_controller) for i in stimulus[:additional]]
        return present_figure

    def define_fix_spot(self, no_of_hex):
        if (self.fixation_spot) and not (no_of_hex == 1):   
            self.fix_size = self.fixation_spot_size   
        else:
            self.fix_size = (0,0)   
            
        fixation_spot = FixationSpot(position=(self.screen.size[0]/2,self.screen.size[1]/2),
                     anchor='center',
                     color=self.fixation_spot_color,
                     size = self.fix_size)    
        return fixation_spot
    
    def control_figure_timing(self, t):
        self.passed = True
        if self.use_eyetracker: 
            self.check_fix_ok()
        self.classify()
        return t    
    
    def control_display(self, t):
        return self.letters
    
    def classify(self):
        try:
            if not self.test_classify:
                OUT= self._data.get("cl_output")
            else:
                OUT = np.random.rand(8)
            OUT_arr = np.asarray(OUT)
            new_input = OUT_arr[np.asarray(self.f_index)]
            
            if self.classifier_reset == 1:
                print "classifier on, self.classifier reset"
                
            if self.use_class_update == 1:
                if len(self.classifier) == 0 or self.classifier_reset == 1:
                    self.classifier = [0, 0, 0, 0, 0, 0]
                    self.classifier_reset = 0
                else:
                    self.classifier = np.add(self.classifier,np.multiply(0.05, (np.subtract(new_input, self.old_classifier))))            
                self.old_classifier = new_input
            elif self.use_class_update == 2:
                self.classifier = new_input 
                maxim = np.max(self.classifier)
                try:
                    self.classifier = self.classifier.tolist()
                except:
                    pass
                if self.winning_class[1] == 0:
                    self.winning_class = [self.classifier.index(maxim), 1]
                    #print "now winning: ", self.winning_class
                elif self.winning_class[0] == self.classifier.index(maxim):
                    self.winning_class[1] = self.winning_class[1] + 1
                    #print "getting more: ", self.winning_class
                else:
                    self.winning_class[1] = self.winning_class[1] - 1 
                    #print "getting less: ", self.winning_class
            elif self.use_class_update == 3:
                self.classifier.append(new_input)
        except:
            print "excepted"
            OUT = [0, 0, 0, 0, 0, 0, 0, 0]

    def calc_average(self):
        if self.use_class_update == 1:
            if len(self.classifier) == 0:
                self.classifier = [[0, 3, 2, 2, 2, 2]]
            maxim = np.max(self.classifier)
            try:
                self.classifier = self.classifier.tolist()
            except:
                pass
            self.use_hex = self.classifier.index(maxim)
            
        elif self.use_class_update == 2:
            self.use_hex = self.winning_class[0]
        elif self.use_class_update == 3:
            print self.classifier, "class"
            a = np.max(self.classifier, 0)
            print a, "class po"
            maxim = np.max(a)
            print maxim, "maxim"
            try:
                a = a.tolist()
            except:
                pass
            #self.use_hex = self.classifier.index(maxim)
            print a, "a"
            self.use_hex = a.index(maxim)
        self.display_on = True          
                    
    def check_fix_ok(self):
        """checks by means of eye tracker if fixation is ok"""
        if self.et.x is None:
            print "No eyetracker data received!"
            self.on_stop()
            return t
        self.et_currentxy = (self.et.x, self.et.y)
        self.et_duration = self.et.duration
        tx, ty = self.et_targetxy[0], self.et_targetxy[1]
        cx, cy = self.et_currentxy[0], self.et_currentxy[1]
        dist = math.sqrt(math.pow(tx - cx, 2) + math.pow(ty - cy, 2))
        #self.et_outside = 0
        # Check if current fixation is outside the accepted range    
        if dist > self.et_range:
            #self.et_outside = 1
            # Check if the off-fixation is beyond temporal limits
            if self.et_duration > self.et_range_time:
                self.send_parallel(self.INVALID)
                self.present_many_many.parameters.go_duration = (0,'frames')
                self.present_many_one.parameters.go_duration = (0,'frames')
                if self.et_fixate_center == True:
                    self.next_to_display = self.error_message
                else:
                    self.next_to_display = self.error_message2
                #self.show_display()
                self.presentation_stopped = True
                #self.sound_invalid.play()
                self.play_sound("winSpaceCritStop.wav")
                self.passed = False
                self.number_of_trials = self.number_of_trials + 1
        
    def display_tick(self):
        #print self.use_hex, "use_hex"
        self.send_parallel(Testing.SHORTPAUSE_START)

        col1 = (0.0, 0.0, 0.0)
        col2 = (1.0, 1.0, 1.0)
        if not (self.number_of_trials % 2) == 0:
            stimuli = self.present_many_stop.parameters.viewports
            [stimuli] = stimuli
            stimuli = stimuli.parameters.stimuli
            for i in range(6):

                if i == self.use_hex:
                   stimuli[i].set_cols(col1, col2)
                else:
                   stimuli[i].set_cols(col2, col1)
            #stimuli[self.use_hex].set_color = (0.0, 0.0, 0.0)
            stimuli =self.present_many_many.parameters.viewports
            [stimuli] = stimuli
            stimuli = stimuli.parameters.stimuli
            hex_used = stimuli[self.use_hex]
            self.letters = hex_used.get_my_letters()
#            #print letters
            stimuli = self.present_many_one.parameters.viewports
            [stimuli] = stimuli
            stimuli = stimuli.parameters.stimuli
            for i in range(len(self.letters)):
                stimuli[i].set_my_letters(self.letters[i])
            
#            stimuli = self.present_one_many.parameters.viewports
#            [stimuli] = stimuli
#            stimuli = stimuli.parameters.stimuli 
#            stimuli[0].set_my_letters(self.letters) 
#            #position = self.figure_positions[self.use_hex]  
#            #stimuli[0].set_position(position)        
            self.present_many_stop.go()
#            
        else:
            stimuli = self.present_one_stop.parameters.viewports
            [stimuli] = stimuli
            stimuli = stimuli.parameters.stimuli
            for i in range(6):
                if i == self.use_hex:
                   stimuli[i].set_cols(col1, col2)
                else:
                   stimuli[i].set_cols(col2, col1)
            #stimuli[self.use_hex].set_color = (0.0, 0.0, 0.0)
            stimuli1 = self.present_one_stop.parameters.viewports

            [stimuli1] = stimuli1

            stimuli1 = stimuli1.parameters.stimuli
            for i in range(len(self.letters)):
                stimuli1[i].set_my_letters(self.letters[i])
        
            stimuli =self.present_many_many.parameters.viewports
            [stimuli] = stimuli
            stimuli = stimuli.parameters.stimuli
#            print len(self.letters), "len letters"
            old_letters = self.letters
            self.letters = self.letters[self.use_hex]
#           
            print self.letters, "letters" 
            if self.letters == "<" or self.letters == " ":
                self.number_of_trials = self.number_of_trials + 2 
                
                if (self.letters == "<") and len(self.spelled) > 0:
                    self.spelled = self.spelled[:-1]
#                
            else:
                self.spelled = self.spelled + self.letters
                print self.spelled, "spelled"
                stimuli[-1].set_my_letters(self.spelled)   
                #stimuli[-1].set_position(position)
#            self.present_display.go()
            self.present_one_stop.go()
            self.letters = old_letters
        self.display_on = False

    def play_sound(self, file_name):
        """ plays given sound """ 
        file= file_name
        print "playing ", file
        if self.platform == "windows":
           from winsound import PlaySound, SND_FILENAME, SND_ASYNC
           PlaySound(file, SND_FILENAME|SND_ASYNC)
        elif self.platform == "linux":
           from wave import open as waveOpen
           from ossaudiodev import open as ossOpen
           s = waveOpen('tada.wav','rb')
           (nc,sw,fr,nf,comptype, compname) = s.getparams( )
           dsp = ossOpen('/dev/dsp','w')
           try:
             from ossaudiodev import AFMT_S16_NE
           except ImportError:
             if byteorder == "little":
               AFMT_S16_NE = ossaudiodev.AFMT_S16_LE
             else:
               AFMT_S16_NE = ossaudiodev.AFMT_S16_BE
           dsp.setparameters(AFMT_S16_NE, nc, fr)
           data = s.readframes(nf)
           s.close()
           dsp.write(data)
           dsp.close()  
                              
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
    hs = Testing()
    hs.on_init()
    hs.on_play()
    print "left all"        