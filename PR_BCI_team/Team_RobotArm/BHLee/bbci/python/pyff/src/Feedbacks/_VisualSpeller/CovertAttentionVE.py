'''Feedbacks._VisualSpeller.CovertAttentionVE
# Copyright (C) 2010  "Nico Schmidt"
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

Created on Nov 17, 2010

@author: "Nico Schmidt"
'''

import logging, pygame, math, os, datetime
import numpy as NP
from time import sleep
from sys import platform
if platform == 'win32':
    import winsound

from FeedbackBase.VisionEggFeedback import VisionEggFeedback
from lib.P300Layout.CircularLayout import CircularLayout
from lib.marker import TRIAL_START, COUNTDOWN_START
from lib.eyetracker import EyeTracker
#from VEShapes import FilledTriangle
from VisionEgg.MoreStimuli import FilledCircle, Arrow, Target2D
from VisionEgg.Text import Text, PangoText

def mytime():
    """ Return microsecond-accurate time since last midnight. 
    Workaround for time() having only 10ms accuracy when VE is running.
    """
    n = datetime.datetime.now()
    return 60. * (60 * n.hour + n.minute) + n.second + n.microsecond / 1000000.

class CovertAttentionVE(VisionEggFeedback):
    '''
    classdocs
    '''
    
    # Triggers:
    MASKER = 9
    CUE = 10 # 10-10+nr_elements-1
    TARGET = 30 # 30 to 30+nr_elements-1 for target; 30+nr_elements to 30+2*nr_elements-1 for nontarget
    KEY_1, KEY_2 = 100,101 # KEY_1=
    INVALID_FIXATION = 99
        
    
    def init(self):
        
        self.log_filename = "CovertAttentionVE.log"
        
        self.geometry = [0, 0, 1280, 800]
        self.fullscreen = False

        # Init trigger
        self.TRIAL_START = TRIAL_START         # wird in Matlab anders gesetzt
        self.COUNTDOWN_START = COUNTDOWN_START # wird in matlab anders gesetzt
       
        # sizes:
        self.fixationpoint_radius = 5
        self.arrow_size = (60., 12.)
        self.screen_radius = 300
        self.circle_radius = 90
        self.target_size = 120
        self.countdown_font_size = 150     
        self.input_box_size = (600, 70)
        self.input_text_size = 60
        
        # trials:
        self.nr_elements = 4 # number of directions
        self.nr_trials = 10 # number of trials
        self.nr_stimuli = 5 # number of stimuli per trial
        self.p_correct = 0.7 # fraction of correct trials (cue=target) / IN 
        
        # stimulus options
        self.target_symbols = [u'\u00D7', '+'] # first target, second nontarget
        self.use_masker = True
        
        # times:
        self.nr_countdown = 2
        self.fixation_duration = 1
        self.cue_duration = 0.1
        self.trial_interval = [1., 10.]
        self.stimulus_jitter = [-0.5, 1.5]
#        self.stimulus_jitter = [[-0.1, 0.5],[-10,10],[0,0],[10,15],[-1.5,1.5]]
        self.target_duration = 0.2
        self.masker_duration = 0.14
        self.end_duration = 0.5
        self.pause_duration = [1, 5]
        
        # colors:
        self.arrow_color = (1, 1, 1) # masker cues
        self.cue_color = (0,0,1) # attended cue
        self.bg_color = 'black'
        self.countdown_color = (0, 0, 1)
        self.target_color = (0.7, 0.7, 0.7)
        self.circle_color = (1, 1, 1)
        self.circle_textcolor = (0, 0, 0)
        self.fixationpoint_color = (1, 1, 1)
        self.input_box_color = (0.2, 0.2, 0.2)
        self.input_text_color = (1, 1, 1)
#        self.cue_colors = [(0.0, 1.0, 0.8), \
#                           (0.2, 0.0, 0.8), \
#                           (0.4, 1.0, 0.8), \
#                           (0.2, 0.0, 0.8), \
#                           (0.7, 1.0, 0.8), \
#                           (0.2, 0.0, 0.8)]
#        self.cue_bordercolor = (0.0, 0.0, 1.0)
#        self.user_cuecolor = 4 # the index of the color, which is used as cue 
        
        self.multi_stimulus_mode = True
        self.start_angle = 0
        
        # Eyetracker settings
        self.use_eyetracker = False
        self.et_range = 100        # Maximum acceptable distance between target and actual fixation
        self.et_range_time = 200    # maximum acceptable fixation off the designated point
        
        # properties of idle function after __run_tick() finished
        self.idle_fs = 0.05
        self.idle_max_time = 0.1 # choose this carefully. if too short, screen presentation will be inaccurate!
    
    
    def run(self):
        """
        Main run method.
        """
        NP.random.seed()
        
        # setup logger:
        handler = logging.FileHandler(self.log_filename, 'w')
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s: %(message)s")
        handler.setFormatter(formatter)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(handler)
        
        # catch some parameter-errors
        if not self.multi_stimulus_mode:
            self.nr_stimuli = 1
        if self.nr_stimuli==1:
            self.multi_stimulus_mode = False
        if self.nr_stimuli<1:
            self.logger.error("can't run with %d stimuli. using 1." % self.nr_stimuli)
            self.nr_stimuli = 1
        if len(self.stimulus_jitter) != 2 and len(self.stimulus_jitter) != self.nr_stimuli:
            self.logger.error("wrong jitter format. Jitter has to be a list of length 2 [min, max], or a list of length nr_stimuli with [[min1,max1], [min2,max2], ...].")
            return
        elif len(self.stimulus_jitter) == self.nr_stimuli:
            for i in range(self.nr_stimuli):
                if type(self.stimulus_jitter[i])==list and len(self.stimulus_jitter[i]) != 2:
                    self.logger.warning("wrong jitter format. Jitter has to be a list of length 2 [min, max], or a list of length nr_stimuli with [[min1,max1], [min2,max2], ...].")
                    return
        if not self.use_masker:
            self.masker_duration = 0.0
        
        self._centerPos = [self.geometry[2]/2., self.geometry[3]/2.]
        
        self._accept_response = False
        self._accept_end = False
        self._response_text = ""
        self._cues = []
        self._circles = []
        self._letters = []
        
        # Feedback state booleans
        self.__reset_states()
        
        self._current_trial = 0
        self._current_stimulus = 1
        self._current_countdown = self.nr_countdown
        
        self._restart_trial = False
        
        
        # calc needed time:
#        self.stat_time_needed = sum([t[3] for t in self._trial_list]) + self.nr_trials * (self.fixation_duration +
#                                                                                      self.cue_duration +
#                                                                                      self.responsetime_duration + 290 +
#                                                                                      self.rest_duration) + self.nr_countdown
#        
#        print "time needed for experiment: approx. %d:%d" % (NP.floor(self.stat_time_needed/1000/60.), NP.mod(self.stat_time_needed/1000, 60)), "(mm:ss)"
#        self.save_triallist_stats()
        
#        self.cue_color = [tuple([255*i for i in colorsys.hsv_to_rgb(*self.cue_colors[j])]) for j in xrange(self.nr_elements)]
#        self.cue_color.append(tuple([255*i for i in colorsys.hsv_to_rgb(*self.cue_bordercolor)]))
        
        self.__build_trials()
        self.__init_screen()
        
        
        
        # trials
        self._hits = 0
        while self._current_trial < self.nr_trials: 

            # countdown
            self._current_countdown = self.nr_countdown
            countdown = self.stimulus_sequence(self.__countdown_tick, 1.)
            countdown.run()

            self.trigger = self.TRIAL_START
            #print "TR IAL",self._current_trial
            # create stimulus sequence
            self._stimulus_durations = self.__build_stimulus_durations()
            self._stimulus_sequence = self.stimulus_sequence(self.__run_tick, self._stimulus_durations)
            self._i_ = -1 # current duration index-1
            self._stimulus_sequence._last_start = datetime.datetime.now() #mytime()
            
            # run trial:
            self._stimulus_sequence.run()
            
            if self._restart_trial:
                self._restart_trial = False 
                self.__reset_states()
                self.__reset_screen()
                self._current_stimulus = 1
                self._current_countdown = self.nr_countdown
                #countdown = self.stimulus_sequence(self.__countdown_tick, 1.)
                #countdown.run()
                continue
             
            # get response 
            if self.multi_stimulus_mode:
                self._accept_response = True
            response = self.stimulus_sequence(self.__wait_for_response, 0.05)
            response.run()
            self._current_trial += 1
            
            # short pause: 
            self._state_pause = True
            self._pause_sequence = self.stimulus_sequence(self.__pause, NP.random.rand()*NP.diff(self.pause_duration)[0] + self.pause_duration[0])
            self._pause_sequence.run()
             
        self._accept_end = True
        response = self.stimulus_sequence(self.__feedback, 0.05)
        response.run()
        
        # Exit the feedback main loop
        if self.use_eyetracker:
            self.et.stop()
        self.on_stop()
        self.quit()
            
    
    def __init_screen(self):
        """
        Initialize the surfaces and fonts.
        """
        
        # add circles and targets:
        circle_layout = CircularLayout(nr_elements=self.nr_elements,
                                       radius=self.screen_radius,
                                       start=(90-self.start_angle+360/self.nr_elements)/360.*2*NP.pi)
        circle_layout.positions.reverse()
        self._circle_positions = [(x+self.geometry[2]/2, y+self.geometry[3]/2) for (x,y) in circle_layout.positions]
        for i in xrange(self.nr_elements):
            self._circles.append(FilledCircle(radius=self.circle_radius,
                                              position=self._circle_positions[i],
                                              color=self.circle_color)) #(1,i*1./self.nr_elements,i*1./self.nr_elements)))#self.circle_color))
            self._letters.append(PangoText(position=self._circle_positions[i],
                                           text=" ",
                                           font_descr_string="serif "+str(self.target_size),
                                           color=self.target_color,
                                           anchor='center',
                                           on=False))
        
        # add maskers:
        if self.use_masker:
            for i in xrange(self.nr_elements*(len(self.target_symbols)-1)):
                self._letters.append(PangoText(position=self._circle_positions[i%self.nr_elements],
                                               text=" ",
                                               font_descr_string="serif "+str(self.target_size),
                                               color=self.target_color,
                                               anchor='center',
                                               on=False))
        self.add_stimuli(*self._circles)
        self.add_stimuli(*self._letters)
        
        # add cue:
        for i in xrange(self.nr_elements):
            self._cues.append(Arrow(color=self.arrow_color,
                                    position=self._centerPos,
                                    size=self.arrow_size,
                                    on=False))
        self._cues[self.nr_elements-1].set(color=self.cue_color)
        self.add_stimuli(*self._cues)
        
        # add fixation point:
        self._fixationpoint = FilledCircle(radius=self.fixationpoint_radius,
                                           position=self._centerPos,
                                           color=self.fixationpoint_color)
        self.add_stimuli(self._fixationpoint)
        
        # add countdown
        self._countdown = Text(position=self._centerPos,
                               text=" ",
                               font_size=self.countdown_font_size,
                               color=self.countdown_color,
                               anchor='center',
                               on=False)
        self.add_stimuli(self._countdown)
        
        # add message with input box
        self._input_box = Target2D(position=(self._centerPos[0],self._centerPos[1]+self.countdown_font_size),
                                   size=self.input_box_size,
                                   color=self.input_box_color,
                                   on=False)
        self._input_text= Text(position=(self._centerPos[0],self._centerPos[1]+self.countdown_font_size),
                               text=" ",
                               font_size=self.input_text_size,
                               color=self.input_text_color,
                               anchor='center',
                               on=False)
        self.add_stimuli(self._input_box)
        self.add_stimuli(self._input_text)
        
        
    def __reset_screen(self):
        for c in self._circles:
            c.set(on=True)
        for l in self._letters:
            l.set(on=False)
        for c in self._cues:
            c.set(on=False)
        self._fixationpoint.set(on=False)
        self._countdown.set(on=False)
        self._input_box.set(on=False)
        self._input_text.set(on=False)
        
    def __reset_states(self):
        self._state_fixation = True
        self._state_cue = False
        self._state_shifting_attention = False
        self._state_target = False
        self._state_masker = False
        self._state_response = False
        self._state_rest = False  
        
    
    def __build_trials(self):
        """
        build the list of trials as:
        [[nr_targets, [target_direction, cue_direction, symbol, stimulus-jitter], ...additional stimuli...],
         [nr_targets, [target_direction, cue_direction, symbol, stimulus-jitter], ...additional stimuli...],
         ...additional trials...
         [nr_targets, [target_direction, cue_direction, symbol, stimulus-jitter], ...additional stimuli...]]
        build the trials semi-deterministic for the cue/target location and stimulus-jitter
        but probabilistic for the validity and target symbol.
        """
        self._trial_list = []
            
        # balance (but shuffle) cue_directions:
        cue_direction = range(self.nr_elements)*NP.ceil(float(self.nr_trials)/self.nr_elements)
        cue_direction = cue_direction[:self.nr_trials]
        NP.random.shuffle(cue_direction)
        
        for t in range(self.nr_trials):
            # define number of targets:
            nr_targets = NP.random.randint(self.nr_stimuli+1)
            trial = [nr_targets]
            
            # define at which stimuli the targets appear:
            target_appearance = NP.append(NP.zeros(trial[0]), NP.ones(self.nr_stimuli-trial[0])) #[0 for _ in range(trial[0])]+[1 for _ in range(self.nr_stimuli-trial[0])]
            NP.random.shuffle(target_appearance)
                
            # target direction
            if self.multi_stimulus_mode or NP.random.rand()<=self.p_correct:
                # valid condition (in multi-stimulus-mode always valid)
                target_direction = cue_direction[t]
            else:
                # invalid condition (cue different from target)
                target_direction = NP.random.randint(self.nr_elements)
                if target_direction>=cue_direction[t]:
                    target_direction += 1
            
            # define the stimuli-jitter:
            for s in range(self.nr_stimuli):
                if (len(self.stimulus_jitter) == 2 and
                    (type(self.stimulus_jitter[0])==int or type(self.stimulus_jitter[0])==float) and
                    (type(self.stimulus_jitter[1])==int or type(self.stimulus_jitter[1])==float)):
                    # same jitter-range for all stimuli
                    jitter = NP.random.rand()*NP.diff(self.stimulus_jitter)[0] + self.stimulus_jitter[0]
                else:
                    # individual jitter-range for each stimulus
                    jitter = NP.random.rand()*NP.diff(self.stimulus_jitter[s])[0] + self.stimulus_jitter[s][0]
                            
                # save stimulus
                trial.append([target_direction,
                              cue_direction[t],
                              int(target_appearance[s]),
                              jitter])
            # save trial
            self._trial_list.append(trial)
        #print self._trial_list
    
    def __build_stimulus_durations(self):
        durations = []
        stimuli_distance = self.trial_interval[1] / (self.nr_stimuli - 1.)
        last_ival = self.trial_interval[0]
        durations.append(self.fixation_duration)
        durations.append(self.cue_duration)
        for s in xrange(1,self.nr_stimuli+1): # start with 1 (0 is nTargets)
            jitter = self._trial_list[self._current_trial][s][3]
            durations.append(max(0.0, last_ival + jitter)) # ISI + jitter, but not negative
            durations.append(self.target_duration)
            if self.use_masker:
                durations.append(self.masker_duration)
            last_ival = stimuli_distance - self.target_duration - self.masker_duration - jitter
        durations.append(self.end_duration)
        #print durations
        return durations
    
    def __wait_for_response(self):
        if self._accept_response:
            if self.multi_stimulus_mode:
                self._input_box.set(on=True)
                self._input_text.set(on=True, text="Nr. targets: "+self._response_text)
            return True
        else:
            return False
        
    def __pause(self):
        if self._state_pause:
            self._input_box.set(on=False)
            self._input_text.set(on=False)
            self._state_pause = False
            return True
        else:
            return False
    
    def __feedback(self):
        if self._accept_end:
            self._input_box.set(on=True)
            self._input_text.set(on=True, text="Correct Hits: %d/%d (%2.2f%%)" %(self._hits, self.nr_trials, 100.*(float(self._hits)/self.nr_trials)))
            return True
        else:
            return False
        
    
    def __countdown_tick(self):
        self._input_box.set(on=True)
        self._input_text.set(on=True, text="Your target: "+self.target_symbols[0])
        self._countdown.set(on=True, text=str(self._current_countdown))
        if self._current_countdown >= 1:
            if self._current_countdown == self.nr_countdown:
                #self.send_parallel(self.COUNTDOWN_START)
                self.logger.info("[TRIGGER] %d" % self.COUNTDOWN_START)
            self._current_countdown -= 1
            return True
    
    
    def __run_tick(self):
        """
        called every frame, if in play mode.
        """
            
        running = False
        if self.trigger is not None:
            self.send_parallel(self.trigger)
            self.logger.info("[TRIGGER] %d" % self.trigger)
        
        if self.use_eyetracker and not self.__eyetracker_input():
            # restart trial on eye movements:
            self._restart_trial = True
            return False
        
        if self._state_fixation:
            #print "__fixation_tick"
            self.__fixation_tick()
            running = True
        elif self._state_cue:
            #print "__cue_tick"
            self.__cue_tick()
            running = True
        elif self._state_shifting_attention:
            #print "__shifting_attention_tick"
            self.__shifting_attention_tick()
            running = True
        elif self._state_target:
            #print "_state_target"
            self.__target_tick()
            running = True
        elif self._state_masker:
            #print "_state_masker"
            self.__masker_tick()
            running = True
        else:
            self._state_fixation = True
            #print "end_trial"
        
        
        if self._i_ >= 0:
#            if self._i_ < len(self._stimulus_durations):
#                print self._stimulus_durations[self._i_]
#            else:
#                print "out of durations"
            # .second
            remaining_time = (self._stimulus_sequence._last_start.second - mytime() + 
                              self._stimulus_durations[self._i_])
            while running and remaining_time > self.idle_max_time:
#                print "remaining time:", remaining_time
                running = self.__idle()
                sleep(self.idle_fs)
                remaining_time = (self._stimulus_sequence._last_start.second - mytime() + 
                                  self._stimulus_durations[self._i_])
            
        self._i_ += 1
        return running
    
    def __idle(self):
        if self.use_eyetracker and not self.__eyetracker_input():
            # restart trial on eye movements:
            self._restart_trial = True
            return False
        else:
            return True
            
    
    def __fixation_tick(self):
        self._countdown.set(on=False)
        self._input_box.set(on=False)
        self._input_text.set(on=False)
        self._fixationpoint.set(on=True)
        self._state_fixation = False
        self._state_cue = True
        self.trigger = None
        
    def __cue_tick(self):
        self._fixationpoint.set(on=False)
        # turn on cue:
        #print self._current_trial, self._current_stimulus, self._trial_list[self._current_trial][self._current_stimulus][1]
        orientation_deg = -90.+self.start_angle + self._trial_list[self._current_trial][self._current_stimulus][1] * 360. / self.nr_elements
        orientation_rad = -orientation_deg/360. * 2*NP.pi
        #print "CUE POINTS TO DIRECTION", self._trial_list[self._current_trial][self._current_stimulus][1], ":", orientation_deg, "deg/", orientation_rad, "rad"
        self._cues[self.nr_elements-1].set(on=True,
                                           orientation=orientation_deg,
                                           position=(self._centerPos[0]+NP.cos(orientation_rad)*self.arrow_size[0]/2.,
                                                     self._centerPos[1]+NP.sin(orientation_rad)*self.arrow_size[0]/2.))
        
        # turn on masker cues:
        for i in range(self.nr_elements-1):
            i_ = 0 if i<self._trial_list[self._current_trial][self._current_stimulus][1] else 1
            orientation = -90.+self.start_angle + (i+i_)*360./self.nr_elements
            self._cues[i].set(on=True,
                              orientation=orientation,
                              position=(self._centerPos[0]+NP.cos(-orientation/360. * 2*NP.pi)*self.arrow_size[0]/2.,
                                        self._centerPos[1]+NP.sin(-orientation/360. * 2*NP.pi)*self.arrow_size[0]/2.))
        self._state_cue = False
        self._state_shifting_attention = True
        self.trigger = self.CUE + self._trial_list[self._current_trial][1][1]
    
    def __shifting_attention_tick(self):
        for i in range(self.nr_elements):
            self._cues[i].set(on=False)
        for i in xrange(len(self._letters)):
            self._letters[i].set(on=False)
        self._fixationpoint.set(on=True)
        self._state_shifting_attention = False
        if self._current_stimulus > self.nr_stimuli:
            self._current_stimulus = 1
        else:
            self._state_target = True
        self.trigger = None
   
    def __target_tick(self):
        target_direction = self._trial_list[self._current_trial][self._current_stimulus][0]
        target_symbol = self._trial_list[self._current_trial][self._current_stimulus][2]
        
        # enable target symbol
        #print self._letters
        #print target_direction
        self._letters[target_direction].set(on=True, text=self.target_symbols[target_symbol])
        
        if self.multi_stimulus_mode:
            # set random symbols to the other circles
            for i in range(target_direction)+range(target_direction+1, self.nr_elements):
                self._letters[i].set(on=True, text=self.target_symbols[NP.random.randint(2)])
        else:
            self._accept_response = True
        self._state_target = False
        if self.use_masker:
            self._state_masker = True
        else:
            self._state_shifting_attention = True
            self._current_stimulus += 1
        if self.multi_stimulus_mode:
            self.trigger = self.TARGET + target_symbol
        else:
            self.trigger = self.TARGET + target_direction + self.nr_elements*target_symbol
    
    def __masker_tick(self):
        # turn on all target symbols at all or one circle
        for j in xrange(len(self.target_symbols)):
            if self.multi_stimulus_mode:
                for i in xrange(self.nr_elements):
                    self._letters[i+j*self.nr_elements].set(on=True,text=self.target_symbols[j])
            else:
                target_direction = self._trial_list[self._current_trial][self._current_stimulus][0]
                self._letters[target_direction+j*self.nr_elements].set(on=True,text=self.target_symbols[j])
        self._state_masker = False
        self._state_shifting_attention = True
        self._current_stimulus += 1
        self.trigger = None       # kein Masker trigger um ueberlagerung von triggern zu vermeiden
        #self.trigger = self.MASKER
        

    def __eyetracker_input(self):
        # Control eye tracker
        if self.et.x is None:
            #print("[ERP Hex] No eyetracker data received!")
            self.logger.warning("[EYE_TRACKER] No eyetracker data received!")
        tx, ty = self._centerPos[0], self._centerPos[1]
        cx, cy = self.et.x, self.et.y
        if type(self.et.x)!=int or type(self.et.y)!=int:
            self.logger.error("no eyetracker input. stopping...")
            self.on_stop()
        dist = math.sqrt(math.pow(tx - cx, 2) + math.pow(ty - cy, 2))
        #self.logger.info("[EYE_TRACKER] position=(%f,%f)" % (self.et.x, self.et.y))
        # Check if current fixation is outside the accepted range 
        if dist > self.et_range:
            # Check if the off-fixation is beyond temporal limits
            if self.et.duration > self.et_range_time:
                # Break off current trial !!
                if platform == 'win32':
                    winsound.PlaySound(self.soundfile, winsound.SND_ASYNC)

                # Send break-off trigger
                self.send_parallel(self.INVALID_FIXATION)
                self.logger.info("[TRIGGER] %d" % self.INVALID_FIXATION)
                return False
        return True
    
    
    def keyboard_input(self, event):
        """ Handle pygame events like keyboard input. """
        quit_keys = [pygame.K_q, pygame.K_ESCAPE]
        if event.key in quit_keys or event.type == pygame.QUIT:
            self.quit()
        if self._accept_response:
            if self.multi_stimulus_mode:
                if event.key >= pygame.K_0 and event.key <= pygame.K_9:
                    self._response_text += chr(event.key)
                    print self._response_text
                elif event.key >= pygame.K_KP0 and event.key <= pygame.K_KP9:
                    self._response_text += chr(event.key-208)
                elif event.key == pygame.K_BACKSPACE:
                    self._response_text = self._response_text[:-1]
                elif event.key == pygame.K_KP_ENTER or event.key == pygame.K_RETURN:
                    if len(self._response_text)>0 and int(self._response_text)<=self.nr_stimuli:
                        self._accept_response = False
                        if self.KEY_1:
                            self.send_parallel(self.KEY_1 + int(self._response_text))
                            self.logger.info("[TRIGGER] %d" % (self.KEY_1 + int(self._response_text)))
                        if int(self._response_text)==self._trial_list[self._current_trial][0]:
                            self._hits += 1
                        print "Total hits: %d, response: %d, trial list[0]: %d" % (self._hits,int(self._response_text),self._trial_list[self._current_trial][0])
                        self._response_text = ""
            else:
                if event.key == pygame.K_LCTRL: # Plus (+)-key was pressed
                    self.send_parallel(self.KEY_1)
                    self.logger.info("[TRIGGER] %d" % self.KEY_1)
                    self._accept_response = False
                    #if self._trial_list[self._current_trial][self._current_stimulus][2] == 1:
                    #    print "correct"
                    #else:
                    #    print "wrong"
                elif event.key == pygame.K_RCTRL: # (x)-key was pressed
                    self.send_parallel(self.KEY_2)
                    self.logger.info("[TRIGGER] %d" % self.KEY_2)
                    self._accept_response = False
                    #if self._trial_list[self._current_trial][self._current_stimulus][2] == 2:
                    #    print "correct"
                    #else:
                    #    print "wrong"
        if self._accept_end:
            if event.key == pygame.K_KP_ENTER or event.key == pygame.K_RETURN:
                self._accept_end = False
    
#    def evaluate_triallist(self):
#        self.stat_symbol_locations = NP.zeros([2,6])
#        self.stat_symbol_locations[self._trial_list[0][2]-1, self._trial_list[0][0]] += 1
#        self.stat_followers_target = NP.zeros([6,6])
#        self.stat_followers_cue = NP.zeros([6,6])
#        this_target = self._trial_list[0][0]
#        this_cue = self._trial_list[0][1]
#        next_target = -1
#        next_cue = -1
#        for i in xrange(1,self.nr_trials):
#            next_target = self._trial_list[i][0]
#            next_cue = self._trial_list[i][1]
#            self.stat_followers_target[this_target, next_target] += 1
#            self.stat_followers_cue[this_cue, next_cue] += 1
#            self.stat_symbol_locations[self._trial_list[i][2]-1, next_target] += 1
#            this_target = next_target
#            this_cue = next_cue
#        
#    def save_triallist_stats(self):
#        from datetime import datetime
#        
#        self.evaluate_triallist()
#        
#        f = open("triallist_stats.dat", "w")
#        tl = NP.array(self._trial_list)
#        symbols = tl[:,2]
#        targets = tl[:,0]
#        cues = tl[:,1]
#        f.write('CovertAttentionHex experiment on %s:\n' % str(datetime.now()))
#        f.write('time needed for experiment: approx. %d:%d (mm:ss)\n' % (NP.floor(self.stat_time_needed/1000/60.), NP.mod(self.stat_time_needed/1000, 60)))
#        f.write('distance to optimal fractions: %s%%\n' % str(100*sum(NP.abs(self.stat_differences))))
#        f.write('symbol = x: %d, (%.1f%%)\n' % (len(symbols[symbols==2]), len(symbols[symbols==2])/float(self.nr_trials)*100))
#        f.write('symbol = +: %d, (%.1f%%)\n' % (len(symbols[symbols==1]), len(symbols[symbols==1])/float(self.nr_trials)*100))
#        for i in xrange(6):
#            f.write('cue = %d: %d (%.1f%%)\n' % (i, len(cues[cues==i]), len(cues[cues==i])/float(self.nr_trials)*100))
#        for i in xrange(6):
#            f.write('target = %d: %d (%.1f%%)\n' % (i, len(targets[targets==i]), len(targets[targets==i])/float(self.nr_trials)*100))
#        f.write('\nfollowers_target:\n')
#        for [a,b,c,d,e,f_] in self.stat_followers_target:
#            f.write('[%d, %d, %d, %d, %d, %d]\n' % (a,b,c,d,e,f_))
#        f.write('\nfollowers_cue:\n')
#        for [a,b,c,d,e,f_] in self.stat_followers_cue:
#            f.write('[%d, %d, %d, %d, %d, %d]\n' % (a,b,c,d,e,f_))
#        f.write('\nsymbol locations:\n')
#        for [a,b,c,d,e,f_] in self.stat_symbol_locations:
#            f.write('[%d, %d, %d, %d, %d, %d]\n' % (a,b,c,d,e,f_))
#        f.write('\ntriallist:\n')
#        for [t, c, s, d] in self._trial_list:
#            f.write('[%d, %d, %d, %d]\n' % (t,c,s,d))
#        f.write('\n\n\n')
#        f.close()



if __name__ == "__main__":
    fb = CovertAttentionVE()
    fb.on_init()
    fb.on_play()
