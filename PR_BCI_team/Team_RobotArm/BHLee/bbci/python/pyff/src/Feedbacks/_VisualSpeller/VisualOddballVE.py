
# Copyright (C) 2011  Simon Scholler
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


# Q::
# - yield (marker and response) - big problem, needs to be solved!
# - pyff('setint','geometry',VP_SCREEN) not working. how can view-settings be changed?
# 
#
#
#


"""Class for Visual Oddball Experiments using Vision Egg for Stimulus Presentation."""

import pygame
#import random
import os
import sys

from scipy import *
import time
import VisionEgg
import random

from VisionEgg.Textures import Texture
from VisionEgg.MoreStimuli import Target2D, FilledCircle
from VEShapes import FilledCross, FilledTriangle, FilledHourglass

from FeedbackBase.VisionEggFeedback import VisionEggFeedback
    
from lib import marker
    
# TODO: 
# - EVTL: hit-miss-counter

# Q:
# - color word  (as in self._view.center_word)  
# - problems with yield (class variables do not get updated!)
# - standard and deviant marker codes: evtl. include in marker.py (ask benjamin, basti)
    
    
class VisualOddballVE(VisionEggFeedback):    
    
    #STANDARD, DEVIANT = list(),list()    
    STANDARD, DEVIANT = 10, 20
    STIM_OFF = 100
    # standards have markers 10,11,12,... ; deviants 30,31,32,... (cf. get_stimuli())
    # if self.group_stim_markers==True, then there exist only two markers, one for
    # group standard (10), and one for group deviant (20)
    #RESP_STD, RESP_DEV = 1,2
    #SHORTPAUSE_START, SHORTPAUSE_END = 249, 250
    
    
    def init_parameters(self):
        
        self.geometry = [0, 0, 1280, 1024]
        self.nTrials = 200
        self.nTrials_per_block = 50
        self.dev_perc = 0.25        
        self.countdown_start = 5
        self.show_standards = True
        self.show_fixation_cross = False
        self.duration_break = 3
        self.give_feedback = False   # will be ignored if self.response=='none'
        self.group_stim_markers = False
        self.dd_dist = 0    # no constraint if deviant-deviant distance is 0 (cf. constraint_stim_sequence() )            
        
        self.DIR_DEV = 'C:\img_oddball\dev'
        self.DIR_STD= 'C:\img_oddball\std'
        self.logfilename = ''
#        stimuli_opts = ['load', 'predefined']
#        self.stimuli = stimuli_opts[1]
        
        # response options        
        self.eob_response = True     # type in e.g. number of deviants after each block?   
        response_opts = ['none', 'dev_only', 'both'] # none: subject should not press a key
                                                     # dev_only: subject should press only for deviants
                                                     # both: subject response for both stds and devs
        self.response = response_opts[0]
        self.rsp_key_dev = 'f'
        self.rsp_key_std = 'j'        
        self.hitstr = 'Hit'
        self.falsestr = 'False'
        self.norespstr = 'Too Late'        


        # Durations of the different blocks
        self.prestim_ival = .8
        #self.fixation_cross_time = self.prestim_ival
        self.stim_duration = .4
        self.responsetime_duration = 0.
        self.feedback_duration = 1.     
        
        self.VEstimuli = True            
        self.bg_color = 'grey'
        
        self.error_check()
        
        
        
    def run(self):
        
        self.center = [self.geometry[2]/2, self.geometry[3]/2]
        if self.VEstimuli:
            self.std, self.dev = self.define_stimuli() 
        else:
            self.std, self.dev = self.load_stimuli()   
        self.create_log() 
        self.create_stim_seq(self.nTrials)
            
        # Init image and text object
        self.image = self.add_image_stimulus(position=tuple(self.center), on=False)
        self.text = self.add_text_stimulus(position=tuple(self.center),on=False)

        # This feedback uses a generator function for controlling the stimulus
        # transition. Note that the function has to be called before passing
        generator = self.prepare()
        
        # Pass the transition function and the pre-stimulus display durations
        # to the stimulus sequence factory. 
        if self.give_feedback:
            s = self.stimulus_sequence(generator, [
                    self.prestim_ival,
                    self.stim_duration,
                    self.responsetime_duration,
                    self.feedback_duration
                ], pre_stimulus_function=self.trigger_marker)                
        else:
            s = self.stimulus_sequence(generator, [
                    self.prestim_ival,
                    self.stim_duration,
                    self.responsetime_duration,
                ], pre_stimulus_function=self.trigger_marker)            
        
        # Start the stimulus sequence
        self._view.countdown()
        s.run()                    
        
        # Close logfile and exit the feedback main loop
        self.close_log()
        self.quit()
        

    def prepare(self):
        """ This is a generator function. It is the same as a loop, but
        execution is always suspended at a yield statement. The argument
        of yield acts as the next iterator element. In this case, none
        is needed, as we only want to prepare the next stimulus and use
        yield to signal that we are finished.
        """      
        for n, stim in enumerate(self.stim_pres_seq):                      

            self.next_trigger = ''
            # Init Trial
            self.responded = False
            if self.stim_seq[n] == 1:
                self.isdeviant = True
            else:
                self.isdeviant = False            

            self._view.update()          
                
            # Prestimulus interval            
            if self.show_fixation_cross:
                self.next_trigger = 'FIXATION_START'
                self.text.set(text='+',on=True)            
            yield
            if self.show_fixation_cross:
                self.next_trigger = 'FIXATION_END'
                self.text.set(text='',on=False)
                #self._view.update()          

            # Present Stimulus            
            self.response = True
            if self.isdeviant:
                self.next_trigger = 'DEVIANT'
            else:
                self.next_trigger = 'STANDARD'
            if self.show_standards or self.isdeviant: 
                if self.VEstimuli:       
                    self.set_stimuli(stim)
                else:            
                    self.image.set_file(stim)#, texture=VisionEgg.Textures.Texture(stim))
                    self.image.set(on=True)
            yield  
            
            # Wait for Response
            self.next_trigger = 'STIM_OFF'
            if self.VEstimuli:
                self.set_stimuli(self.text)
            else:
                self.image.set(on=False)
            yield

            self.response = False
            
            # Give Feedback about Response
            if self.give_feedback:
                #self._trigger(marker.FEEDBACK_START)
                if not self.responded:
                    self.text.set(text=self.norespstr,on=True)
                else:
                    self.text.set(on=True)
                yield
                self.text.set(on=False)
            self._view.update()

            if (n+1) % self.nTrials_per_block == 0:
                # User Input
                self._flag.toggle_suspension()
                time.sleep(1)
                if self.eob_response:
                    self.user_input = ''
                    self.eob_responded = False  
                    self.text.set(text='Type in #Deviants',on=True)
                    while not self.eob_responded:   # while no end_of_block response
                        self._view.update()           
                    self.text.set(text='',on=False)
                    self.write_log(self.user_input+'\n')
                    block_no= n/self.nTrials_per_block
                    nTargets= sum(self.stim_seq[self.nTrials_per_block*block_no:self.nTrials_per_block*(block_no+1)-1])
                    print "Block " + str(block_no+1) + " User input: " + self.user_input + " (correct: " + str(nTargets) + ")"

                self.image.set(on=False)
                if n == self.nTrials-1:
                    self._view.present_center_word('End of Session', 5)
                else:
                    self._view.present_center_word('Short Break', self.duration_break)
                    self._view.countdown()
                self._flag.toggle_suspension()
    
    
    def create_stim_seq(self,nTrials=None):
        if nTrials == None:
            nTrials = self.nTrials_per_block
            
        self.stim_seq = self.constrained_stim_sequence(nTrials,self.dev_perc,self.dd_dist)
        # convert stimuls sequence to a stimulus sequence of image filenames
        self.stim_pres_seq = list()        
        for n in range(nTrials):
            if self.stim_seq[n]==0:
                self.stim_pres_seq.append(self.std[random.randint(0,len(self.std)-1)])
            else:  
                self.stim_pres_seq.append(self.dev[random.randint(0,len(self.dev)-1)])  
    
    
    def keyboard_input(self, event):
        """ Handle pygame events like keyboard input. """
        quit_keys = [pygame.K_q, pygame.K_ESCAPE]
        number_keys = [pygame.K_0, pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5, pygame.K_6, pygame.K_7, pygame.K_8, pygame.K_9]
        ### QUIT PROGRAM ###
        if event.key in quit_keys or event.type == pygame.QUIT:
            self.quit()
        ####################
        ### END-OF-BLOCK RESPONSE ###
        if self.eob_response:
            if event.key in number_keys:            
                self.user_input += str(event.unicode)
                self.text.set(text=self.user_input,on=True)
            elif event.key == pygame.K_BACKSPACE:
                self.user_input = self.user_input[0:-1]
                self.text.set(text=self.user_input,on=True)
            elif event.key == pygame.K_RETURN:
                self.eob_responded = True
        #############################
        ### RESPONSE AFTER STIMULUS PRESENTATION ###
        elif event.type == pygame.KEYDOWN and self.response!='none' and self.response:
            if not self.responded:
                if event.unicode == unicode(self.rsp_key_dev):
                    #self.send_parallel(self.RESP_DEV)
                    self.responded = True
                    if self.isdeviant:
                        self.text.set(text=self.hitstr,on=False)
                    else:
                        self.text.set(text=self.falsestr,on=False)
                elif event.unicode == unicode(self.rsp_key_std) and self.response=='both':
                    #self.send_parallel(self.RESP_STD)
                    self.responded = True
                    if self.isdeviant:
                        self.text.set(text=self.falsestr,on=False)
                    else:
                        self.text.set(text=self.hitstr,on=False)
        ############################################
                    
    def load_stimuli(self):
        """
        Loads standard and deviant stimuli from folders.
        """        
        sd = os.listdir(self.DIR_STD)         
        dv = os.listdir(self.DIR_DEV)
        std = []
        dev = []
        
        for s in range(len(sd)):
            fn, ext = sd[s].split('.')
            if ext == ('jpg' or 'jpeg' or 'png'):
                std.append(self.DIR_STD + '/' + sd[s])        
        for d in range(len(dv)):
            fn, ext = dv[d].split('.')
            if ext == ('jpg' or 'jpeg' or 'png'):
                dev.append(self.DIR_DEV + '/' + dv[d])

        print std
        print dev
        return std, dev
        
        
    def define_stimuli(self):
        """
        Creates standard and deviant stimuli.          
        """        
        # The stimuli. This can be anything compatible with VisionEgg
        #dev = FilledCircle(color=(0,1.0,0), position=self.center, radius=100)
        #std1 = Target2D(color=(0,0,1.0), position=self.center, size=(200,200))
        #std2 = Target2D(color=(0,0,1.0), position=self.center, size=(200,200))
        #dev = Text(text='Deviant', font_size=72, position=(300, 200), anchor='center')
        #std1 = Text(text='Standard I', font_size=72, position=(300, 200), anchor='center')
        #std2 = Text(text='Standard II', font_size=72, position=(300, 200), anchor='center')        
        #dev = FilledTriangle(size=200, color=(0.0, 0.0, 1.0), innerSize=60, innerColor=(.745,.745,.745),
        #                     watermark_size=80, watermark_color=(.0,.0,.0,.15), position=self.center)
        #std1 = FilledTriangle(size=200, color=(0.0, 0.0, 1.0), innerSize=60,  innerColor=(.745,.745,.745), position=self.center)
        #std2 = FilledCross(size=(30., 180.), orientation=45., color=(0.0, 1.0, 1.0), innerColor=(.745,.745,.745), position=self.center)
        #return [std1, std2], [dev]
        #dev = FilledTriangle(size=200, color=(1.0, 0.0, 0.0), innerSize=60,  innerColor=(.745,.745,.745), position=self.center)
        #dev = FilledHourglass(color= (0.86, 0.0, 0.86), position=self.center, orientation=90, size=130)
        #std = FilledCircle(color=(0,0.7,0), position=self.center, radius=100)
        std = FilledHourglass(color= (0.0, 0.53, 0.0), position=self.center, orientation=0, size=100)
        dev = FilledCross(size=(30., 180.), orientation=45., color=(1.0, 0.0, 0.0), innerColor=(.745,.745,.745), position=self.center)
        return [std], [dev]
        
        
    def constrained_stim_sequence(self, N, dev_perc, dd_dist=0):    
        """
        N:          total number of trials
        perc_dev:   percentage of deviants
        dd_dist:    constraint variable: minimal number of standards between two deviants 
                   (default: no constraint (=0))
        """    
        devs_to_be_placed= round(N*dev_perc)
        if devs_to_be_placed+(devs_to_be_placed-1)*dd_dist>N:
            solve_prob = 'Increase the number of trials, or decrease the percentage of deviants or the minimal dev-to-dev-distance.' 
            raise Exception('Oddball sequence constraints cannot be fulfilled. ' + solve_prob)
        
        devs = -1
        while round(N*dev_perc) != devs:
            sequence= zeros((N,1));
            devs_to_be_placed= round(N*dev_perc)
            ptr = 0
            while ptr < N:        
                togo= N-ptr
                prob= devs_to_be_placed/floor(togo/2)        
                if random.random() < prob:
                    sequence[ptr] = 1
                    devs_to_be_placed = devs_to_be_placed - 1
                    ptr = ptr + dd_dist
                ptr= ptr +  1
            devs = sum(sequence)
        #print sequence
        return sequence
    
    
    def stim_sequence(self, nStim, stim_perc):
        """ 
        Creates a randomly shuffled list with numbers ranging from 0-(nStim-1)
        The percentages of the numbers occurring are given by the list stim_perc
        """
        list = [];
        for n in range(len(stim_perc)):
            list.extend([n] * int(nStim*stim_perc[n]))
        random.shuffle(list)
        return list
        
        
    def error_check(self):
        """
        Check the settings for errors
        """
        if not self.show_standards:
            if self.response != 'dev_only':
                self.response = 'dev_only'
                warnings.warn('Standards not shown. Response option will be changed to ''dev_only''')
        if self.response == 'none':
            self.give_feedback = False  
            self.feedback_duration = 0    
        
    def create_log(self):
        """
        Creates a logfile for the userinputs.
        """
        if not self.logfilename=='': 
            self.logfile = open(self.logfilename,'w')
        
    def write_log(self, logstr):
        """
        Appends logstr to logfile.
        """
        if not self.logfilename=='':
            self.logfile.write(logstr)
    
    def close_log(self):
        """
        Close the logfile.
        """
        if not self.logfilename=='':
            self.logfile.close()

    def trigger_marker(self):
        """
        Triggers the respective markers
        """
        if self.next_trigger == "FIXATION_START":
            self._trigger(marker.FIXATION_START, wait=True)
        elif self.next_trigger == "FIXATION_END":
            self._trigger(marker.FIXATION_END, wait=True)
        elif self.next_trigger == "DEVIANT":
            self._trigger(self.DEVIANT, wait=False)
        elif self.next_trigger == "STANDARD":
            self._trigger(self.STANDARD, wait=False)
        elif self.next_trigger == "STIM_OFF":
            self._trigger(self.STIM_OFF, wait=False)
        self.next_trigger= ''
        
        
if __name__ == '__main__':
    vo = VisualOddballVE()
    #vo.__init__()
    vo.pre_mainloop()
    vo._mainloop()    
    #vo.run()
