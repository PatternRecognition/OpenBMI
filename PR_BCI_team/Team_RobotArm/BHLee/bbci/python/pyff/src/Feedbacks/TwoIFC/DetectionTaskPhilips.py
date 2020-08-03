
import pygame
import os, time
from time import sleep
from random import shuffle


from lib.pyffserial import pyffserial
import sys, traceback

#from FeedbackBase.MainloopFeedback import MainloopFeedback
#from MainloopFeedback import MainloopFeedback
from FeedbackBase.MainloopFeedback import MainloopFeedback
from lib.ExperimentalDesign.OrthogonalDesign import orthogonalDesign


class DetectionTaskPhilips(MainloopFeedback):
    
    # Triggers
    RUN_START, RUN_END = 252, 253
    TRIAL_START = 248
    #CW_ON = 234   #  1000 Hz CW
    CW_ON = 218    # 500 Hz CW
    #NONTARGET,TARGET = 8,9
    #RIGHT,WRONG = 6,7
    # Light source
    LIGHT_OFF = 176
    READ_VALUE = 174
    
    def init(self):

        # Duty cycle  (% light on) und  freq (flicker frequenz) setzen
        self.freq = (26,42,58,74,90,106,122,138,154,170,186,202)  # 40-150 Hz in steps of 10 Hz
        
        self.duty = (0,0)  # 50% duty cycle

        # Helligkeit (12 bits) besteht aus 2 bytes!
        # Erstes byte: obere 4 bits, zweites byte untere 8 bits
        self.brightness_cw1 = 11      # upper 4 bits
        self.brightness_cw2 = 180      # lower 8 bits
        #self.brightness1 = (20,20,20,20) # upper 4 bits
        #self.brightness2 = (40,40,40,40) # lower 8 bits
        self.brightness1 = 11 # upper 4 bits
        self.brightness2 = 180 # lower 8 bits
        self.nTrialsPerCondition = 2

        # Timing
        self.tPreTrial = .8
        self.tPreStim = .2
        self.tStim = 2
        self.tBetweenStim = 2
        self.tAfterTrial = 3
        self.tPreExperiment = 4
        #self.tStim = 250 		# timing of stimulus
        #self.tFixation = 400 		# timing of fixation cross
        #self.tBlankAfterFix = 100 	# blank screen between fixation cross and stimulus
        #self.tBlankAfterStim = 100 	# blank screen after stimulus
        #self.tITI = 2000  		# intertrial interval (when selfpaced = 0)
        self.task = '2IFC'   # can be 'yes-no' or '2IFC'
        self.propCW = .5;   # Proportion of CW trials (for yes/no task)
        self.doShuffle = 1   # Shuffling of trials enabled or disabled


    def pre_mainloop(self):

        self.nTrials = len(self.freq)*self.nTrialsPerCondition    # number of trials per condition(!!)
        self.send_parallel(self.RUN_START)
        self.serial_port = pyffserial()

        if self.task == '2IFC':
            self.targetIval = range(2)    # 0= target in first ival, 1=target in 2nd ival
            # Specify design
            self.nTrials = len(self.freq)*self.nTrialsPerCondition
            self.trials = orthogonalDesign([self.freq,self.targetIval],self.nTrials)
        else:                   # yes/no task
            # Calculate number of CW trials
            lf = len(self.freq)
            ncw = int(round((self.propCW*lf)/(1.- self.propCW)))
            # Append flicker to self.freq
            self.freq.extend([self.CW_ON]*ncw)
            self.nTrials = len(self.freq)*self.nTrialsPerCondition
            self.trials = orthogonalDesign([self.freq],self.nTrials)
            
        self.currentTrial = 0
        if self.doShuffle:
            shuffle(self.trials)
        print self.trials

        #self.state_stim = True
        self.state_response = False
        self.state_iti = False
        
        self.currentTrial = 0
        
        # Just to be sure: Set light off
        self.serial_port.send_serial(self.LIGHT_OFF)
        sleep(self.tPreExperiment)


    def tick(self):
        pass            

    def play_tick(self):

        if self.task == '2IFC':
            self.play_2IFC()
        elif self.task == 'yes-no':
            self.play_yesno()
        else:
            print "UNKNOWN TASK ",self.task
            
    def play_yesno(self):
        # Plays a trial in yes-no design
        cfreq  = self.trials[self.currentTrial][0]
        self.send_parallel(self.TRIAL_START)
        
        # Wait pre-trial time
        sleep(self.tPreTrial)

        # Wait pre-stimulus #1 time
        sleep(self.tPreStim)

        # Prepare stimulus and show
        isCWTrial = ( cfreq == self.CW_ON )    # cw light or flicker?

        self.serial_port.send_serial(self.READ_VALUE)
        if isCWTrial:
            self.serial_port.send_serial(self.brightness_cw1)
            self.serial_port.send_serial(self.brightness_cw2)
            self.serial_port.send_serial(self.CW_ON)
            self.send_parallel(self.CW_ON)
            print ">>> CW"
        else:
            self.serial_port.send_serial(self.brightness1)
            self.serial_port.send_serial(self.brightness2)
            self.serial_port.send_serial(cfreq)
            self.send_parallel(cfreq)

        sleep(self.tStim)
        self.waiting = 1

        # Wait after trial time
        self.serial_port.send_serial(self.LIGHT_OFF)
        self.send_parallel(self.LIGHT_OFF)
        #sleep(self.tAfterTrial)
        
        # Inc trial count
        self.currentTrial += 1
        if self.currentTrial == self.nTrials:
            self.on_stop()
            sleep(1)

        while self.waiting == 1:
            sleep(.5)

        
    def play_2IFC(self):
        cfreq,target  = self.trials[self.currentTrial]
        self.send_parallel(self.TRIAL_START)
        
        # Wait pre-trial time
        sleep(self.tPreTrial)
       
        # Wait pre-stimulus #1 time
        sleep(self.tPreStim)

        # Prepare first interval and show
        #self.serial_port.send_serial("Serial #1: "+" Duty="+dutyStr+ " Freq=" +freqStr)
        self.serial_port.send_serial(self.READ_VALUE)
        if target==0:
            self.serial_port.send_serial(self.brightness1)
            self.serial_port.send_serial(self.brightness2)
            self.serial_port.send_serial(cfreq)
            #self.send_parallel(self.TARGET)
            self.send_parallel(cfreq)
        else:
            self.serial_port.send_serial(self.brightness_cw1)
            self.serial_port.send_serial(self.brightness_cw2)
            self.serial_port.send_serial(self.CW_ON)
            #self.send_parallel(self.NONTARGET)
            self.send_parallel(self.CW_ON)

        #print "Serial port: ",self.serial_port.read_serial()
        sleep(self.tStim)
        
            
        # Wait between-stimulus time and prepare second interval
        self.serial_port.send_serial(self.LIGHT_OFF)
        self.send_parallel(self.LIGHT_OFF)
        sleep(self.tBetweenStim)


        # Second stimulus 
        self.serial_port.send_serial(self.READ_VALUE)
        if target==1:
            self.serial_port.send_serial(self.brightness1)
            self.serial_port.send_serial(self.brightness2)
            self.serial_port.send_serial(cfreq)
            #self.send_parallel(self.TARGET)
            self.send_parallel(cfreq)           
        else:
            self.serial_port.send_serial(self.brightness_cw1)
            self.serial_port.send_serial(self.brightness_cw2)
            self.serial_port.send_serial(self.CW_ON)
            #self.send_parallel(self.NONTARGET)
            self.send_parallel(self.CW_ON)            
        #print "Serial port: ",self.serial_port.read_serial()
        sleep(self.tStim)
        self.waiting = 1

        # Wait after trial time
        self.serial_port.send_serial(self.LIGHT_OFF)
        self.send_parallel(self.LIGHT_OFF)
        sleep(self.tAfterTrial)
        
        # Inc trial count
        self.currentTrial += 1
        if self.currentTrial == self.nTrials:
            self.on_stop()
            sleep(1)

        while self.waiting == 1:
            sleep(.5)
            
    def post_mainloop(self):
        self.send_parallel(self.RUN_END)
        self.serial_port.close()
        pygame.quit()
        print "Finished"

       

if __name__ == "__main__":
    feedback = TwoIFCPhillips()
    feedback.on_init()
    feedback.on_play()

