# -*- coding: latin-1 -*-
import time, logging, os, sys, pygame
import logging.handlers, inspect
from numpy import random, mod, arange, array
from random import randint

import  SequenceGenerator
import TrialPresentation
from FeedbackBase.MainloopFeedback import MainloopFeedback


class Auditory_stimulus_screening(MainloopFeedback):
    
    def on_init(self):
        self.logger = logging.getLogger("FeedbackController")
        console = logging.StreamHandler()
        self.logger.addHandler(console)
        self.logger.setLevel(logging.DEBUG)
        self.logger.info('initialization!')
        
        self.fbPath = os.path.dirname(os.path.abspath(__file__)) #directory of THIS file 
        sys.path.append(self.fbPath)
        
        self.clock = pygame.time.Clock()
        
        #self.loadAllParameterFiles()
        self.loadParameterSet("StdParameters")
        self.loadParameterSet("paraStimulusScreening")

        #self.loadParameterSet("cond1")
        self.loadParameterSet("cond5")
        #self.loadParameterSet("cond2")
        
        
    def startOfTrial_tick(self):
        """
        starts a trial
        """
        # set target stimulus so that the deviant marker is sent!
        self.currentTargetStim = self.keysToSpell[0]

        # if condition demands, generate random timing schedule for stimuli
        if self.timed_trial:
            t_min = self.MIN_ISI
            t_max = self.MAX_ISI

            self.TrialSeq = []
            last = [(0,0), (0,0)]
            blocked = []
            elap = 0
            for stim in self.SequenceGenerator.seq:

                elap += randint(t_min, t_max)

                if last:
                    test = last.pop()
                    
                    if test[0] in [1,4,7]: blocked = [1,4,7]
                    elif test[0] in [2,5,8]: blocked = [2,5,8]
                    elif test[0] in [3,6,9]: blocked = [3,6,9]

                    if elap - test[1] < self.STIM_TIME and stim in blocked:
                        elap += self.STIM_TIME - (elap - test[1])
                        
                self.TrialSeq.append((stim, elap, elap + self.STIM_TIME))
                
                last = [(stim, elap)] + last


        self.send_parallel(int(self.BEFORE_CUE) )

        self.presentation.start_visual_cue(self.currentTargetStim)
        primingCountdown = self.numPriming            
        while (primingCountdown > 0):
            #first priming --> highlight the marker!
            self.send_parallel(int(self.CUE_SHIFT + self.currentTargetStim) )
            self.presentation.start_substim(self.currentTargetStim)
            pygame.time.delay(self.PRIMING_ISI)
            primingCountdown -= 1
        self.presentation.stop_visual_cue(self.currentTargetStim)

        self.send_parallel(int(self.AFTER_CUE) )

        self.startOfTrial = False
        self.send_parallel(int(self.START_TRIAL) )#send marker!

    
    def endOfTrial_tick(self):
        self.send_parallel (int(self.END_TRIAL))

        self.keysToSpell = self.keysToSpell[1:] # remove the first element
        
        # calibration mode
        print("\n \n ask4counts, correct was ", (self.SequenceGenerator.nRandomSeqBefore + self.SequenceGenerator.numMarkerSequences), "Target was ", self.currentTargetStim)
        self.presentation.printTextInHeader(u"Wie viele Präsentationen?")
        self.primingCountdown = self.numPriming # ??
        if not self.simulate_sbj:
            self.paused = True
        else:
            pygame.time.delay(6000)
             
        #set the system pasued and wait for the answer (unpause with on_interaction_event !!)
        self.startOfTrial = True

        
    def main_loop(self):
        self.logger.info(time.strftime('%x %X') + "--> start main loop")
        self.presentation.deleteTextInFooter()
        #self.presentation.printTextInFooter("geschriebener Text", " ")

        while 1:
            # break statement - normale case
            if len(self.keysToSpell) == 0:
                break
            
            self.initSequenceGenerator()              

            while (self.SequenceGenerator.keepTrialing() or self.paused) and (not self.stopping):
                self.tick()

            # break - stopped
            if self.stopped or self.stopping:
                break
                            
            
    def tick(self):
        """
        One tick of the main loop.
        
        Decides in which state the feedback currently is and calls the appropriate
        tick method.
        """
        
        self.presentation.manageEvents()

        if self.stopping:
            pass # go back to main_loop
        elif self.paused:
            #self.logger.info("paused")
            pygame.time.delay(20)
        elif self.endOfTrial:
            self.endOfTrial_tick()
        elif self.startOfTrial:
            self.startOfTrial_tick()
            pygame.time.delay(self.PRIMING_SPELLING_OFFSET)
            self.cummulatedTime = 0
            self.elapsed = self.clock.tick(self.FPS)
        else:
            # new with fps
            self.elapsed = self.clock.tick(self.FPS)
            self.cummulatedTime += self.elapsed

            if not self.timed_trial: 

                if self.cummulatedTime == self.elapsed and self.cummulatedTime>0:
                    # start a Subtrial!
                    # obtain next subtrial
                    self.currSubtrial =  self.SequenceGenerator.giveTrial()
                    self.send_marker(int(self.currSubtrial), marking=self.SequenceGenerator.sendMarker())
                    
                    self.presentation.start_substim(self.currSubtrial)
                    if not self.SequenceGenerator.keepTrialing():
                        # last Subtrial given
                        self.endOfTrial_tick()
                else:
                    if self.cummulatedTime >= self.STIM_TIME and ((self.cummulatedTime - self.elapsed) < self.STIM_TIME):
                        # end subtrial
                        self.presentation.stop_substim(self.currSubtrial)
                    elif (self.cummulatedTime >= self.ISI): #and ((self.cummulatedTime - self.elapsed) < self.ISI):
                        # end subtrial
                        self.cummulatedTime = 0
                        
            else: # use generated timing schedule
                if not self.SequenceGenerator.keepTrialing() or not self.TrialSeq:
                    self.endOfTrial_tick()
                else:    
                    if self.cummulatedTime >= self.TrialSeq[0][1]:
                        self.currSubtrial =  self.SequenceGenerator.giveTrial()
                        # the first few sequences are presented without markers
                        self.send_marker(int(self.currSubtrial), marking=self.SequenceGenerator.sendMarker())
                        self.presentation.start_substim(self.currSubtrial)
                        self.TrialSeq.remove(self.TrialSeq[0])
                    

                    
        
    def send_marker(self, istim, marking=True):
        self.logger.debug("send_marker ("+str(istim) +")!")
        if istim == self.currentTargetStim:
            istim += self.DEVIANT_SHIFT
        if not marking:
            istim += self.FIRSTSEQUENCES_SHIFT # beginning_shift
        self.send_parallel(int(istim))




      
    def on_interaction_event(self, data):
        """
        handles interaction events that are sent from e.g. Matlab 
        or in other ways over the PYFF Framework.
        Following interaction events are implemented:
         - enable spellerMode
         - receive the number of counts (calibration run) and 
           write them into the log-file and as marker (100+diff) 
        """
        self.logger.info("on_interaction_event, data" + str(data))
            
        if data.has_key(u'spellerMode'):
            if data[u'spellerMode']:
                self.logger.info(str(time.strftime('%x %X')) + "Speller Mode was activated")
                self.spellerMode = True
            else:
                self.logger.info(str(time.strftime('%x %X')) + "Speller Mode was deactivated")
                #self.spellerMode = False #maybe to debug
                
        elif data.has_key(u'numCounts') and self.paused:
            #waiting for the counts
            counts = data[u'numCounts']
            self.logger.info(time.strftime('%x %X') + ": target was " +str(self.currentTargetStim) + "subject counted " + str(counts) + " out of " + str(self.SequenceGenerator.nRandomSeqBefore + self.SequenceGenerator.numMarkerSequences))
            diff = int(counts - (self.SequenceGenerator.nRandomSeqBefore + self.SequenceGenerator.numMarkerSequences))
            self.numCounts = 0
            self.send_parallel(int(100+diff)) 
            self.presentation.deleteTextInHeader()
            
            if len(self.keysToSpell) == 0:
                #send the last marker and stopp the program 
                self.stopping = True
            self.presentation.deleteTextInHeader()
            self.paused = False
            #send 100 if counted number was correct 
            #and e.g. 98 if sbj counted 2 subtrials short!


        elif data.has_key(u'loadCond'):
            condition_file = 'cond' + str(data[u'loadCond'])
            self.logger.info(time.strftime('%x %X') + ": CONDITION " + str(condition_file))
            self.loadParameterSet(condition_file)
            print data[u'loadCond']
            print self.sounds
            print self.timed_trial
        return
    

    def on_control_event(self, data):
        return
         
        
    def on_quit(self):
        """ 
        quits the main loop 
        """
        self.logger.info(str(time.strftime('%x %X')) + "on_quit")
        self.stopping = True
        pygame.time.wait(100)
        while not self.stopped:
            pygame.time.wait(100)
        self.stopped = True
        
        
    def on_pause(self):
        """ 
        pause or unpause! 
        """
        if self.paused:
            self.send_parallel(int(self.PAUSE_STOPP))
            self.presentation.deleteTextInHeader()
        else: 
            self.send_parallel(int(self.PAUSE_START))
            self.presentation.printTextInHeader("Pause")
        self.paused = not self.paused

    
    def on_play(self):
        """
         starts the software:
         - loggers are set up
         - presentation class is instanciated
         - sbjsimulator is instanciated if necessary
         - DefaultParameters are loaded if (not self.manualParameters)
         - main_loop() is called (endless loop until program stops) 
         - pygame is closed
        """
        
        logHandler = logging.handlers.RotatingFileHandler(self.LOG_FILENAME, maxBytes=0)
        self.logger.addHandler(logHandler)
        self.logger.setLevel(logging.DEBUG)
        print "logger saving to ", self.LOG_FILENAME
        
        self.logger.info(time.strftime('%x %X') + "--> on_play")
        if self.adaptiveSequence and not self.spellerMode:
            self.logger.critical("adaptive Sequence has to be in the speller mode!")
        
        #some output in console and logfile for better usabilty
        ttmp = "PARAMETERS:\n spellerMode: "+ str(self.spellerMode)+ \
                         "\n simulate_sbj : "+ str(self.simulate_sbj )+ \
                         "\n keysToSpell: "+ str(self.keysToSpell)+ \
                         "\n MIN_MARKER_SEQ: " + str(self.MIN_MARKER_SEQ)+  \
                         "\n N_MARKER_SEQ: " + str(self.N_MARKER_SEQ)+ \
                         "\n MAX_NONMARKER_SEQ: " + str(self.MAX_NONMARKER_SEQ)+ \
                         "\n EARLY_STOPPING: " + str(int(self.EARLY_STOPPING))+ \
                         "\n DECISION_CRITERION: " + str(self.DECISION_CRITERION) +\
                          "\n ADAPTIVE_SEQUENCE: " + str(self.adaptiveSequence) + "\n \n" 
        self.logger.info(ttmp)
                        

        self.keysToSpell = random.permutation(self.keysToSpell)
        # Instanciate the trial-presentation (sound output) class 
        self.presentation = TrialPresentation.TrialPresentation(self.screenPos, self.sounds, self.templateTexts)

        self.send_parallel(int(self.START_EXP))
        pygame.time.delay(4000)
        self.main_loop()
        self.send_parallel(int(self.END_EXP))

        self.stopped = True
        pygame.quit()
        self.logger.info("pygame was closed")

        
    def get_variables(self):
        pass

    
    def loadParameterSet(self, fname):
        """
        loads all Variables which are specified in fname.py into the T9Speller class.
        Everything is overwritten without questioning!
        ###############################
        Example:
        ex1.py contains:
            ISI = 200
            dump = True
        
        within the T9Speller class, self.loadParameterSet('ex1') leads to
            self. ISI = 200
            self.dump = True
        """        
        self.logger.debug("load File " + fname)
        para = __import__(fname)

        tmp = inspect.getmembers(para)
        for i in range(len(tmp)):
            if tmp[i][0][:2] != '__':
                self.__setattr__(tmp[i][0], tmp[i][1])   
        
    
    def initSequenceGenerator(self):
        """
        initiatiate the SequenceGenerator (adaptive or standard)
        """

        #standard SequenceGenerator
        self.SequenceGenerator = SequenceGenerator.SequenceGenerator( \
            minDistance=self.MIN_DISTANCE, nstim=9, minnumMarkerSequences=self.MIN_MARKER_SEQ, \
            decisionCriterion = self.DECISION_CRITERION, numMarkerSequences=self.N_MARKER_SEQ, \
            maxNonmarkerSeq=self.MAX_NONMARKER_SEQ, earlyStopping =self.EARLY_STOPPING)
    
    
    def loadAllParameterFiles(self):
        flist = os.listdir(self.fbPath)
        paraList = ["StdParameters"]
        for f in flist:
            if (f[0:4] == "para") and (f[-3:] == ".py"):
                paraList.append(f[:-3])

          
        for f in paraList:
            self.loadParameterSet(f)
            self.logger.debug("Parameter File " + f + " loaded!")
            
        
if __name__ == "__main__":
    #import logging
    #import SubjectSimulator 
    #logging.basicConfig(level=logging.DEBUG)
    sim = Auditory_stimulus_screening()
    sim.on_init()
    sim.spellerMode = False
    sim.simulate_sbj  = False
    sim.logger.critical("This is a Testrun which was started on console and standard parameters!!!")
    sim.on_play()   
    
    
     
