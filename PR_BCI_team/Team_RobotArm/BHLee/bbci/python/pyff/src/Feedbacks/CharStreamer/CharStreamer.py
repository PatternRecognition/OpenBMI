# -*- coding: utf-8 -*-
#
# CharStreamer
# 
import time
import sys
import logging
import os
from numpy import mean, var, square, sqrt
from scipy.stats import distributions

import random

from FeedbackBase.Feedback import Feedback
import Queue
import pyolayer
import SequenceGenerator
from Events import __EVENTS__, __STIMULI__
import Events

# only for debug
import inspect
  
class CharStreamer(Feedback):
    
    def on_init(self):
        time.sleep(1) # needed for feedback to load correctly
        
        # init audio
        self.presentation = pyolayer.Audio()

        self.stream_volumes = Events.STREAMS_VOL
        if Events.STIM_VOL:
            self.single_volumes = Events.STIM_VOL

        self.logger.debug('load stimuli')
        for stimulus in __STIMULI__:
            self.logger.debug('%s' % (str(stimulus)))
            self.presentation.load_stimulus(*stimulus)
        self.queue = Queue.Queue()
        
        self.pre_iterations = 1
        self.iterations = 16
        self.SOA = 250
        self.random = False
        self.inter_trial_pause = 7

        self.pause_after_n_trials = 5
        
        self.condition = None

        self.online_mode = True
        self.wrote = []
        self.online_simulation = True
        self.calibration_mode = False
        self.target = random.choice(reduce(lambda x,y: x+y, Events.STREAMS_STIM))

        # for early stopping
        self.early_stopping = True
        self.n_for_iteration = len(__EVENTS__) * max([len(s) for s in __EVENTS__])
        # number of cl_responses for at least one iteration of all streams
        # we have to "wait" for the longest stream to be completed
        self.min_iterations = 6
        self.p_criterion = 0.001

        # logging
        self.file_log = ''
        self.__log_handler__ = None

        self.send_parallel(Events.INIT)
        print "Feedback successfully initiated."
        self.logger.info('INIT')

        # TESTING
        #self.on_interaction_event(['volume_calibration'])
        return


    def new_trial(self):
        print 'New Trial.'

        seq = []
        i = 0
        for s in SequenceGenerator.iterateX(self.pre_iterations + self.iterations, __EVENTS__, self.random):
            seq.extend(SequenceGenerator.paramTimer(s, minISI=self.SOA, maxISI=self.SOA, startTime=100 + i*(self.SOA/len(__EVENTS__))))
            i += 1
            
        self.sequence = SequenceGenerator.mix(seq)
        self.queue.setSequence(self.sequence)

        # reset feedback status
        self.stopping = False
        self.stopped = False
        self.pause = False
        self.trial_end = False

        self.n_presentations = 0
        self.n_iteration = 0
        self.n_targets = 0
        
        # reset the classification containers
        self.n_responses = 0
        self.cl_responses = {}
        self.p_decisions = {}
        self.p_fulfilled = False

        # only generating neat logging output
        log_str = 'NEW TRIAL ('
        if self.condition:
            log_str += 'cond. %d' % (self.condition)
        log_str += ', SOA=%d' % (self.SOA)
        if self.random:
            log_str += ', pseud-rand'
        if self.calibration_mode:
            log_str += ', calibration mode, target = %s' % (self.target)
        log_str += ', iterations = %d + %d = %d)' % (self.pre_iterations, self.iterations, self.pre_iterations + self.iterations)
        self.logger.info(log_str)
        return

    
    def on_quit(self):
        print "Now quitting."
        self.on_stop()
        
        # closing activities
        self.presentation.close()
        self.__log_handler__.close()
        self.logger.removeHandler(self.__log_handler__)

        print "Quitted."
        return


    def on_stop(self): # !! never call from main_loop
        print "Now stopping."
        self.stopping = True
        
        self.send_parallel(Events.STOP)

        print "Waiting for main loop to stop."
        while not self.stopped:
            pass

        print "Stopped."
        return

        
    def on_pause(self):
        self.pause = not self.pause
        return
        

    def on_play(self):
        self.pause = False
        self.stopping = False
        self.stopped = False
        
        time.sleep(1)
        
        # CALIBRATION MODE
        #
        if self.calibration_mode: # in calibration mode a cueing phase precedes
            if not self.target:
                print "Warning! Target must be set in calibration mode! Unless you're just demonstrating."
                self.target = random.choice(reduce(lambda x,y: x+y, Events.STREAMS_STIM))

            self.new_trial()

            self.logger.info("TARGET = %s / %s / %s" % (self.target, self.lookup(self.target), self.lookup(self.target) + Events.TARGET_OFFSET))

            # cueing phase
            for i in range(3):
                self.presentation.present_stimulus(self.target)
                self.send_parallel(self.lookup(self.target) + Events.CUE_OFFSET)
                time.sleep(1)
            time.sleep(2)
            
            # set target to marker (only for easier and faster detection and target offset calculation)  
            self.target = self.lookup(self.target)

            # start main loop               
            self.main_loop()
            
            if self.trial_end: # trial ended normally
                # end phase of calibration trial
                time.sleep(1.5)
                self.presentation.present_stimulus('end')
                
                self.n_targets = [stim for (typ, mrk, stim) in reduce(lambda x,y: x+y, [events for (clock, events) in self.sequence])].count(self.lookup(self.target))
                self.logger.info('Target %s was presented %d times.' % (self.lookup(self.target), self.n_targets))
                print 'Target %s was presented %d times.' % (self.lookup(self.target), self.n_targets)
                
            self.target = ''
            
            
        # ONLINE MODE
        #    
        elif self.online_mode: # in online mode, keep in loop until stopped through other parts (hopefully classifiers)
            trial_counter = 0
            while not self.stopping:
                print 'trial counter:', trial_counter
                # in online mode, pause after n trials
                trial_counter += 1
                if trial_counter >= self.pause_after_n_trials:
                    print 'automatically pausing.'
                    trial_counter = 0
                    self.send_parallel(Events.PAUSE_ON)
                    logging.info('pausing after %d trials' % (self.pause_after_n_trials))
                    self.presentation.present_stimulus('pause10')
                    time.sleep(5)
                    self.presentation.present_stimulus('pause5')
                    time.sleep(5)
                    self.presentation.present_stimulus('pause0')
                    print 'go on.'
                    time.sleep(1)
                    self.send_parallel(Events.PAUSE_OFF)

                # generate new trial
                self.new_trial()

                # signal for trial start
                self.presentation.present_stimulus('end')
                time.sleep(0.5)
                
                self.main_loop()
                if self.trial_end or self.p_fulfilled: # trial ended normally, get decision
                    decision = self.tell_decision()
                    if decision[0]: # otherwise no decision could be made
                        
                        print 'Detected:', decision[0]

                        # tell decision to subject
                        time.sleep(1)
                        self.presentation.present_stimulus('erkannt')
                        time.sleep(1)
                        self.presentation.present_stimulus(decision[0])
                        time.sleep(1)

                        # handling of the detected decision
                        # in this case only appending to wrote list
                        self.wrote.append(decision[0])

                        
                        self.logger.info('Written: %s' % ' '.join(self.wrote))
                        print 'Written:', ' '.join(self.wrote)


                
                if self.online_simulation: # don't loop when in simulation, so matlab can send new target
                    pass
                    #break
                
                time.sleep(self.inter_trial_pause)
        return
    

    def main_loop(self):
        print "Enter main loop."

        self.stopped = False

        self.send_parallel(Events.START_TRIAL)                
        self.queue.start()
        
        while not self.trial_end and not self.stopping and not self.p_fulfilled:
            if self.pause:
                continue
            self.tick()
        if self.trial_end or self.p_fulfilled:
            time.sleep(0.1) # damit letzter Marker nicht verdeckt wird
            self.send_parallel(Events.END_TRIAL)  
        elif self.stopping:
            self.send_parallel(Events.STOP)         

        print "Left main loop."
        self.stopping = False
        self.stopped = True
        return


    def tick(self):
        # get event(s) from queue for elapsed time and handle them
        for event in self.queue.tick():
            type, marker, stimulus = event

            if type == 'STIM':
                self.presentation.present_stimulus(stimulus)
                self.send_marker(marker)
                self.n_presentations += 1
                
                if self.online_simulation:
                    data = test_cl_out(self.lookup(stimulus), stimulus == self.target)
                    self.on_control_event(data)
                continue
            elif type == 'EOS':
                self.trial_end = True
                print '[END OF SEQUENCE]'
                
        return


    def lookup(self, obj):
        if type(obj) == str: # name
            return Events.NameLookup[obj]
        elif type(obj) == int: # marker
            return Events.MarkerLookup[obj]


    def send_marker(self, marker):
        self.logger.debug('MARKER %d from %s' % (marker, inspect.stack()[1][3]))
        if self.n_presentations <= self.pre_iterations * self.n_for_iteration: # still in pre/fake iterations phase
            self.send_parallel(marker + Events.PRE_PRES_OFFSET)
        elif self.calibration_mode and self.target == marker:
            self.send_parallel(marker + Events.TARGET_OFFSET)
        else:
            self.send_parallel(marker)

        
    def on_control_event(self, data):
        """
        handles control events from PYFF
        here, the data of the online Classification is 
        received      
        """

        if 'cl_output' in data: # classification output was sent
            self.logger.info('Received classifier output: %s' % (data))
            print 'Received classifier output: %s' % (data)
            cl_out = data['cl_output'][0]
            stim = int(data['cl_output'][1])  # is a marker not string
            self.manage_response(stim, cl_out)

        if 'user_count' in data: # user counted target presentations from matlab
            self.logger.info('Received user count: %s' % (data))
            count = data['user_count']
            self.logger.info('presented=%d, counted=%d, diff=%d' % (self.n_targets, count, self.n_targets - count))
            print 'presented=%d, counted=%d, diff=%d' % (self.n_targets, count, self.n_targets - count)
            
        if 'user_rating' in data:
            self.logger.info('Received user rating: %s' % (data))
            rating = data['user_rating']
            self.logger.info('target=%s, rated=%d' % (self.target, rating))
            print 'target=%s, rated=%d' % (self.target, rating)



    def on_interaction_event(self, data):
        if 'file_log' in data:
            print 'Logging to file %s.' % data['file_log']
            if self.__log_handler__:
                self.__log_handler__.close()
                self.logger.removeHandler(self.__log_handler__)
            self.__log_handler__ = logging.FileHandler(data['file_log'], mode='w')
            self.logger.addHandler(self.__log_handler__)

        if 'volume_calibration' in data:
            print 'Doing volume calibration of streams.'
            self.cal()
            self.logger.info('stream volumes set to: %s' % (self.stream_volumes))


    def cal(self):
        # reopen input, otherwise EOFError is raised for raw_input
        if sys.platform == "win32":
            sys.stdin = open("CON:")
        else:
            sys.stdin = open('/dev/tty')
            
        streams = __EVENTS__
        # ugly code for getting each stream against each other, but only once and shuffle them
        tmp = [(x,y) for x in range(len(streams)) for y in range(len(streams)) if x!=y]
        random.shuffle(tmp)
        tests = []
        for (x,y) in tmp:
            if (y,x) not in tests:
                tests.append((x,y))
        random.shuffle(tests)
        for index in tests:
            time.sleep(2)
            vol_set = False
            while not vol_set:
                for a,b in zip(streams[index[0]], streams[index[1]]):
                    self.presentation.present_stimulus(a[2])
                    time.sleep(0.25)
                    self.presentation.present_stimulus(b[2])
                    time.sleep(0.25)

                do_next = False
                while not do_next:
                    v = raw_input('volume of second is %s. Change? (+/-/<enter>/ok):' % (self.stream_volumes[index[1]]))
                    if v == '':
                        do_next = True
                    elif v == 'ok':
                        do_next = True
                        vol_set = True
                    elif v == '+':
                        self.stream_volumes[index[1]] += 0.1
                    elif v == '-':
                        self.stream_volumes[index[1]] -= 0.1
                for type, mark, stim in streams[index[1]]:
                    vol = self.stream_volumes[index[1]]
                    if stim in Events.STIM_VOL.keys():
                        vol *= Events.STIM_VOL[stim]
                    self.presentation.set_stimulus(stim, vol=vol) 
        return


    def manage_response(self, stim, response):
        """
        handles classifier outputs and checks if earlystopping criterum is fulfilled (if enabled)
        """
        self.n_responses += 1
        self.cl_responses.setdefault(stim, []).append(response)

        d, m = divmod(self.n_responses, self.n_for_iteration)
        # divmod(x, y) -> (quotient, remainder) -- quo*y + rem == x.
        # m == 0 --> exactly one or more iterations passed
        # d --> should be equal higher than the minimum iterations, but remember the non counting pre iterations
        if self.early_stopping and m == 0 and d - self.pre_iterations >= self.min_iterations:
            # print decision candidates
            print 'candiate stimuli:', [self.lookup(stim) for _, stim in sorted([(mean(self.cl_responses[stim]), stim) for stim in self.cl_responses])[:5]]
            
            # only test the best of each stream
            means = [(mean(self.cl_responses[stim]), stim) for stim in self.cl_responses]
            min_stims = []
            for stream in Events.STREAMS_STIM:
                this_stream_means = [(val, mrk) for (val, mrk) in means if self.lookup(mrk) in stream]
                min_stims.append(min(this_stream_means)[1])
            
            self.update_p_decisions(min_stims) 
            if min(self.p_decisions.values()) < self.p_criterion: 
                        self.p_fulfilled = True
                        self.logger.info('Early stopping criterion fulfilled after %d responses = %d iterations.' % (self.n_responses, self.n_responses / self.n_for_iteration))
  
  
    def update_p_decisions(self, stims):
        for stim in stims:
            selection = []
            for stream in Events.STREAMS_STIM:
                mrk = [self.lookup(s) for s in stream]
                if stim in mrk:
                    selection = mrk
                    break
            selection.remove(stim)

            print 't-test', stim, 'against', selection
            self.logger.info('t-test: %s against %s' % (stim, selection))

            rest = reduce(lambda x, y: x+y, [self.cl_responses[s] for s in selection if s != stim])
            pval2 = ttest2_p(self.cl_responses[stim], rest)
            self.p_decisions[stim] = pval2            
        print 'updated p', self.p_decisions


    def tell_decision(self):
        if not self.p_decisions:
            print 'No decision!'
            self.logger.info('No decision!')
            return('', 1)
        min_stim = min(self.p_decisions, key=self.p_decisions.get)
        min_p = self.p_decisions[min_stim]
        self.logger.info('Decision is: %s (%d) with p=%f' % (self.lookup(min_stim), min_stim, min_p))
        self.send_parallel(min_stim + Events.DECISION_OFFSET)
        return (self.lookup(min_stim), min_p)


def ttest2_p(v1, v2):
    # ttest with unequal variance, returns the pvalue
    # http://beheco.oxfordjournals. org/cgi/content/full/17/4/688
    n1 = len(v1)+0.
    n2 = len(v2)+0.
    mu1 = mean(v1)
    mu2 = mean(v2)
    s1 = var(v1)
    s2 = var(v2)
    
    t = (mu1-mu2)/sqrt(s1/n1 + s2/n2)
    u = s2/s1
    df = round(square(1/n1 + u/n2) / (1/(square(n1)*(n1-1)) + square(u) / (square(n2)*(n2-1)))) 

    pval = distributions.t.cdf(t,df)
    return(pval)


def test_cl_out(stim, target):
    #print '!! SIMULATION OF CLASSIFIER OUTPUT !!'
    m = 0
    s = 1
    dif = 2
    if target:
        print 'TARGET!!!'
        cl_out = random.normalvariate(m - dif ,s)
    else:
        cl_out = random.normalvariate(m, s)
    #print cl_out, stim
    return {'cl_output': (cl_out, stim)}
