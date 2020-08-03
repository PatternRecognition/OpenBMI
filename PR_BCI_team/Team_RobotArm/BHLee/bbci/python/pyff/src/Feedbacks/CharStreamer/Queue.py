# class Queue
# a event timer (scheduler)
#   every time tick with the current time stamp is called Queue looks if a event in the sequence has to be emitted
#   handles a sequence of the form [(time1, event), (time2, event), ...
#   caller function sequence generator must provide the correct time format (e.g. milliseconds)
#   returns event which then has do be handled by other function
#   Queue also can be reseted if it has to
#
# TODO: in tick. if there is another event in the queue which is overdue
#   e.g. seq = [(1000, 1), (1010, 2)] and tick(1020) is called then only the first is emitted
# TODO: try heapq
from copy import deepcopy
import sys
import time

RESOLUTION = 1000

if sys.platform == "win32":
    # On Windows, the best timer is time.clock()
    default_timer = time.clock
else:
    # On most other platforms, the best timer is time.time()
    default_timer = time.time


class Queue:

    def __init__(self, sequence=None):
        self.sequence = None
        self.currentSeq = None
        if sequence:
            self.setSequence(sequence)


    def setSequence(self, sequence):
        self.sequence = deepcopy(sequence)
        self.sequence.reverse()
        self.currentSeq = deepcopy(self.sequence)


    def getSequence(self):
        return self.sequence


    def time(self):
        return RESOLUTION * default_timer()


    def elapsedTime(self):
        return self.time() - self.startTime


    def tick(self):
        if self.currentSeq and self.currentSeq[-1][0] <= self.elapsedTime():
            return self.currentSeq.pop()[1]
        if self.currentSeq == []: # reached end of sequence
            return [('EOS', None, None), ]
        return []


    def start(self):
        self.startTime = self.time()


    def reset(self):
        self.currentSeq = deepcopy(self.sequence)
        
    
        
    
