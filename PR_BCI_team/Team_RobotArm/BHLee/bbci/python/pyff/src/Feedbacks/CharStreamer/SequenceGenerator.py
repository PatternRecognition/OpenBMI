# SequenceGenerator
#
# this should be the logic behind a feedback
# this class should generate the experiment depending sequences, randomized, synchron/asynchron whatever
# the generated sequence must be in the form that Queue understands
# the emitted events have to be understood by Main and Presentation
#
# Sequences have to be in the form of [(time of event, [event(s)]), ...]
#
# common way is to generate the apropriate sequence of events (your function), then add
# the timing with paramTimer
# if you have several streams of events going on you can mix them with the mix function
import random
import Events
from copy import deepcopy


def iterateX(iterations, streams, random=False):
    max_len = max([len(s) for s in streams])
    n_stim = max_len * iterations
    
    result = []
    for stream in streams:
        it = (n_stim / len(stream) + 1)
        if random:
            tmp = random_repeat(stream, it)
        else:
            tmp = stream * it
        tmp = tmp[:n_stim]
        result.append(tmp)
    return result


def random_repeat(lst, N, min_distance=2):
    seq = []
    nseq = 0

    while nseq < N:
        subseq = deepcopy(lst)
        tmpseq = []
            
        while subseq:
            blocked = (seq + tmpseq)[-min_distance:]
            unblocked = list(set(subseq) - set(blocked)) # remaining pool of stimuli

            if not unblocked: # all remaining stimuli are blocked; restart current subseq
                subseq = deepcopy(lst)
                tmpseq = []
                continue
                
            s = random.choice(unblocked)
            tmpseq.append(s)
            subseq.remove(s)

        seq.extend(tmpseq)
        nseq += 1
        
    return seq


def paramTimer(events, minISI, maxISI, startTime=0):
    seq = []
    time = startTime

    for event in events:
        seq.append((time, [event]))
        time += random.randrange(minISI, maxISI+1) # if min == max => fixed time interval

    return seq


def mix(*args):
    seq = reduce(lambda x,y: x+y, args)
    tmp = {}
    for k,v in seq:
        if type(v) == list:
            tmp.setdefault(k, []).extend(v)
        else:
            tmp.setdefault(k, []).append(v)
    return sorted(zip(tmp.keys(), tmp.values()))

    
if __name__ == "__main__":
    a = list('abcde')
    b = list('fghijk')
    c = list('lmnopqr')

    test = iterateX(3, [a, b, c], random=False)
    print test
    print reduce(lambda x,y: x+y, test).count('o')
    print
