#!/usr/bin/env python


import random


RUNTIME = 45. * 60
# (amount, symbol, avg time per)
RED = 100, 1, 15.0
BLUE = 100, 2, 1.0
GREEN = 100, 3, 0.0
GREY = 0, 4, 0.0

AVAILABLE_TIME = RUNTIME - (RED[0] * RED[2] 
                            + BLUE[0] * BLUE[2]
                            + GREEN[0] * GREEN[2]
                            + GREY[0] * GREY[2])
TRIALS = RED[0] + BLUE[0] + GREEN[0] + GREY[0]

MAX_BURSTLEN = 3
MAX_REPEATS = 3

def get_random_delay():
    return random.randint(0, int(2 * AVAILABLE_TIME / TRIALS))

def main():
    stimulipool = []
    for color in RED, BLUE, GREEN, GREY:
        for i in range(color[0]): 
            stimulipool.append(color[1])
    random.shuffle(stimulipool)
    value, count = 0, 0
    stimuli = []
    while True:
        if len(stimulipool) == 0:
            break
        different_colors = 0
        for col in RED, BLUE, GREEN, GREY:
            if stimulipool.count(col[1]) > 0:
                different_colors += 1
        if different_colors <= 1:
            stimuli.extend(stimulipool)
            break

        stim = stimulipool.pop(0)
        if stim == value:
            count += 1
        else:
            count = 1
            value = stim
        if count <= MAX_REPEATS:
            stimuli.append(stim)
        else:
            stimulipool.append(stim)
    # generate the delays 
    times = [get_random_delay() for i in range(TRIALS)]
    # test max burstlen
    burst = 0
    for i in range(len(times)):
        if times[i] == 0:
            burst += 1
        else:
            burst = 0
        if burst > MAX_BURSTLEN:
            times[i] = 1
    # print it
    for time, klass in zip(times, stimuli):
        print time, klass








if __name__ == "__main__":
    main()

