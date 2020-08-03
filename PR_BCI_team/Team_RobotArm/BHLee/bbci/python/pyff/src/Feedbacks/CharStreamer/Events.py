import os
    
STREAMS_STIM = [['leer', 'paus'] + list('abcdefg'),
                list('hijklmnopq'),
                list('rstuvwxyz') + ['les', 'del']
                ]

STREAMS_PAN = [0,
               0.5,
               1,
               ]

STREAMS_VOL = [0.9,
               0.5,
               0.7,
               ]

STIM_FOLDER = 'stimuli'

## marker
START_TRIAL = 91
END_TRIAL = 92
EXP_END = 254
STOP = 93
INIT = 94
PAUSE_ON = 95
PAUSE_OFF = 96

MARKER_OFFSET = 1
PRE_PRES_OFFSET = 30
CUE_OFFSET = 60
TARGET_OFFSET = 100
DECISION_OFFSET = 160
# einzelne Stimuli in volume anpassen, sollte von der Form {'a':0.5, 'les':0, ...} sein
STIM_VOL = {'c': 0.8, 'paus': 1.5, 'del': 1.5}


###
# stimuli
###
path = os.path.join

__STIMULI__ = []
for i, stims in enumerate(STREAMS_STIM):
    pan = STREAMS_PAN[i]
    vol = STREAMS_VOL[i]
    for stim in stims:
        stim_file = path(STIM_FOLDER, stim + '.wav')

        if stim in STIM_VOL:
            vol = STREAMS_VOL[i] * STIM_VOL[stim]
        else:
            vol = STREAMS_VOL[i]
            
            
        __STIMULI__.append((stim, stim_file, pan, vol))


__STIMULI__.extend([('end', path(STIM_FOLDER, 'end.wav'), 0.5, 1),
                    ('erkannt', path(STIM_FOLDER, 'erkannt.wav'), 0.5, 1.3),
                    ('losche', path(STIM_FOLDER, 'losche.wav'), 0.5, 1.3),
                    ('pause10', path(STIM_FOLDER, 'pause10.wav'), 0.5, 1.0),
                    ('pause5', path(STIM_FOLDER, 'pause5.wav'), 0.5, 1.0),
                    ('pause0', path(STIM_FOLDER, 'pause0.wav'), 0.5, 1.0),
                    ])

##########
# events #
##########
__EOS__ = [('END', 99, None)]
__EVENTS__ = []

## marker lookup -> stim-name to marker and reverse
MarkerLookup = {}
NameLookup = {}
marker = MARKER_OFFSET
for stims in STREAMS_STIM:
    stream_events = []
    for stim in stims:
        event = ('STIM', marker, stim)
        stream_events.append(event)
        MarkerLookup[marker] = stim
        NameLookup[stim] = marker
        marker += 1
    __EVENTS__.append(stream_events)





if __name__ == "__main__":
    print __EVENTS__
    print
    print MarkerLookup
    print NameLookup
    
