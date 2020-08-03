# -*- coding: utf-8 -*-
# audio stimuli presentation module
# -- pyo layer for easy loading, presenting and modifing stimuli
#
# author: Konrad Krenzlin (krenzlin@mail.tu-berlin.de)
# last modified: 07/17/12
#
# !!! Important -- Don't forget to set your audio device(s) and buffersize !!!
#
#
# On Windows try ASIO4All driver (http://www.asio4all.com/)
# sometimes performance is even better than native ASIO driver
#
# For Mac or Linux:
# If you want JACK or CoreAudio connection, change the audio parameter in line...
#
#     Server(duplex=0, buffersize=BUFFERSIZE, audio='portaudio')
#
# to 'jack' or 'coreaudio'
#

# uses the pyo audio server
# see: http://code.google.com/p/pyo/
try:
    from pyo import *
except ImportError:
    print 'pyo not available! Please install!'
    raise ImportError
    exit()

import os
from time import sleep

## Set your system dependend parameters !!!
# BUFFERSIZE : set to lowest possible before crackling
# DEVICES : add your device(s) name(s), priority list
# CHANNELS : default is 2 (e.g. stereo)
#    panning [0..1] will be mapped that pan=0 -> first channel, pan=1 -> last channel
# PLAY_RATE : overall play rate of stimuli, for simple speeding up (should be used with caution, due to pitch shift)
BUFFERSIZE = 256
DEVICES = ['MOTU AUDIO ASIO', 'ASIO4ALL', 'ASIO', 'HDA']
CHANNELS = 2
PLAY_RATE = 1.05



class Audio:
    def __init__(self):
        # init pyo server
        self.s = Server(duplex=0, nchnls=CHANNELS, buffersize=BUFFERSIZE, audio='portaudio')

        # checking sound devices
        print '-- available sound devices --'
        devices = pa_get_output_devices()
        print

        device_id = None
        for device in DEVICES:
            for i,v in enumerate(devices[0]):
                if device in v.upper():
                    device_id = devices[1][i]
                    print 'detected device:', device
                    print v.upper(), device_id
                    break
            if device_id:
                break
        if device_id == None:
            device_id = pa_get_default_output()
            print 'using default device'

        device_index = devices[1].index(device_id)
        print ' -> (%d) %s' % (devices[1][device_index], devices[0][device_index])
        self.s.setOutputDevice(device_id)
        self.device = devices[0][device_index]

        # boot and start server
        self.s.boot()
        self.s.start()

        self.stimuli = {}

        print '-- pyo server running --'
        return
    

    def status(self):
        print '-- status --'
        print 'audio device:', self.device 
        print 'started:', self.s.getIsStarted()
        print 'buffersize:', self.s.getBufferSize()
        print '%d stimuli loaded' % (len(self.stimuli))


    def close(self):
        self.s.stop()
        sleep(0.5) # shutdown freezes if s.stop() isn't fully executed (last buffer is still played)
        self.s.shutdown()
        print
        print '-- pyo server stopped and shutdown --'


    def load_stimulus(self, stim_name, stim_file, stim_pan=0.5, stim_vol=1):
        path = os.path.dirname(__file__)
        data = SndTable(os.path.join(path, stim_file)) # loads file to table
        table = TableRead(table=data, freq=data.getRate() * PLAY_RATE) # makes it playable through TableRead object

        pan = SPan(input=table, pan=stim_pan, mul=stim_vol) # pan the stream, also set volume
        pan.out() # connect stream to output

        self.stimuli[stim_name] = {'stim': table, 'pan': pan} # store objects
        return


    def present_stimulus(self, stimulus):
        self.stimuli[stimulus]['stim'].play()


    def set_stimulus(self, stimulus, vol=None, pan=None, mul=False):
        if pan:
            self.stimuli[stimulus]['pan'].setPan(pan)
        if vol:
            if mul:
                vol *= self.stimuli[stimulus]['pan'].mul
            self.stimuli[stimulus]['pan'].setMul(vol)



if __name__ == "__main__":
    # testing module
    p = Audio()
    p.load_stimulus('a', 'stimuli/a.wav', 0)
    p.load_stimulus('les', 'stimuli/les.wav', 0, 0.7)
    p.load_stimulus('a', 'stimuli/a.wav', 0)
    p.present_stimulus('a')
    sleep(0.5)
    p.present_stimulus('les')
    sleep(0.5)
    p.set_stimulus('les', vol=0.5, pan=1)
    p.present_stimulus('les')
    sleep(0.5)
    p.set_stimulus('les', vol=0.3, pan=1)
    p.present_stimulus('les')
    sleep(0.5)
    p.set_stimulus('les', vol=0.2, pan=1)
    p.present_stimulus('les')
    sleep(0.5)
    p.status()
    p.close()
