# Presentation class
#

try:
    from pyo import *
except ImportError:
    print 'pyo not available! Please install!'
    exit()

import os
from Config import __BUFFERSIZE__, __DEVICE__
from time import sleep

PLAY_RATE = 1.05

class Presentation:
    def __init__(self):
        # init pyo server
        self.s = Server(duplex=0, buffersize=__BUFFERSIZE__, audio='portaudio')

        # checking sound devices
        print '-- available sound devices --'
        devices = pa_get_output_devices()
        print

        device_id = None
        if __DEVICE__ <> None: # device is set in config
            device_id = __DEVICE__
            print 'using specified device'
        else:
            # test for ASIO device
            for i,v in enumerate(devices[0]):
                if 'ASIO' in v.upper():
                    device_id = devices[1][i]
                    print 'detected ASIO device'
                    break
            if not device_id:
                device_id = pa_get_default_output()
                print 'using default device'

        device_index = devices[1].index(device_id)
        print ' -> (%d) %s' % (devices[1][device_index], devices[0][device_index])
        self.s.setOutputDevice(device_id)

        # boot and start server
        self.s.boot()
        self.s.start()

        self.stimuli = {}
        self.keep_alive = set() # container for pyo objects to keep them alive

        print '-- pyo server running --'
        return
    

    def status(self):
        print 'started:', self.s.getIsStarted()
        print 'verbosity:', self.s.verbosity
        print 'buffersize:', self.s.getBufferSize()
        # verbosity : Control the messages printed by the server. It is a sum of 
        # values to display different levels: 1 = error, 2 = message, 4 = warning , 8 = debug.


    def gui(self):
        self.s.gui(locals())

        
    def close(self):
        self.s.stop()
        sleep(0.5) # shutdown freezes if s.stop() isn't fully executed (last buffer is still played)
        self.s.shutdown()
        print
        print '-- pyo server stopped and shutdown --'


    def load_stimulus(self, stim_name, stim_file, stim_pan=0.5, stim_vol=1):
        path = os.path.dirname(__file__)
        data = SndTable(os.path.join(path, stim_file)) # loads file to table
        table = TableRead(table=data, freq=data.getRate() * PLAY_RATE, mul=stim_vol) # makes it playable through TableRead object
        pan = Pan(input=table, pan=stim_pan, spread=0.0) # pan the stream
        pan.out() # connect stream to output

        self.keep_alive.add(pan) # store pan object to keep it alive (needed!)
        self.stimuli[stim_name] = table # store
        return


    def present_stimulus(self, stimulus):
        #print 'Presentation:', stimulus
        self.stimuli[stimulus].play()


if __name__ == "__main__":
    # testing module
    p = Presentation()
    p.status()
    p.load_stimulus('a', 'stimuli/a.wav', 0)
    p.load_stimulus('les', 'stimuli/les.wav', 0)
    p.load_stimulus('a', 'stimuli/a.wav', 0)
    p.present_stimulus('a')
    sleep(0.5)
    p.present_stimulus('les')
    sleep(0.5)
    p.close()
