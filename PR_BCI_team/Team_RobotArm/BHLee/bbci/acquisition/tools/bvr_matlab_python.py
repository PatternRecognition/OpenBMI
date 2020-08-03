import sys
import time

from RecorderRemoteControl import *
                    
if (len(sys.argv) > 1):    
    fcn = sys.argv[1]
    if(fcn == 'loadworkspace'):
        if(len(sys.argv) > 2):
            stopViewing()
            loadWorkspace(sys.argv[2])
        else:
            print 'Couldn''t load workspace: not enough arguments'
            print 'You have to specify a filename.'
            sys.exit(1)
    elif(fcn == 'startrecording'):
        if(len(sys.argv) > 2):
            viewData()
            startRecording(sys.argv[2])
        else:
            print 'Couldn''t start Recording: not enough arguments'
            print 'You have to specify a filename.'
            sys.exit(1)
    elif(fcn == 'startimprecording'):
        if(len(sys.argv) > 2):
            viewImpedance()
            time.sleep(10)
            viewData()
            startRecording(sys.argv[2])
        else:
            print 'Couldn''t start impedance recording: not enough arguments'
            print 'You have to specify a filename.'
            sys.exit(1)

    elif(fcn == 'stoprecording'):
        stopRecording()
    elif(fcn == 'viewsignals'):
        viewData()
    elif(fcn == 'viewsignalsandwait'):
        waitTime = 2
        if(len(sys.argv) > 2):
            waitTime = float(sys.argv[2]) / 1000

        if(getState() != 1):
            viewData()
            time.sleep(waitTime)
    elif(fcn == 'checkimpedances'):
        viewImpedance()
    elif(fcn == 'pauserecording'):
        pauseRecording()
    elif(fcn == 'resumerecording'):
        continueRecording()
    elif(fcn == 'getstate'):
        print getState()
    else:
        print 'Warning: The command ' + fcn + ' is not supported'
        sys.exit(1)
else:
    print 'No command specified'
    sys.exit(1)
