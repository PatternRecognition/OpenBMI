# This is the remote control for the BrainVision Recorder via a DCOM interface
# for configuration of the DCOM interface see file <documentation>.
# If you get an error which states something like "Access Denied" you have to
# check the DCOM settings of the server and your machine
#
# You can get some information about the BrainVision Recorder DCOM interface
# from PythonWin
# start PythonWin and click on Tools->Com Browser
# In the Com Browser go to "Registered Type Libraries" -> "Vision Recorder 1.0 Type Library"
#
# e.g. if you want to find some information about the "recorder.Acquisition.ViewData()"
# method go to "Registered Type Libraries" -> "Vision Recorder 1.0 Type Library"
#           -> "Type Library" -> "IAcquisition" -> "ViewData"
# There you will find the method structure.
#
# To use this file and to have PythonWin available you have to install PythonWin.
# See http://python.net/crew/mhammond/win32/Downloads.html for more information.
#
# 2008/08/28 - Max Sagebaum
#               - File created

import win32com.client

global recorder

def createRecorder():
    """ This method calls the methods to create the BrainVision Recorder COM
    object. You don't have to call this method from your own code. """
    global recorder
    recorder = win32com.client.Dispatch("VisionRecorder.Application")

def closeRecorder():
    """ This method calls the methods to delete the BrainVision Recorder COM
    object. You don't have to call this method from your own code. """
    recorder = None
    
def viewData():
    createRecorder()
    recorder.Acquisition.ViewData()
    closeRecorder()

def viewImpedance():
    createRecorder()
    recorder.Acquisition.ViewImpedance()
    closeRecorder()

def viewTestSignal():
    createRecorder()
    recorder.Acquisition.ViewTestsignal()
    closeRecorder()

def startRecording(filename):
    """ The filename has to be an absolute path."""
    createRecorder()
    recorder.Acquisition.StartRecording(filename)
    closeRecorder()

def pauseRecording():
    createRecorder()
    recorder.Acquisition.Pause()
    closeRecorder()

def continueRecording():
    createRecorder()
    recorder.Acquisition.Continue()
    closeRecorder()

def stopRecording():
    createRecorder()
    recorder.Acquisition.StopRecording()
    closeRecorder()

def stopViewing():
    createRecorder()
    recorder.Acquisition.StopViewing()
    closeRecorder()

def selectMontage(montage):
    createRecorder()
    recorder.Acquisition.SelectMontage(montage)
    closeRecorder()

def viewRefresh():
    createRecorder()
    recorder.Acquisition.ViewRefresh()
    closeRecorder()

def dcCorrection():
    createRecorder()
    recorder.Acquisition.DCCorrection()
    closeRecorder()

def setMarker(description, markerType):
    """ The marker can have any name and any type you want. But the marker types
    Response and Stimulus are recognized by the recorder. """
    createRecorder()
    recorder.Acquisition.SetMarker(description, markerType)
    closeRecorder()

def loadWorkspace(filename):
    createRecorder()
    recorder.CurrentWorkspace.load(filename)
    closeRecorder()

def getNameOfWorkspace():
    createRecorder()
    name = recorder.CurrentWorkspace.Name
    closeRecorder()

    return name

def getFullNameOfWorkspace():
    createRecorder()
    name = recorder.CurrentWorkspace.FullName
    closeRecorder()

    return name

def getRawFileFolder():
    createRecorder()
    folder = recorder.CurrentWorkspace.RawFileFolder
    closeRecorder()

    return folder

def quitRecorder():
    createRecorder()
    recorder.Quit()
    closeRecorder()

def getVersion():
    createRecorder()
    version = recorder.Version
    closeRecorder()

    return version

def getState():
    """ The return value will describe the following state:
        0 - nothing
        1 - view data
        2 - view test signal
        3 - view impedance
        4 - record data
        5 - record test data
        6 - pause record data
        7 - pause record test data

        the list may be incomplete"""
    createRecorder()
    state = recorder.State
    closeRecorder()

    return state
