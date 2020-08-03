import numpy
screenPos = [100, 100, 800, 600] #screenPos

LOG_FILENAME = 'D:/temp/log/test.out' #might be changed from Matlab to sbj_specific folder!
maryDir = 'D:\\tools\\MARY TTS\\'
#JavaDir = 'C:\\Program Files (x86)\\Java\\jre6\\'
#SbjSimulationDataDir = 'C:/svn/bbci/python/pyff/src/Feedbacks/T9Speller/SubjectSimulator/"#D:/svn/bbci/python/pyff/src/Feedbacks/T9Speller/SubjectSimulator/'
SbjForSimulation = 'VPll'
#dictFile = 'C:/svn/bbci/python/pyff/src/Feedbacks/T9Speller/DE-DE-SMALL.dict'
#dictFile = "D:/svn/bbci/python/pyff/src/Feedbacks/T9Speller/py9/DE-DE-SMALL.dict"


simluate_sbj = True#True
spellerMode =  True
adaptiveSequence = False#True
manualParameters = False
EARLY_STOPPING = False
speechMode = True

#Parameters 
ISI = 225#1000
STIM_TIME = 100
PRIMING_ISI = 1000
PRIMING_SPELLING_OFFSET = 3000
TIME_DECISION_PRESENTATION = 4500
numPriming = 3
PAUSE_INTERVAL = 2
currentTargetStim = 0 #for calibration run
FPS=1000
cummulatedTime = 0
currSubtrial = 0

numDecisions = 0
numCounts = 0 
#this is the variabel which is used to transmit 
#the number of counts per trial (is is always set to 0 after a trial)
MIN_DISTANCE = 3

#parameters for the generation of TrialSequences
#default is calibration!!
MIN_MARKER_SEQ = 4 #just for early stopping
DECISION_CRITERION = 0.01 #just needed for early stopping
N_MARKER_SEQ = 2#12
MAX_NONMARKER_SEQ = 1
keysToSpell = [1, 2, 3, 4, 5, 6, 7, 8, 9]
#keysToSpell = [4]


templateTexts = ["absaugen", "umlagern", "CUFF", "ALARM", "Kopf nach links", "Kopf nach rechts", "Danke", "System Pause 3min"]


stopping, stopped,  askForCounts, endOfTrial, startOfTrial =  False, False, False, False, True
in_mode1, in_mode2, in_mode3  = True, False, False
paused = False#True #True

#parameter for the parallel port!
START_EXP, END_EXP = 253, 254
BEFORE_CUE, CUE_SHIFT, AFTER_CUE, CUE = 57, 60, 59, 58
START_TRIAL, END_TRIAL, PAUSE_START, PAUSE_STOPP = 50, 51, 55, 56
DEVIANT_SHIFT = 10  #if 7 is deviant, send 7+DEVIANT_SHIFT instead !#
FIRSTSEQUENCES_SHIFT = 20
DECISION_SHIFT = 150 #a decision for 5 is coded with a 155 marker!

DELAY_SPEECH_SYNTHESIS = 150
readStdEveryTrial = True

#sounds = ["sounds/new_high_l.wav", "sounds/new_high_m4.wav", "sounds/new_high_r.wav", \
#          "sounds/new_middle_l.wav", "sounds/new_middle_m4.wav", "sounds/new_middle_r.wav", \
#          "sounds/new_low_l.wav", "sounds/new_low_m4.wav", "sounds/new_low_r.wav"]


stoppingThresholdsTemplate = numpy.zeros([5])+ (10 ** -4)
