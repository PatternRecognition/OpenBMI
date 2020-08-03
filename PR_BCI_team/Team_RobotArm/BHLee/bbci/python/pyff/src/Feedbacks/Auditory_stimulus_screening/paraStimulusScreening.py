screenPos = [-1920, 0, 1920, 1200]; #for the lab computer!
#self.screenPos = [-1600, 0, 1600, 1200]; #for the lab computer!
#self.screenPos = [0, 0, 1920, 1200];
#screenPos = [-1920, 0, 1920, 1200] ORIGINAL

LOG_FILENAME = 'T9Speller.out' #might be changed from Matlab to sbj_specific folder!
simulate_sbj  = False
spellerMode =  False
adaptiveSequence = False
manualParameters = False
EARLY_STOPPING = False
speechMode = False
N_MARKER_SEQ = 14

sounds = ["sounds/bariton_ti.wav", "sounds/tenor_it.wav", "sounds/sopran2_ti.wav", \
          "sounds/bariton_ta3.wav", "sounds/tenor_at.wav", "sounds/sopran2_ta.wav", \
          "sounds/bariton_to2.wav", "sounds/tenor_ot.wav", "sounds/sopran2_to.wav"]
