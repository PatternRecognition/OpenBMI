import ctypes

class raweth(object):
    
    def __init__(self, path):
        # Load DLL into memory.    
        outdll = ctypes.cdll.LoadLibrary(path)
        print("   A")
       
        self.startup = outdll.startup
        self.startup.restype = ctypes.c_int
        self.startup.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        print("   B")
        self.cleanup = outdll.cleanup
        self.cleanup.restype = ctypes.c_void_p
        self.startup.argtypes = [ctypes.c_void_p]
        print("   C")     
        self.add_participant = outdll.add_participant
        self.add_participant.restype = ctypes.c_int
        self.add_participant.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]        

        self.set_intensities = outdll.set_intensities
        self.set_intensities.restype = ctypes.c_void_p
        self.set_intensities.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double]        
            
        self.set_stimulation_mode = outdll.set_stimulation_mode
        self.set_stimulation_mode.restype = ctypes.c_void_p
        self.set_stimulation_mode.argtypes = [ctypes.c_int]        
        
        self.stimulate = outdll.stimulate
        self.stimulate.restype = ctypes.c_void_p
        self.stimulate.argtypes = [ctypes.c_int,  ctypes.c_int]        
