import os, pygame, random

#pygame version (no pySonic)

class SoundModule:
    """Handles collections of impact sounds."""

    def __init__(self, path):
        """Read every wave file in a directory. Each file should be in the format 
        basenamexxx.wav, where xxx is a three digit sequence number. A list of
        (filename, sample) tuples is mapped to each unique basename. The samples
        are only loaded after load_bank is called."""       
        files = os.listdir(path)
        self.mapping = {}
        
        for file in files:
            dot_point = file.find('.')
            if dot_point>0:
                extension = file[dot_point:-1]
                name = file[0:dot_point-3]
                number = file[dot_point-2:dot_point]
      
                fullpath = os.path.join(path, file) 
                if self.mapping.has_key(name):
                    self.mapping[name] = self.mapping[name] + [[fullpath, None]]
                else:
                    self.mapping[name] = [[fullpath, None]]
        
    
        
        
    def play_from_bank(self, bank, volume):    
    
        if self.mapping.has_key(bank):
            samples = self.mapping[bank]
            sample = random.choice(samples)
            
            if sample[1] != None:
            
                if volume<0:
                    volume = 0
                if volume>255:
                    volume = 255    
                    
                
                sample[1].set_volume(volume/255.0)
                sample[1].play()
                
    def free_bank(self, bank):
        if self.mapping.has_key(bank):
            for f in self.mapping[bank]:                
                del f[1].Sound
                f[1] = None
    
    def get_banks(self):
        """Return all the bank names."""
        return mapping.keys()
        
    def load_bank(self, bank):        
        """Load all of the samples for a given bank name"""
        if self.mapping.has_key(bank):
            for f in self.mapping[bank]:                                
                
                f[1] = pygame.mixer.Sound(f[0])
    
            


