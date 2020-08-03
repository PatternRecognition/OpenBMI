try:
    import numpy
    have_numpy = True
    class Ringbuffer:
        def __init__(self, size, init=0):
            if size<1:
                throw(Exception("Invalid size for a ringbuffer: must be >=1"))          
            self.n_samples = size        
            self.samples = numpy.ones((size,))*init
            self.read_head = 1
            self.write_head = 0
            self.sum = 0
            
        def get_length(self):
            return self.n_samples
            
            
        def get_samples(self):        
            return numpy.hstack((self.samples[self.read_head-1:],self.samples[0:self.read_head-1]))
        
            
        def get_sum(self):
            return self.sum
            
        def get_output(self):
            #self.read_head %= self.n_samples
            return self.samples[self.read_head-1]
            
        def get_mean(self):
            return self.sum / float(self.n_samples)
            
            
            
        def forward_index(self, i):
            new_index = self.read_head+i-1
            new_index = new_index % self.n_samples
            return self.samples[new_index]
            
        def reverse_index(self, i):
            
            new_index = self.write_head-i-1
            while new_index<0:
                new_index+=self.n_samples
            return self.samples[new_index]  
            
        def new_sample(self, x):
            s = self.samples[self.write_head]
            self.samples[self.write_head] = x        
            self.sum += x
            self.sum -= self.samples[self.read_head]
            self.read_head += 1
            self.write_head += 1
            self.read_head %= self.n_samples
            self.write_head %= self.n_samples
            return s
        

        
except:
    have_numpy = False
    class Ringbuffer:
        def __init__(self, size, init=0):
            if size<1:
                throw(Exception("Invalid size for a ringbuffer: must be >=1"))          
            self.n_samples = size        
            self.samples = [init] * size 
            self.read_head = 1
            self.write_head = 0
            self.sum = 0
            
        def get_length(self):
            return self.n_samples
            
            
        def get_samples(self):        
            return self.samples[self.read_head-1:] + self.samples[0:self.read_head-1]
            
                        
        
            
        def get_mean(self):
            return self.sum / float(self.n_samples)
            
            
        def get_sum(self):
            return self.sum
            
        def get_output(self):
            #self.read_head %= self.n_samples
            return self.samples[self.read_head-1]
            
            
        def forward_index(self, i):
            new_index = self.read_head+i-1
            new_index = new_index % self.n_samples
            return self.samples[new_index]
            
        def reverse_index(self, i):
            
            new_index = self.write_head-i-1
            while new_index<0:
                new_index+=self.n_samples
            return self.samples[new_index]  
            
        def new_sample(self, x):
            s = self.samples[self.write_head]
            self.samples[self.write_head] = x        
            self.sum += x
            self.sum -= self.samples[self.read_head]
            self.read_head += 1
            self.write_head += 1
            self.read_head %= self.n_samples
            self.write_head %= self.n_samples
            return s
            

class GeneralRingbuffer:
        def __init__(self, size, init=None, init_fn=None):
            if size<1:
                throw(Exception("Invalid size for a ringbuffer: must be >=1"))          
            self.n_samples = size        
            
            # either stuff with initial values or call init_fn for each new value
            if init_fn:
                self.samples = []
                for i in range(size):
                    self.samples.append(init_fn())
            else:
                self.samples = [init] * size 
                
            self.read_head = 1
            self.write_head = 0
            
            
        def get_length(self):
            return self.n_samples
            
            
        def get_samples(self):        
            return self.samples[self.read_head-1:] + self.samples[0:self.read_head-1]
            
                        
        
            
            
        def get_output(self):
            #self.read_head %= self.n_samples
            return self.samples[self.read_head-1]
            
            
        def forward_index(self, i):
            new_index = self.read_head+i-1
            new_index = new_index % self.n_samples
            return self.samples[new_index]
            
        def reverse_index(self, i):
            
            new_index = self.write_head-i-1
            while new_index<0:
                new_index+=self.n_samples
            return self.samples[new_index]  
            
        def new_sample(self, x):
          
            s = self.samples[self.write_head]
            self.samples[self.write_head] = x                    
            self.read_head += 1
            self.write_head += 1
            self.read_head %= self.n_samples
            self.write_head %= self.n_samples
            return s
            
    
    

