import math, random
from ringbuffer import Ringbuffer
from ringbuffer import GeneralRingbuffer
import biquad



try:
    # begin numpy required block
    #
    #
    
    import numpy, scipy, scipy.signal

    #sg coefficient computation
    def savitzky_golay(window_size=None,order=2):
        if window_size is None:
            window_size = order + 2

        if window_size % 2 != 1 or window_size < 1:
            raise TypeError("window size must be a positive odd number")
        if window_size < order + 2:
            raise TypeError("window size is too small for the polynomial")

        # A second order polynomial has 3 coefficients
        order_range = range(order+1)
        half_window = (window_size-1)//2
        B = numpy.mat(
            [ [k**i for i in order_range] for k in range(-half_window, half_window+1)] )


        M = numpy.linalg.pinv(B)
        return M
        

    # savitzky-golay polynomial estimator, with optional derivatives
    class SavitzkyGolay:
        def __init__(self, size, deriv=0, order=4):
            if size%2==0:
                print "Size for Savitzky-Golay must be odd. Adjusting..."
                size = size + 1        
            self.size = size        
            self.deriv = deriv
            if order<deriv+1:
                order = deriv+1
            sgolay = savitzky_golay(size, order)
            diff = numpy.ravel(sgolay[deriv, :])        
            self.filter = Filter(diff, [1])
            
        def new_sample(self, x):
            return self.filter.new_sample(x)
                    
            

    class ButterFilter:
        def __init__(self, n, band, type='low'):    
            ba = scipy.signal.butter(n, band, btype=type)
            self.filter = Filter(ba[0], ba[1])
        
        def new_sample(self, x):
            return self.filter.new_sample(x)
            
            

    class BesselFilter:
        def __init__(self, n,  band, type='low'):    
            ba = scipy.signal.bessel(n, band, btype=type)
            self.filter = Filter(ba[0], ba[1])
        
        def new_sample(self, x):
            return self.filter.new_sample(x)
            
            
    class FIRFilter:
        def __init__(self, n, cutoff):
            b = scipy.signal.firwin(n, cutoff)
            self.filter = Filter(b, [1])
           
           
        def new_sample(self, x):
            return self.filter.new_sample(x)
            
            
    # simple 1d projection of an nd vector
    class Project:
        def __init__(self, matrix): 
            self.matrix = matrix
            
        def new_sample(self, vec):  
            return numpy.dot(self.matrix,vec)[0]
            
    # apply a matrix to an nd vector
    class MatrixOperator:
        def __init__(self, A, homo=False):
            self.A = A
            self.homo = homo
            
        def new_sample(self, vec):
            if homo:
                vec = numpy.hstack(numpy.array(vec), numpy.array([1]))
                return numpy.dot(self.A, vec)[0:-1]
            else:
                vec = numpy.array(vec)
                return numpy.dot(self.A, vec)  
                
    class PolyResponse:
        def __init__(self, x, y, order=None):
            if order==None:
                order = len(x)
            self.coeff = numpy.polyfit(x,y,order)
            
        def new_sample(self, x):
            numpy.polyval(self.coeff, x)    
    # end numpy required block
    #
    #
except:
        have_numpy = False
        
# simple threshold
class BinaryThreshold:
    def __init__(self, thresh):
        self.thresh = thresh
        
    def new_sample(self, x):
        if x>thresh:
            return 1.0
        else:
            return 0.0
            

            
            
class MultiDimensionalProcessor:
    def __init__(self, processors):
        self.processors = processors
        
    def new_sample(self, vec):
        s = []
        for v,p in vec,processors:
            s.append(p.new_sample(v))
            
        return s
        
        
# returns a binary stream of 0,1 with the given transition probabilities
class BinaryMarkov:
    def __init__(self, ps=[0.001, 0.001]):
        self.ps = ps
        self.state = random.choice([0,1])        
        
    def set_ps(self, ps):
        self.ps = ps
        
    def new_sample(self,x):
        r = random.random()                                
        if self.state==0:            
            if r<self.ps[0]:                
                self.state = 1                        
        elif self.state==1:
            if r<self.ps[1]:
                self.state = 0


# implement a biquad cascade filter
# i.e. an sos filter. uses the RBJ filter designs
# Types are: 
# lpf = lowpass
# hpf = highpass
# bpf = bandpass
# notch = notch
# apf = allpass
# peaking = peaking filter
# lowshelf = low shelf
# highshelf = highshelf
# each filter can take a specifier either of Q, bandwidth (BW) or S
# dbGain applies only to peaking, highshelf and lowshelf, and specifies how much
# boost/cut to apply to those regions
# power specifies the length of the cascade
# note that there will be a delay of 2*power samples
class BiquadCascade:
    def __init__(self, freq=0.5, fs=1.0, type="lpf", Q=1, BW=None, S=None, dbGain=0.0, power=1):
        self.filters = []
        for p in range(power):
            B,A = biquad.design_biquad(f0=freq, Fs=fs, type=type, Q=Q, BW=BW, S=S, dbGain=dbGain)
            self.filters.append(Filter(B,A))
            
    def new_sample(self, x):
        y = x
        for filter in self.filters:
            y = filter.new_sample(y)
        return y
        
# apply a different filter depending on whether a signal is increasing or decreasing       
class AsymmetricFilter:
    def __init__(self, inc_filter, dec_filter, fill_blank =True):
        self.inc_filter = inc_filter
        self.dec_filter = dec_filter
        self.x = None
        self.fill_blank = fill_blanl
        
    def new_sample(self,x):
        if self.x==None:
            self.x = x
            return x
            
        if x>self.x:
            if self.fill_blank:
                self.dec_filter.new_sample(x)                
            return self.inc_filter.new_sample(x)
        else:
            if self.fill_blank:
                self.inc_filter.new_sample(x)                
            return self.dec_filter.new_sample(x)
  
class AsymmetricAcquistion:
    def __init__(self, inc_acq, dec_acq):
        self.x = None
        self.inc_acq = inc_acq
        self.dec_acq = dec_acq
        
    def new_sample(self, x):
        if self.x==None:
            self.x = x
            return x
        
        if x>self.x:
            self.x = (1-self.inc_acq)*self.x + (self.inc_acq)*x
        else:
            self.x = (1-self.dec_acq)*self.x + (self.dec_acq)*x
        return self.x
  
        
class MultiRegionResponse:
    def __init__(self):
        self.regions = []
    
    # regions should not overlap!
    def add_region(self, range, chain):
        self.regions.append(range, chain)
        
    def new_sample(self, x):
        for region in self.regions:
            range,chain = region
            if x>=range[0] and x<range[1]:
                return chain.new_sample(x)                
        return x
        
        
    
        
class LeakyIntegrator:
    def __init__(self, gain=1.0, leakage=1.0, saturation=None):
        self.saturation = saturation
        self.gain = gain
        self.leakage = leakage
        self.integ =0 
        
        if saturation!=None:
            saturation = list(saturation)
            
            if saturation[0]==None:                            
                saturation[0] = -1e50
                
            if saturation[1]==None:            
                saturation[1] = 1e50
        else:
            saturation = [-1e50, 1e50]
            
        self.clamp = Clamp(saturation[0], saturation[1])
        
    def reset(self):
        self.integ = 0
        
    def new_sample(self, x):
        self.integ = self.clamp.new_sample((self.integ+self.gain*x)*self.leakage)
        return self.integ
        
        
        
class PIDControl:
    def __init__(self, p=1.0, i=0.0, d=0.0, leakage=1.0, max_int=1e50):
        self.p = p
        self.i = i
        self.d = d
        self.int = 0                
        self.deriv = Differentiator()
        self.integrator = LeakyIntegrator(leakage=leakage, saturation=(-max_int, max_int))
         
        
    def new_sample(self, x):
        error = x        
        dc = self.deriv.new_sample(error) * self.d
        ic = self.integrator.new_sample(error) * self.i
        pc = (error) * self.p
        return dc+ic+pc
        
        
        
            
            
            
# call a function when a signal passes through a threshold. passes value and sign of transition to the function, along with 
# optional context data
class TransitionNotify:
    def __init__(self, thresh, notify_fn, data=None):
        self.notify_fn = notify_fn
        self.thresh = thresh
        self.last_x = None
        self.data = data


    def new_sample(self, x):
        if x<self.transition and self.last_x>=self.transition:
            self.notify_fn(x, True, self.data)

        if x>self.transition and self.last_x<=self.transition:
            self.notify_fn(x, False, self.data)
        


class Spring:
    def __init__(self, k, d, init_x = 0, init_dx = 0):
        self.x = [init_x,init_dx,0]        
        self.k = k
        self.d = d
        
    def new_sample(self, x):
        self.x[2] = k*(self.x[0]-x)
        self.x[1] = self.x[1] * self.d + self.x[2]
        self.x[0] = self.x[0] + self.x[1]
        return self.x[0]
    
    

        
        
class PiecewiseResponse:

    #regions should be ordered and non-overlapping
    def __init__(self, regions,  type="linear"):
        self.type = type
        self.regions= regions
        
        
    def linear(self, x):
        for r in self.regions:
            
            (start, end, start_v, end_v) = r            
            if x>=start and x<end:
                pos = (x-start) / float((end-start-1))                               
                value = (1-pos)*start_v + pos*end_v
                return value
        return x
        
        
    def constant(self, x):    
        for r in self.regions:
            (start, end, start_v, end_v) = r
            if x>=start and x<end:                
                value = 0.5*start_v + 0.5*end_v
                return value
        return x
        
            
    def cubic(self, x):   
        if not have_numpy:
            return self.linear(x)
            
        index = 0
        for r in self.regions:
            (start, end, start_v, end_v) = r
            
            if x>=start and x<end:
                pos = (x-start) / float(end-start-1)
            
                # must be consecutive for cubic interpolation
                if index>0 and index<len(self.regions)-1:
                    next = self.regions[index+1]
                    prev = self.regions[index-1]
                    if prev[1] == start and next[0]==end:
                        xv = numpy.array([prev[0], start, end, next[1]])
                        yv = numpy.array([prev[2], start_v, end_v, next[3]])
                        coeff = numpy.polyfit(xv, yv, 3)
                        return numpy.polyval(coeff, x)
                       
                
            
                # otherwise, just use the linear interpolation                
                value = (1-pos)*start_v + pos*end_v
                return value
            index = index + 1
        return x
        
                
        
    def new_sample(self, x):
        if self.type=="linear":
            return self.linear(x)            
        elif self.type=="cubic":
            return self.cubic(x)            
        else:
            return self.constant(x)
        
        
class SparseMultiTapDelay:
    def __init__(self, taps):
        self.taps = taps
        max_t = 0
        for time,scale in taps:
            if time>max_t:
                max_t = time
                
        self.buffer = Ringbuffer(max_t)
        
    def new_sample(self, x):
        result = 0
        for time, scale in self.taps:
            result = result + self.buffer.reverse_index(time) * scale
        return result
                
        
  


        
class Delay:
    def __init__(self, delay):
        self.delay = delay
        self.buffer = Ringbuffer(delay)
        
    def new_sample(self, x):
        return self.buffer.new_sample(x)
        

        
class WindowedAverage:
    def __init__(self, window):
        if have_numpy:
            self.window = numpy.array(window)
        else:
            self.window = list(window)
            
        self.buffer = Ringbuffer(len(window))
        
        
    def new_sample(self, x):
        self.buffer.new_sample(x)
        
        if have_numpy:
            avg = numpy.mean(self.buffer.get_samples() * self.window)
        else:
            i = 0
            sum = 0
            for s in self.buffer.get_samples():
                sum += self.window[i]*s
                i = i + 1
                
            avg = sum/len(self.window)
            
            
        return avg
        
class MovingAverage:
    def __init__(self, window_len):
        self.window_len = window_len
        self.buffer = Ringbuffer(window_len)
        
        
    def new_sample(self, x):
        self.buffer.new_sample(x)
        return self.buffer.get_sum()/float(self.window_len)
        
        
        
class Quantize:
    def __init__(self, step_size=1.0):
        self.step_size = step_size
        
    def new_sample(self, x):
        s = math.floor(x/self.step_size)*self.step_size
        return s
        
        
class NoiseSource:
    def __init__(self, amount, gaussian=False, seed=None): 
        self.amount = amount
        self.gaussian = gaussian
        self.source = random.Random(seed)
        
        
    def new_sample(self, x):
        if self.gaussian:
            return x + self.source.gauss(0, amount)
        else:
            return self.x + self.source.uniform(-amount, amount)
        
        
        
class Hysteresis:

    def __init__(self, transition, width):
        self.transition = transition
        self.width = width
        self.x_offset = 0
        self.last_x = None
        
    def new_sample(self, x):
    
        if self.last_x==None:
            self.last_x = x
            return x
            
        if x<self.transition and self.last_x>=self.transition:
            self.x_offset = -width

        if x>self.transition and self.last_x<=self.transition:
            self.x_offset = width
            

        self.last_x = x
        return x + self.x_offset
    

        

    
class BinaryOperatorFilter:
    def __init__(self, binop, src_a, src_b):
        self.binop = binop
        self.src_a = src_a
        self.src_b = src_b
        
    def new_sample(self, x, y):
        return self.binop(self.src_a.new_sample(x),self.src_b.new_sample(y))
    

# apply any given operator (+,-,sin,cos, etc.)
class OperatorFilter:
    def __init__(self, op):
        self.op = op
        
    def new_sample(self, x):
        return self.op(x)
        

class FirstOrderLag:
    def __init__(self, lag):
        self.lag = lag
        self.state = None
        
    def new_sample(self, x):
        if self.state!=None:
            self.state = self.lag*self.state+(1-self.lag)*x
        else:
            self.state = x
            
        return self.state
  

class SecondOrderLag:
    def __init__(self, lag_a, lag_b):
        self.filter = Filter([0,0,0], [1, lag_a, lag_b])
        
        
    def new_sample(self, x):
        return self.filter.new_sample(x)
        
  
  
class Monotonize:
    def __init__(self, negative=False):
        self.negative = negative
        self.minmax = None
        
        
    def set_sign(self, negative):
        self.negative = negative
        
        
    def get_sign(self):
        return self.negative
        
    def reset(self):
        self.minmax = None
        
    def new_sample(self, x):
        if self.minmax==None:
            self.minmax = x
            return x
        else:
            if self.negative:
                if x<self.minmax:
                    self.minmax = x                
            else:
                if x>self.minmax:
                    self.minmax = x
                    
            return self.minmax
  
# a zone around where scale is different
class RescaleZone:
    def __init__(self, start, end, scale):
        self.start = start
        self.end = end
        self.scale = scale
        self.region_size = self.end-self.start
        self.difference = self.region_size - self.region_size*self.scale
        self.centre = (self.start+self.end)/2.0
        
    def new_sample(self, x):
        if x<self.start:
            y = x + self.difference / 2.0
        elif x>self.end:
            y = x - self.difference / 2.0
        else:
            y = ((x - self.centre) * self.scale) + self.centre
            
        return y
        


class EnergyBuffer:
    def __init__(self, len):
        self.buffer = Ringbuffer(len)
        self.differentiator = Differentiator()
        
    def new_sample(self, x):
        d = self.differentiator.new_sample(x) 
        self.buffer.new_sample(x*x)
        return math.sqrt(self.buffer.get_mean())
        
        

class Rescale:
    def __init__(self, gain, offset):
        self.gain = gain
        self.offset = offset
        
        
    def new_sample(self, x):
        return x*self.gain-self.offset
  



class Clamp:
    def __init__(self, min, max):
        self.min = min
        self.max = max
        
        
    def new_sample(self, x):
        if x<self.min:
            return self.min
        elif x>self.max:
            return self.max
        return x
            
        
# Model a smooth linear transition between start and end
class LinearTransition:
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.width = end-start
        
        
    # return the value of the function given a new position
    def new_sample(self, x):
        if x<self.start:
            return 0
            
        if x>self.end:
            return 1
            
        t = (x - self.start) / self.width
        return t       


class FirstOrderHP:
    def __init__(self, lag):
        self.lag = lag
        self.state = None
        
    def new_sample(self, x):
        if self.state!=None:
            self.state = self.lag*self.state+(1-self.lag)*x
        else:
            self.state = x
            
        return x-self.state
        

        
        
class Differentiator:
    def __init__(self, default_dt = 1.0):
        self.last = 0
        self.default_dt = default_dt
        
    def new_sample(self, x, dt=None):
        if dt==None:
            dt = self.default_dt
            
        d = (x - self.last)/dt
        self.last = x
        return d
    
class Integrator:
    def __init__(self, rate=1.0):
        self.sum =0 
        self.rate = rate
        
        
    def reset(self):
        self.sum =0 
        
        
    def new_sample(self, x):
        self.sum+=x*self.rate
        return self.sum

class DeadZone:
    def __init__(self, zone, offset=0):
        self.set_zone(zone)
        self.offset = offset
        self.state = None
        
    def set_zone(self, zone):
        self.zone = zone
        
    def new_sample(self, x):
    
        if self.state!=None:
            if self.state - x - self.offset>self.zone:
                self.state = x+self.zone
            elif self.state-x-self.offset<-self.zone:
                self.state = x-self.zone
        else:
            self.state = x
            
        return self.state
        


        

        
   
   
class Unwrap:
    def __init__(self, wrap=math.pi):
        self.wrap = 0
        self.old_x = None
        self.wrap_level = wrap
    
    def new_sample(self, x):
        if self.old_x:
            if x<-self.wrap_level/2 and self.old_x>self.wrap_level/2:
                self.wrap = self.wrap + 2*self.wrap_level
                
        if x>self.wrap_level/2 and self.old_x<-self.wrap_level/2:
                self.wrap = self.wrap - 2*self.wrap_level                
        
        self.old_x = x
            
        return x+self.wrap
        
        
        
        
class ProcessChain:
    # set the processes
    # if a process is a tuple
    # this chain will get an attribute with the name of the second tuple value
    # i.e. (Unwrap(100), "unwrapper") adds an attribute unwrapper to the process chain
    def __init__(self, processes):
        self.process_list = []
        for process in processes:
            if type(process) == type((2,3)):
                self.process_list.append(process[0])
                setattr(self, process[1], process[0])
            else:
                self.process_list.append(process)
        
        
        
    def new_sample(self, x):        
        for process in self.process_list:
            x = process.new_sample(x)
                        
        return x
    
 
class SoftLock:
    def __init__(self, exponent = 0.2):
        self.state = None
        self.target = 0
        self.lock = 0.0
        self.exponent = exponent
        
        
    def set_lock(self, v):        
        self.lock = v**self.exponent
                
        
    def new_sample(self, x):
        if self.state==None:
            self.state = x
        self.target = x        
        self.state = (self.lock)*self.state + (1.0-self.lock)*self.target
        return self.state   
        
        
       
class SignalLock:
    def __init__(self):
        self.state = 0
        self.locked = False
        self.unlocking = False
        self.last_value = 0
        self.count_down = 0
        
        
        
    def lock(self):
        if not self.locked:
            self.state = self.last_value
            self.locked = True
        self.unlocking = False
        
    def unlock(self, after=0):
        if not self.unlocking:    
            self.count_down = after
            self.unlocking = True    
        
        
        
        
    def new_sample(self, x):
    
        if self.unlocking:
            if self.count_down>0:
                self.count_down = self.count_down -1
            else:
                self.locked=False
            
        self.last_value = x
        if self.locked:
            return self.state
        else:
            return x
        
    
class DataStreamLogger:
    def __init__(self, name):
        self.name = name
        self.value = None
        
    def new_sample(self, x):
        self.value = x

class MultiLogger:
    def __init__(self, ofname):
        self.loggers = []        
        self.ofile = open(ofname, "w")
        
    def create_logger(self, name):
        d = DataStreamLogger(name)
        self.loggers.append(d)        
        return d
        
    def tick(self):
        loggers = {}
        for logger in self.loggers:
            loggers[logger.name] = logger.value
        self.ofile.write(str(loggers))
        self.ofile.flush()
                
    def close(self):
        self.ofile.close()
        
        
        
                        
    

class TimeInterpolator:
    def __init__(self, fs, average = False):        
        self.fs = fs
        self.last_t = 0
        self.sample_time = 1.0/fs
        self.first_time = True
        self.last_x = 0
        self.new_packets = []
        self.true_t = 0
        self.average = average
        
    
    def get_packets(self):
        return self.new_packets
               
    def add_packet(self, x, dt):
    
        if self.first_time:        
            self.last_t = dt
            self.last_x = x
            self.first_time = False
            self.new_packets = [x]
            return
            
        self.true_t = self.true_t + dt
        
        difference = self.true_t - self.last_t                       
        samples = math.floor(difference / self.sample_time)        
            
    
        # insert the interpolated packets
        self.new_packets = []
        for i in range(samples):
            interp = i/float(samples)
            new_sample = (1-interp)*self.last_x + (interp)*x
            self.new_packets.append(new_sample)
            
        
        self.last_t += samples*self.sample_time       
        self.last_x = x
    
             
             
class Interpolator:
    def __init__(self, wrap_limit = 256):        
        self.wrap = 0
        self.wrap_limit = wrap_limit
        self.last_t = 0
        self.last_true_t = 0
        self.first_time = True
        self.last_x = 0
        self.new_packets = []
        
    def add_packet(self, x, t):
    
        if self.first_time:
            self.last_true_t = t
            self.last_t = t
            self.last_x = x
            self.first_time = False
            self.new_packets = [x]
            return
        
    
        #if we see a wrap
        if t<self.wrap_limit/2 and self.last_t>self.wrap_limit/2:
            self.wrap+=self.wrap_limit

        
        true_t = self.wrap+t
        
        difference = true_t - self.last_true_t
        
        
        
        # this packet isn't new
        if difference==0:
            self.new_packets = []
            
        # just one new packet
        elif difference==1:
            self.new_packets = [x]
                        
        else:
            #linear interpolate
            self.new_packets = []
            for d in range(difference):
                a = float(d) / float(difference)
                i = (a)*self.last_x+(1-a)*x                
                self.new_packets = [i] + self.new_packets
        
        self.last_t = t
        self.last_true_t = true_t
        self.last_x = x
        
    def get_packets(self):
        return self.new_packets
        

class SOSFilter:
    def __init__(self, sos, g=1.0):
        self.filters = []
        if type(g)!=type([]):
            g = [g]
            
        
        if len(g)<(len(sos)+1):
            g = g + ([1.0] * ((len(sos)+1)-len(g)))
            
        print g
        
        self.g = g
        for row in sos:
            self.filters.append(Filter(row[0:3], row[3:6]))
        
            
    def new_sample(self, x):
        y = x
        gindex = len(self.g)-1
        for filter in self.filters:
            y = filter.new_sample(y)*self.g[gindex]
            gindex = gindex-1
        return y*self.g[0]
            
        
class Identity:
    def __init__(self):
        pass
        
    def new_sample(self, x):
        return x
        
        
class Constant:
    def __init__(self, c):
        self.constant = c
        
    def set_constant(self,c):
        self.constant = c
        
    def new_sample(self,x):
        return c
        
        
        
# estimate period of zero (or other threshold) crossings
# signmode can be "both", "up" or "down"
class CrossingEstimator:
    def __init__(self, threshold=0, signmode="both", averaging=1):
        self.threshold = threshold
        self.signmode = signmode
        self.averaging = averaging
        self.state = None
        if averaging>1:
            self.averager = MovingAverage(averaging)
        else:
            self.averager = Identity()
        self.count = 0
        self.period = 0
            
    def new_sample(self, x):
        if self.state==None:
          self.period = 0
        elif self.state<self.threshold and x>self.threshold:
            if self.signmode=="both" or self.signmode=="up":
                self.period = self.averager.new_sample(self.count)                                    
                self.count = 0
        elif self.state>self.threshold and x<self.threshold:
            if self.signmode=="both" or self.signmode=="down":
                self.period = self.averager.new_sample(self.count)                                    
                self.count = 0
        
        self.count = self.count + 1
        
        self.state = x
        return self.period
        
        
        
        
class FIRFilter:
    def __init__(self, kernel):
        self.filter = Filter(kernel, [1.0, 0.0, 0.0])
        
    def new_sample(self, x):
        return self.filter.new_sample(x)
        
        
class Filter:
    def __init__(self, B, A):
        """Create an IIR filter, given the B and A coefficient vectors"""
        self.B = B
        self.A = A
        if len(A)>2:
            self.prev_outputs = Ringbuffer(len(A)-1)
        else:
            self.prev_outputs = Ringbuffer(3)
            
        self.prev_inputs = Ringbuffer(len(B))
        
    def filter(self, x):
        #take one sample, and filter it. Return the output
        y = 0
        self.prev_inputs.new_sample(x)
        k =0 
        for b in self.B:
            y = y + b * self.prev_inputs.reverse_index(k)
            k = k + 1
        
        k = 0
        for a in self.A[1:]:
            y = y - a * self.prev_outputs.reverse_index(k)
            k = k + 1
            
        y = y / self.A[0]
        
        self.prev_outputs.new_sample(y)
        return y
        
        
    def new_sample(self,x):
        return self.filter(x)
        
        
class RandomDelay:
    def __init__(self, window):
        self.window = window
        self.buffer = Ringbuffer(window)
        
    def new_sample(self, x):
        self.buffer.new_sample(x)
        s = self.buffer.get_samples
        return random.choice(s)
        


class ReduceFilter:
    def __init__(self, binop):
        self.state = None
        self.binop = binop
        
    def new_sample(self, x):
        if self.state==None:
            self.state = x
        else:
            self.state = self.binop(self.state,x)
        return self.state
    

class WindowedReduceFilter:
    def __init__(self, binop, window):
        self.state = None
        self.window = window
        self.buffer = Ringbuffer(window)
        self.binop = binop
        
    def new_sample(self, x):
        self.buffer.new_sample(x)
        samples = self.buffer.get_samples()
        state = reduce(self.binop, samples)
        return state
        
    
        
        
class Median:
    def __init__(self, window):
        self.window = window
        self.buffer = Ringbuffer(window)                               
        
    def new_sample(self, x):
        self.buffer.new_sample(x)
        b = list(self.buffer.get_samples())
        b.sort()
        lb = len(b)
        # even case
        if lb%2==0:        
            return b[len(b)/2]+b[len(b)/2+1]
        else:      
        # odd case
            return b[len(b+1)/2]
            
            
class Dilation:
    def __init__(self, window):
        self.window = window
        self.buffer = Ringbuffer(window)
                                    
    def new_sample(self, x):
        self.buffer.new_sample(x)
        return max(self.buffer.get_samples())
        

class Erosion:
    def __init__(self, window):
        self.window = window
        self.buffer = Ringbuffer(window)
        
    def new_sample(self, x):
        self.buffer.new_sample(x)
        return min(self.buffer.get_samples())
        
        
class Opening:
    def __init__(self, erode, dilate):
        self.eroder = Erosion(erode)
        self.dilator = Dilation(dilate)
        
    def new_sample(self, x):
        return self.dilator.new_sample(self.eroder.new_sample(x))
        


class Closing:
    def __init__(self, erode, dilate):
        self.eroder = Erosion(erode)
        self.dilator = Dilation(dilate)
        
    def new_sample(self, x):
        return self.eroder.new_sample(self.dilator.new_sample(x))
        
        
    

class Decimator:
    def __init__(self, decimation=10):
        self.decimation = 10
        self.ctr =0 
        
    def new_sample(self, x):  
        self.ctr = self.ctr+1
        if self.ctr == self.decimation:
            self.ctr = 0
            v = x
        else:
            v = None                    
        return v        
    
    
class PerlinNoise:
    def __init__(self, octaves=3, persistence=0.5, scale=1, freq_scale=1):
        self.freq_scale = freq_scale
        self.scale = scale
        self.persistence = persistence
        self.octaves = octaves
        
    def hash(self, k):
        pass
        
    def new_sample(self, x):
        k = x*self.freq_scale
        power = scale
        s = 0
        for octave in self.octaves:
            k = k / 2.0            
            power = power / 2.0
            ki = math.floor(k)
            kf = k - ki
            a1 = self.hash(ki)
            a2 = self.hash(ki+1)
            f = kf*a1 + (1-kf)*a2
            s = s + f * power
        return s
            
        
    
    
class PLL:
    #phase locked loop
    def __init__(self, bp_filter, lp_filter, centre_freq, sr, loop_gain=1.0, in_gain=1.0):
        self.sr = sr
        self.base_rate = (centre_freq*2*math.pi) / sr
        self.int = 0
        self.bp_filter = bp_filter
        self.lp_filter = lp_filter
        self.phase_buffer = Ringbuffer(int(sr/centre_freq))
        self.loop_gain = loop_gain
        self.in_gain = in_gain
        
                
        
    def set_gain(self, gain):
        self.gain = gain
        
        
    def set_base_freq(self, freq):
        self.base_rate = (freq*2*math.pi) / self.sr
        
        
    def get_sync(self):
        return abs(self.phase_buffer.get_sum())
        
        
        
    def new_sample(self, x):
        (osc,int,y) = self.run(x)
        return int
        
    def run(self, x):                
    
        #band pass the input
        y = self.bp_filter.new_sample(x*self.in_gain)
        
        
            
        # drive the oscillator at natural frequency
        self.int = self.int + self.base_rate
        osc = math.sin(self.int)
        
        #compute phase difference
        phase_com = osc * y
        
        #store phase comparison
        self.phase_buffer.new_sample(phase_com)
        
        # filter the phase comparator
        ph_adj = self.lp_filter.new_sample(phase_com)
        
        # shift the integrator
        self.int = self.int + ph_adj * self.loop_gain
        
        #return value of oscillator, the phase value (unwrapped), and the bp filtered input signal
        return (osc, self.int, y)
        
        
if __name__=="__main__":

    pass
    # sos = [[1, 0, -1, 1, -1.94594030509963, 0.956681300335391],
# [1, -1.99742192067555, 1, 1, -1.98231078518321, 0.989724245837518],
# [1, -1.95356465061816, 1.00000000000000, 1, -1.96887023582848, 0.984916779883396],
# [1, -1.99869468405039, 1.00000000000000, 1, -1.96158991868108, 0.969980685039232],
# [1, -1.90928561493704, 1.00000000000000, 1, -1.94763351342008, 0.961508162903116]]

    # g = [0.00443116147840653,0.00587748761327371,1,0.118821588151725,0.934414851676825,13.0073618266420]
    
    # f = SOSFilter(sos, g)
    # print f.new_sample(1)
    # for i in range(10):
        # print f.new_sample(0)
        
    
    # t = TimeInterpolator(5)
    # for i in range(100):
        # t.add_packet(math.sin(i/50.0), 0.1)
        # print t.get_packets()
    
    
    # t = AsymmetricAcquistion(0.95, 0.1)
    # print t.new_sample(0)
    # for i in range(50):
        # print t.new_sample(50)
    # for i in range(50):
        # print t.new_sample(0)
    
    
    