import math


# BiQuad design based on equations from Robert Bristow-Johnson's cookbook
# http://www.musicdsp.org/files/Audio-EQ-Cookbook.txt


def design_biquad(f0, Q=None, BW=None, S=None, dbGain=0.0, Fs=1.0, type="lpf"):
    if Q==None and BW==None and S==None:
        print "No Q, BW or S specified in filter design! Setting Q=1.0"
        Q = 1.0

   
    A = 10.0**(dbGain/40.0)
    w0 = 2*math.pi*f0/Fs
    cw0 = math.cos(w0)
    sw0 = math.sin(w0)
    
    if Q:
        alpha = sw0/(2*Q)
    if BW:
        alpha = sw0*math.sinh(math.log(2)/2 * BW * w0/sw0)
    if S:
        alpha = sw0/2 * math.sqrt((A+1.0/A)*(1.0/S-1)+2)
        
    tAA = 2*math.sqrt(A)*alpha
        
    if type=="lpf":
        b = [(1-cw0)/2, 1-cw0, (1-cw0)/2]
        a = [1+alpha, -2*cw0, 1-alpha]
    if type=="hpf":
        b = [(1+cw0)/2, -1*(1+cw0), (1+cw0)/2]
        a = [1+alpha, -2*cw0, 1-alpha]
    
    if type=="bpf":
        b = [Q*alpha, 0, -Q*alpha]
        a = [1+alpha, -2*cw0, 1-alpha]
        
    if type=="notch":
        b = [1, -2*cw0, 1]
        a = [1+alpha, -2*cw0, 1-alpha]
        
    if type=="apf":
        b = [1-alpha, -2*cw0, 1+alpha]
        a = [1+alpha, -2*cw0, 1-alpha]
        
    if type=="peaking":
        b = [1+alpha*A, -2*cw0, 1-alpha*A]
        a = [1+alpha/A, -2*cw0, 1-alpha/A]
        
    
    if type=="lowshelf":        
        b = [A*((A+1) - (A-1)*cw0 + tAA), 2*A*((A-1)-(A+1)*cw0), A*((A+1) - (A-1)*cw0 - tAA)]
        a = [(A+1) + (A-1)*cw0 + tAA, -2 * ((A-1) + (A+1)*cw0), (A+1)+(A-1)*cw0 - tAA]
    
    if type=="highshelf":        
        b = [A*((A+1)+(A-1)*cw0 + tAA), -2*A*((A-1)+(A+1)*cw0), A*((A+1)+(A-1)*cw0-tAA)]
        a = [(A+1)-(A-1)*cw0 + tAA, 2*((A-1)-(A+1)*cw0), (A+1) - (A-1)*cw0 - tAA]
        
    a0  = a[0]
    b = [bv/a0 for bv in b]
    a = [av/a0 for av in a]
    return b,a
    
    
# nb this test function uses numpy/pylab/scipy
# the filter design code does not depend on numpy etc.
figure_index = 0
def test_filters():
        import numpy, pylab, scipy.signal
        
        def test_filter(title, type, Q=1, dbGain=0.0):
        
            global figure_index
            figure_index =figure_index+1            
            b1,a1 = design_biquad(0.25, Q=Q,type=type, dbGain=dbGain)
            w,h = scipy.signal.freqz(b1,a1)
            pylab.subplot(4,4,figure_index)
            pylab.title(title)
            pylab.plot(w/(math.pi*2),(numpy.log(numpy.abs(h)+1e-8)/math.log(10.0))*20)
            pylab.axis([0, 0.5, -80, 20])

        
        test_filter("lowpass", "lpf")
        test_filter("highpass", "hpf")
        test_filter("bandpass", "bpf")
        test_filter("bandpass hiQ", "bpf", Q=5)
        test_filter("notch", "notch", Q=1)
        test_filter("notch hiQ", "notch", Q=5)
        test_filter("allpass", "apf", Q=0.1)
        test_filter("peaking", "peaking", Q=1, dbGain=12.0)
        test_filter("peaking low", "peaking", Q=1, dbGain=-12.0)
        test_filter("highshelf", "highshelf", Q=1, dbGain=-12.0)
        test_filter("lowshelf", "lowshelf", Q=1, dbGain=-12.0)
        
        
        pylab.show()
    
        
        
    
if __name__=="__main__":
    test_filters()