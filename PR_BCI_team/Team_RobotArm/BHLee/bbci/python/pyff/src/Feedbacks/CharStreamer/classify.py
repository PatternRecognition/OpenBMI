from numpy import mean, var, square, sqrt
from scipy.stats import distributions

import random

### copied from T9Speller
class Test:
    def __init__(self):
        self.n_responses = 0
        self.cl_responses = {}
        self.n_for_iteration = 5
        self.min_iterations = 2
        self.early_stopping = True
        self.p_decisions = {}
        self.p_criterion = 0.05
        self.p_fulfilled = False

        
    def on_control_event(self, data):
        """
        handles control events from PYFF
        here, the data of the online Classification is 
        received and transmitted to the SequenceGenerator!         
        """

        if 'cl_output' in data: #classification output was sent
            cl_out = data['cl_output'][0]
            stim = int(data['cl_output'][1])
            self.manage_response(stim, cl_out)


    def manage_response(self, stim, response):
        """
        handles classifier outputs and checks if earlystopping criterum is fulfilled (if enabled)
        """
        self.n_responses += 1
        self.cl_responses.setdefault(stim, []).append(response)

        d, m = divmod(self.n_responses, self.n_for_iteration)
        if self.early_stopping and m == 0 and d >= self.min_iterations:
            self.update_p_decisions() 
            if min(self.p_decisions.values()) < self.p_criterion:
                        self.p_fulfilled = True
                        print 'YEAH found p'


    def update_p_decisions(self):
        for stim in self.cl_responses:
            rest = reduce(lambda x, y: x+y, [self.cl_responses[s] for s in self.cl_responses if s != stim])
            pval2 = ttest2_p(self.cl_responses[stim], rest)
            self.p_decisions[stim] = pval2            
        print 'update', self.p_decisions


def ttest2_p(v1, v2):
    # ttest with unequal variance, returns the pvalue
    # http://beheco.oxfordjournals. org/cgi/content/full/17/4/688
    n1 = len(v1)+0.
    n2 = len(v2)+0.
    mu1 = mean(v1)
    mu2 = mean(v2)
    s1 = var(v1)
    s2 = var(v2)
    
    t = (mu1-mu2)/sqrt(s1/n1 + s2/n2)
    u = s2/s1
    df = round(square(1/n1 + u/n2) / (1/(square(n1)*(n1-1)) + square(u) / (square(n2)*(n2-1)))) 

    pval = distributions.t.cdf(t,df)
    return(pval)


if __name__ == '__main__':
    print 'TEST'
    print
    t = Test()
    for _ in range(10):
        for i in range(5):
            if i == 3:
                t.on_control_event({'cl_output': [random.random() - 1, i]})
            else:
                t.on_control_event({'cl_output': [random.random(), i]})
        
                                
