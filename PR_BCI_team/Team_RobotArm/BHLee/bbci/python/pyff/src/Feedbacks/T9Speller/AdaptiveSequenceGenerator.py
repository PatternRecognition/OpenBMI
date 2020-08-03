from numpy import arange, zeros, mod, \
    where, exp, abs, array, min, sqrt, ones, square,\
    shape, random, argsort, mean, std, cumsum, var
from scipy.stats import distributions
from scipy import stats
import logging


class AdaptiveSequenceGenerator:       
    
    def __init__(self, minDistance=3, nstim=9, constantShift=0.2, minPresentation=3, decisionCriterion = 0.0001, maxNumSubtrials=200, earlyStopping = False):
        self.logger = logging.getLogger("FeedbackController")
        self.minDistance = minDistance
        self.nstim = nstim
        self.p_std = zeros(self.nstim) + 1./self.nstim
        self.constantShift = constantShift
        self.minPresentation = minPresentation
        self.maxNumSubtrials = maxNumSubtrials
        self.startNewBlock()
        self.numResponses = 0 #number of cl_outs received already        
        self.decisionCriterion = decisionCriterion
        self.CriteriumFulfilled = False
        self.seq = []
        self.rows = [[0,1,2],[3,4,5],[6,7,8]]
        self.earlyStopping = earlyStopping

    def startNewBlock(self):
        self.responseContainer = [[],[],[],[],[],[],[],[],[]]
        self.p_decision = ones(self.nstim)
        self.lockedSubtrials = [] #stimuli which are currently locked and their...
        self.iSubtrial = 0
        self.setDefaultProbs()
        self.currPermutation = (random.permutation(9)+1).tolist() 
        self.nextPermutation = self.currPermutation
        
    def setDefaultProbs(self):
        self.p = zeros(self.nstim)+ (1./self.nstim)
        
    def sendMarker(self):
        """
            Since the AdaptiveSequenceGenerator is only used in the speller_mode, 
            a marker is sent with every subtrial!
        """
        return True
        
    def updateProbs(self):
        if (self.iSubtrial < (self.minPresentation * self.nstim)): #within the first block
            self.setDefaultProbs()
            
        else: #first complete blocks were presented --> each stimulus was presented at least 'minPresentation' times!
            meanResponses = zeros(self.nstim)
            for i in range(self.nstim):
                meanResponses[i] = mean(self.responseContainer[i])
             
            meanmean = mean(meanResponses)
            diffmean = meanResponses - meanmean
            tmp = exp((-1.*diffmean / (1.*std(meanResponses))))
            tmp += self.constantShift
            
            self.p = tmp * self.p_std
            
        #set prob of locked stim to 0
        self.p[array(self.lockedSubtrials)-1] = 0

        #prevent the next tone having the same pitch
        tmpp = self.p
        tmpp[self.rows[ int(self.seq[-1]-1) / 3 ] ] = 0.
        if sum(tmpp) > 0.:
            self.p = tmpp
        
        if sum(self.p) == 0.:
            r = random.randint(9)
            while r in (array(self.lockedSubtrials)-1):
                r = random.randint(9)
            self.p[r] = 1.
            print "TOTALLY RANDOM WAS GENERATED with respect to neihborhoodConstraint!!"
        #normalize to 1
        self.p *= 1/sum(self.p)
        
      
    def manageResponse(self, response, stim=0):
        self.numResponses += 1 
        self.responseContainer[stim-1].append(response)
        
        
        if self.numResponses > (self.minPresentation * self.nstim):
            self.updateProbs()
            #self.updateDecisionVars()
        
        if (self.numResponses > (self.minPresentation * self.nstim) and mod(self.numResponses, self.nstim)==0):
            self.updateDecisionVars() #compute ttests every nstim subtrial
            print "updateDecisionVars", self.numResponses
            
        
    def setDecisionCriterion(self, critPval):
        self.decisionCriterion = critPval
    
    def giveTrial(self):
        """     
        returns a random trial as integer (1..9), based
        on the current settings
        """        
        self.iSubtrial += 1
        #HEURISTIC
        r = random.uniform(0,1)
        cumProbs = cumsum(self.p)
        randTrial = where(cumProbs > r)[0][0] + 1 #new trial: int i..9 
        #set new stimulus 'locked'
        self.lockedSubtrials.append(randTrial)
        self.modifyLockedSubtrials()
        
        self.seq.append(randTrial)
        self.updateProbs()
        #self.logger.info(self.lockedSubtrials)
        return(randTrial)
    
    def modifyLockedSubtrials(self):        
        if (self.iSubtrial >= (self.minPresentation * self.nstim)):
            self.lockedSubtrials = self.lockedSubtrials[1:]
            #delete Stimuli from the locked list
        
        if (len(self.lockedSubtrials) > (self.nstim)):
            self.lockedSubtrials = self.lockedSubtrials[1:]
            print "ALL STIMULI ARE LOCKED ---> sth wrong"
            #prevent full locked lists
        
        # if we presented all stimuli once (after nstim * minpresentation presentations), 
        # all the Trials excect of the last 'minDistance' ones get unlocked
        if ((len(self.lockedSubtrials) == self.nstim) and \
        (self.iSubtrial <= (self.minPresentation * self.nstim))):
            # IF end of iteration   AND 
            # (not enough subtrials for heuristic
            
            #delete all except of the last 'mindistance' ones!
            self.lockedSubtrials = self.lockedSubtrials[(self.nstim-self.minDistance-1):]
            self.setDefaultProbs()
        

    def updateDecisionVars(self):
        for i in range(self.nstim):
            rest = []

            for j in range(self.nstim): #refactor!
                if j!= i:
                    rest.extend(self.responseContainer[j])
            
            pval2 = ttest2_p(self.responseContainer[i], rest)

            self.p_decision[i] = pval2    
        if self.p_decision.min() < self.decisionCriterion and self.earlyStopping:
            self.CriteriumFulfilled = True
            
        
    def keepTrialing(self):
        if (self.iSubtrial >= self.maxNumSubtrials or self.CriteriumFulfilled):
            return False #stop trialing
        else:
            return True # decision was not made so far
        
    def tellDecision(self):
        return [self.p_decision.argmin() + 1, self.p_decision.min()]                      
            
def ttest2_p(v1, v2):
    # ttest with unequal variance, returns the pvalue
    # http://beheco.oxfordjournals.org/cgi/content/full/17/4/688
    n1 = len(v1)+0.
    n2 = len(v2)+0.
    mu1 = mean(v1)
    mu2 = mean(v2)
    s1 = var(v1)
    s2 = var(v2)
    
    t = (mu1-mu2)/sqrt(s1/n1 + s2/n2)
    
    u = s2/s1
    
    df = round(  square(1/n1 + u/n2) / (1/(square(n1)*(n1-1)) + square(u) / (square(n2)*(n2-1)) )  ) 
    
    pval = distributions.t.cdf(t,df)
    return(pval)