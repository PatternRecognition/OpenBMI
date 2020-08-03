from numpy import arange, zeros, mod, \
    where, floor, exp, abs, array, min, sqrt, ones, square,\
    shape, random, argsort, mean, std, cumsum, var, ceil
from scipy.stats import distributions
from scipy import stats
import logging


class SequenceGenerator:       
    
    def __init__(self, minDistance=3, nstim=9,  minnumMarkerSequences=3,\
                  decisionCriterion = 0.0001, numMarkerSequences=14, maxNonmarkerSeq=3,\
                  earlyStopping = False, indivStoppingThresholds = []):
        self.minDistance = minDistance
        self.nRandomSeqBefore = ceil(random.rand() * maxNonmarkerSeq)
        self.numMarkerSequences = numMarkerSequences        
        self.earlyStopping = earlyStopping
        self.nstim = nstim
        self.CriteriumFulfilled = False #earlystopping criterium
        self.numResponses = 0 #number of cl_outs received already
        self.minnumMarkerSequences = minnumMarkerSequences
        self.decisionCriterion = decisionCriterion
        self.seq = self.computeRandomSeq2(self.numMarkerSequences + self.nRandomSeqBefore)
        self.iSubtrial = 0 #running index
        self.maxNumSubtrials = self.nstim * (self.nRandomSeqBefore + self.numMarkerSequences)
        self.responseContainer = [[],[],[],[],[],[],[],[],[]]
        self.p_decision = ones(self.nstim)
        
        print 'setup logger'
        self.logger = logging.getLogger("FeedbackController")
        self.logger.setLevel(logging.DEBUG)
        self.logger.debug('asked for decision: responsecontainer was:') 
        
        print 'test'
        
        if len(indivStoppingThresholds) in [0,nstim]:
            self.indivStoppingThresholds = indivStoppingThresholds
        else: 
            self.logger.warning("individualStoppingThreshold have wrong size!")
            self.indivStoppingThresholds = []
    
    def computeRandomSeq(self, numseq):
        seq = (random.permutation(self.nstim)+1).tolist()
        nseq = 1
        
        while nseq < numseq:
            nextSequence = (random.permutation(self.nstim)+1).tolist()
            for i in arange(self.minDistance):
                if nextSequence[i] in seq[-(self.minDistance-i):]:
                    #swap with another subtrial which is not 'locked'
                    for j in arange(self.minDistance-1, self.nstim):
                        if not (nextSequence[j] in seq[-(self.minDistance+i):]):
                            #swap
                            tmp = nextSequence[i]
                            nextSequence[i] = nextSequence[j]
                            nextSequence[j] = tmp
                            break # break j loop
            seq = seq + nextSequence #concatenate
            nseq += 1
        return(seq)
    
    
    
    
    def computeRandomSeq2(self, numseq):
        """
            Generate a sequence where neighboring subtrials have different pitch! 
            Stimuli with middle orientation (2,5,8) are just presented if the preceding 2 subtrials had a different pitch! 
            Minimum distance between the same subtrial has be fulfilled!!! 
        """
    
        nseq=0
        lastSubtrial=0
        nstim=9#self.nstim
        rows = [[0,1,2],[3,4,5],[6,7,8]]
        
        
        seq = []
        overlap = []
        lastTwo = [] # middle stim is locked for 2 iterations 
        while nseq < numseq:  
            #build 1 sequence
            nextSequence = zeros(nstim).tolist()
            ishown = []
            valid = True
            for i in range(nstim):
                p_i = ones(nstim)
                p_i[overlap] = 0.
                p_i[ishown] = 0.
                if len(lastTwo) > 2: lastTwo = lastTwo[1:]
                if lastSubtrial > 0: 
                    ixrowlock = int(lastSubtrial-1) / int(3)
                    lockedMiddle = int((lastTwo[0]-1) / 3)*3 + 1 
                    #print "test",i, lockedMiddle, lastSubtrial, lastTwo, seq 
                    p_i[rows[ixrowlock]] = 0.
                    p_i[lockedMiddle] = 0. 
                    
                if sum(p_i) == 0.:
                    #not valid seq generated... try again (do not increase nseq)
                    valid = False
                    lastTwo = seq[-2:]
                    if len(seq)>0: 
                        overlap = (array(nextSequence)-1).tolist()[-self.minDistance:]
                        lastSubtrial = seq[-1]
                    else: 
                        overlap = []
                        lastSubtrial=0
                    
                    break
                
                p_i = p_i / sum(p_i) #normalize to 1
                r = random.uniform(0,1)
                
                cumProbs = cumsum(p_i)
                thisSub = where(cumProbs > r)[0][0] + 1 #new trial: int i..9 
                ishown.append(thisSub-1)
                lastTwo.append(thisSub)
                nextSequence[i] = thisSub
                
                if len(overlap) > 0: overlap = overlap[1:]
                lastSubtrial = thisSub
            
            if valid:
                overlap =  (array(nextSequence)-1).tolist()[-3:]# self.minDistance :]
                seq = seq + nextSequence #concatenate
                nseq += 1
#        print "seq"
#        print seq
        return(seq)
        
            
    def manageResponse(self, response, stim):
        """
            handles classifier outputs and checks if earlystopping criterum is fulfilled (if enabled)
        """
        self.numResponses += 1
        self.responseContainer[stim-1].append(response)
        if (mod(self.numResponses, self.nstim) ==0) \
        and (floor(self.numResponses/self.nstim)>(self.minnumMarkerSequences - self.nRandomSeqBefore) ) \
        and self.earlyStopping:
            #check if conditions for early stopping are fulfilled
            self.logger.info('check for early-Stop')
            self.updateDecisionVars() 
            if len(self.indivStoppingThresholds) != 0:
                if  self.p_decision.min() < self.indivStoppingThresholds:
                    self.logger.info("early stopping, pvalue of" + str(self.p_decision.min())+ " after " + str(self.numResponses) + " cl_outs")
                    self.CriteriumFulfilled = True
            elif self.p_decision.min() < self.decisionCriterion:
                self.logger.info("early stopping, pvalue of" + str(self.p_decision.min())+ " after " + str(self.numResponses) + " cl_outs")
                self.CriteriumFulfilled = True


           
    def setDecisionCriterion(self, critPval):
        self.decisionCriterion = critPval
    
    def giveTrial(self):
        """     
        returns a random trial as integer (1..9), based
        on the current settings
        """        
        subtrial = self.seq[self.iSubtrial]
        self.iSubtrial += 1
        return subtrial
    
    def sendMarker(self):
        return (self.iSubtrial >= (self.nRandomSeqBefore * self.nstim))


    def updateDecisionVars(self):
        for i in range(self.nstim):
            rest = []

            for j in range(self.nstim): #refactor!
                if j!= i:
                    rest.extend(self.responseContainer[j])
            
            pval2 = ttest2_p(self.responseContainer[i], rest)

            self.p_decision[i] = pval2            
            
        
    def keepTrialing(self):
        if self.iSubtrial >= self.maxNumSubtrials or self.CriteriumFulfilled:
            #STOP TRIALING too many subtrials already
            return False
                
        return True #continue
      
        
    def tellDecision(self):
        self.logger.info('asked for decision: responsecontainer was:') 
        self.logger.info(self.responseContainer)
        if self.numResponses > self.nstim:
            self.updateDecisionVars()
            i_decision = self.p_decision.argmin()
            pvalue = self.p_decision.min()
            # sanity chacks --> cpmpare i_decision with the earlyStoppingThresholds 
            if  sum(self.p_decision.min() < self.indivStoppingThresholds)>0 : #threshold was exceeded
                if  sum(self.p_decision.min() < self.indivStoppingThresholds)>1:
                    self.logger.warning('several keys exceeded the threshold. best pvalue chosen!')
                elif where(self.p_decision.min() < self.indivStoppingThresholds)[0][0] != i_decision:
                    self.logger.warning('The key that exceeded the threshold is not the min, min pvalue was taken anyway!')
 
            return [i_decision+1, pvalue]
        else:
            self.logger.critical("Asked for Decision but there is not enough cl_outs recorded yet. RANDOM NUMBER RETURNED!")
            return [random.randint(9)+1,1.]                      
            
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

if __name__ == "__main__":
    ss = SequenceGenerator()
    print ss.computeRandomSeq(150)