import SubjectSimulator.SubjectSimulator
import OnlineTrialGeneration
import time, pylab, logging, os, sys, logging, numpy

class Simulation:
    def __init__(self):
        sys.stdout.write('init')
        self.logger = logging.getLogger("FeedbackController")
        #self.clock = pygame.time.Clock()
        self.trialGenerationBot = OnlineTrialGeneration.OnlineTrialGeneration()
        self.subject = SubjectSimulator.SubjectSimulator.SubjectSimulator()
        self.logger = logging.getLogger("FeedbackController")

                
    def main(self):
        self.logger.info("start simulation")
        psave = numpy.zeros([300,9])
        pDecision_save = numpy.zeros([300,9]) 
        lockedProbs=[]
        stimsave = []
        #while self.trialGenerationBot.keepTrialing():
        for i in range(300):
            stim = self.trialGenerationBot.giveTrial()
            resp = self.subject.respond(stim)
            self.trialGenerationBot.manageResponse(resp, stim)
            #self.logger.info("stim = "+str(stim))
            #self.logger.info("resp = "+str(resp))
            psave[i] = self.trialGenerationBot.p
            pDecision_save[i] = self.trialGenerationBot.p_decision
            stimsave.append(stim)
        #psave[psave<10^(-10)] = numpy.nan
        pylab.subplot(221)
        pylab.plot(range(self.trialGenerationBot.numStimInBlock),numpy.array(psave),'.')
        pylab.legend(numpy.array(range(9))+1)
        pylab.subplot(222)
        pylab.hist(stimsave,9)
        pylab.subplot(224)
        pylab.boxplot(self.trialGenerationBot.responseContainer)        
        pylab.subplot(223)
        pDecisionPlot = -1.*numpy.log10(pDecision_save)
        pylab.plot(pDecisionPlot)
        pylab.show()
        
        
if __name__ == "__main__":
    import logging
    import SubjectSimulator 
    import OnlineTrialGeneration
    logging.basicConfig(level=logging.DEBUG)
    sim = Simulation()
    sim.main()  
        