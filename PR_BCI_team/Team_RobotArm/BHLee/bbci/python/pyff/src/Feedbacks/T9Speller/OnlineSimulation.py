import SubjectSimulator.SubjectSimulator
import OnlineTrialGeneration
import time, pylab, logging, os, sys, logging, numpy
import AuditoryOddball
import py9.BCIpy9interface #, msvcrt, time, os.path
#import BCIpy9interface



class OnlineSimulation:
    def __init__(self):
        sys.stdout.write('init')
        self.logger = logging.getLogger("FeedbackController")
        #self.clock = pygame.time.Clock()
        self.trialGenerationBot = OnlineTrialGeneration.OnlineTrialGeneration()
        self.subject = SubjectSimulator.SubjectSimulator.SubjectSimulator(realSubjectID="VPll")
        self.logger = logging.getLogger("FeedbackController")
        self.lettersToSpell = "Hello World"
        self.p9 = py9.BCIpy9interface.BCIpy9interface()
        self.keysToSpell = py9.py9.getkey(self.lettersToSpell)

        #self.presentation = AuditoryOddball()
        #self.presentation.on_init()
                
    def main(self):
        self.logger.info("start simulation")
        
        for singleKey in self.keysToSpell:
            self.subject.setTarget(int(singleKey))
            self.trialGenerationBot.startNewBlock()
            while self.trialGenerationBot.keepTrialing():
                #stay within one block --> spelling one number
                stim = self.trialGenerationBot.giveTrial()
                resp = self.subject.respond(stim)
                self.trialGenerationBot.manageResponse(resp, stim)
                #self.logger.info("stim = "+str(stim))
                #self.logger.info("resp = "+str(resp))
            proposedKey = self.trialGenerationBot.tellDecision()[0]
            print "real key:",singleKey,", proposed key:", proposedKey, ",  nTrials:", str(self.trialGenerationBot.numStimInBlock)
            self.p9.handleInput(str(proposedKey))
            print "proposed phrase: ", self.p9.gettext()

        
        
if __name__ == "__main__":
    import logging
    import SubjectSimulator 
    import OnlineTrialGeneration
    logging.basicConfig(level=logging.DEBUG)
    sim = OnlineSimulation()
    sim.main()  
        