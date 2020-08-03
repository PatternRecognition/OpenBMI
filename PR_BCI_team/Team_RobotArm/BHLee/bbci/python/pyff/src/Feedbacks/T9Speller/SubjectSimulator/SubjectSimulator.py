from numpy import random, floor


class SubjectSimulator:
    def __init__(self, targetStimulus = 1, mean_targetResponse = -1.3,  
                 mean_nontargetResponse = 1., var_response= 1.5, realSubjectID="VPll", data_dir=""):
        self.targetStimulus = targetStimulus
        self.realSubjectID = realSubjectID 
        print "SubjectSimulatordir"; print data_dir
        if (self.realSubjectID == ""):        
            self.mean_targetResponse = mean_targetResponse
            self.mean_nontargetResponse = mean_nontargetResponse
            self.var_response = var_response
        else:
#            data_dir = "D:/svn/bbci/python/pyff/src/Feedbacks/T9Speller/SubjectSimulator/"
            
            tmp = open(data_dir +"\\"+ realSubjectID + "\\cl_output_on_Target_Speller.txt",'rt').readline().split()
            d = []
            for x in tmp:
                d.append(float(x))
            self.targetPool = d
            
            tmp = open(data_dir +"\\"+ realSubjectID + "\\cl_output_on_nonTarget_Speller.txt",'rt').readline().split()
            d = []
            for x in tmp:
                d.append(float(x))           
            self.nontargetPool =  d
    
    def setTarget(self, newtarget):
        self.targetStimulus = newtarget
        
    def respond(self, stim):
        if (self.realSubjectID == ""):
            if stim == self.targetStimulus:
                return(random.normal(self.mean_targetResponse,self.var_response)) #return a random number
            else:
                return(random.normal(self.mean_nontargetResponse,self.var_response)) #return a random number
        else:
            if stim == self.targetStimulus:
                return(self.targetPool[int(floor(random.rand()*len(self.targetPool)))]-3.4)  #just for testing
            else:
                return(self.nontargetPool[int(floor(random.rand()*len(self.nontargetPool)))])
       
  
