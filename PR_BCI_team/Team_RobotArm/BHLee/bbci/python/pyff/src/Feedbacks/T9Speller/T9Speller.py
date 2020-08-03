# -*- coding: latin-1 -*-
import SubjectSimulator.SubjectSimulator
import AdaptiveSequenceGenerator, SequenceGenerator
import time, logging, os, sys,  pygame
import TrialPresentation
import logging.handlers, inspect
import py9.BCIpy9interface #, msvcrt, time, os.path
from numpy import random, mod, arange, array
import jpype, MaryClient
#import BCIpy9interface

from FeedbackBase.MainloopFeedback import MainloopFeedback


class T9Speller(MainloopFeedback):
    def on_init(self):
        self.logger = logging.getLogger("FeedbackController")
        console = logging.StreamHandler()
        self.logger.addHandler(console)
        self.logger.setLevel(logging.DEBUG)
        self.logger.info('initialzation!')
        
        self.fbPath = os.path.dirname( os.path.abspath(__file__)) #directory of THIS file 
        sys.path.append(self.fbPath)
        self.dictFile = os.path.abspath(self.fbPath+"\\py9\\DE-DE-SMALLextracted.dict")
       #dictFile = "C:/svn/bbci/python/pyff/src/Feedbacks/T9Speller/py9/DE-DE-SMALL.dict"
        self.p9 = py9.BCIpy9interface.BCIpy9interface(dict = self.dictFile)
        #self.keysToSpell = py9.py9.getkey(self.lettersToSpell)
        
        self.clock = pygame.time.Clock()
        
        self.loadAllParameterFiles()
        #self.loadParameterSet("paraAudiolab")
        
        
            
            
        self.keysToSpell = random.permutation(self.keysToSpell)
        if self.speechMode:
            self.maryClient = MaryClient.MaryClient(self.maryDir)

       
        
    def startOfTrial_tick(self):
        """
        starts a trial:
            in calibration mode, chooses target cue and does the priming!
        """
        if (not self.spellerMode):
            #set target stimulus sothat the deviant marker is sent!
            self.currentTargetStim = self.keysToSpell[0]
            self.send_parallel(int(self.BEFORE_CUE) )
            self.presentation.start_visual_cue(self.currentTargetStim)
            primingCountdown = self.numPriming            
            while (primingCountdown > 0):
                #first priming --> highlight the marker!
                self.send_parallel(int(self.CUE) )
                self.presentation.start_substim(self.currentTargetStim)
                pygame.time.delay(self.PRIMING_ISI)
                primingCountdown -= 1
            self.presentation.stop_visual_cue(self.currentTargetStim)
            self.send_parallel(int(self.AFTER_CUE) )
        self.send_parallel(int(self.START_TRIAL) )#send marker!

        self.startOfTrial = False
    
    def endOfTrial_tick(self):
        self.send_parallel (int(self.END_TRIAL))
        self.keysToSpell = self.keysToSpell[1:] #remove the first element
        if self.spellerMode:
            self.startOfTrial = True
        
        else: #calibration mode
            print("ask4counts, correct was ", (self.SequenceGenerator.nRandomSeqBefore + self.SequenceGenerator.numMarkerSequences))
            self.presentation.printTextInHeader("Wie viele Praesentationen?")
            self.primingCountdown = self.numPriming
            if not self.simulate_sbj and self.pause_after_calibTrial : self.paused = True 
            #set the system pasued and wait for the answer (unpause with on_interaction_event !!)
            else:
                pygame.time.delay(7000) #make a short pause between 2 trials
                self.presentation.deleteTextInHeader()
            
            self.startOfTrial = True

                
    def main_loop(self):
        self.logger.info(time.strftime('%x %X') + "--> start main loop")
        self.presentation.deleteTextInFooter()
        #self.presentation.printTextInFooter("geschriebener Text", " ")

        if self.speechMode:
            while self.maryClient.isBusy():
                pygame.time.wait(100)
        while 1:
            if len(self.keysToSpell) == 0 and not self.spellerMode:
                break
            if self.simulate_sbj :
                singleKey =self.keysToSpell[0]
                self.subject.setTarget(int(singleKey))
            self.initSequenceGenerator()              
            if self.spellerMode and self.speechMode:
                if self.in_mode1:
                    self.readStdSpelling()
                elif self.in_mode2: #in mode1 !!
                    self.readSuggestions(self.suggestions)
                elif self.in_mode3:
                    self.readTemplates()

            while (self.SequenceGenerator.keepTrialing() or self.paused) and (not self.stopping):
                self.tick()
            if self.spellerMode:
                if not self.SequenceGenerator.CriteriumFulfilled and not self.EARLY_STOPPING:
                    #max number of subtrials presented! 
                    self.logger.info("\n \n wait a while untill the last remaining cl_outs arrive")
                    pygame.time.delay(400)
                    pygame.time.delay(400)
                    pygame.time.delay(400)
                    #self.logger.info("finished waiting")
                self.numDecisions += 1
                proposedKey = self.SequenceGenerator.tellDecision()[0]
                if self.simulate_sbj : self.logger.info("real key:" +str(singleKey) + ", proposed key:" + str(proposedKey) + ",  nTrials:" + str(self.SequenceGenerator.iSubtrial) )
                self.logger.info(time.strftime('%x %X') + "--> DECISION: " + str(proposedKey) + " after " + str(self.SequenceGenerator.iSubtrial) + " Subtrials and "+ str(self.SequenceGenerator.numResponses) + " cl_outs")
                self.onlineLogger.info(time.strftime('%x %X') + "--> DECISION: " + str(proposedKey) + " after " + str(self.SequenceGenerator.iSubtrial) + " Subtrials and "+ str(self.SequenceGenerator.numResponses) + " cl_outs")
                self.logger.info("SubtrialSeq:")
                self.logger.info(self.SequenceGenerator.seq)
                self.logger.info("ResponseContainer:")
                self.logger.info(self.SequenceGenerator.responseContainer)
                self.logger.info("p_decisions:")
                self.logger.info(self.SequenceGenerator.p_decision)
                self.logger.info(self.SequenceGenerator.responseContainer)
                self.send_parallel(int(self.DECISION_SHIFT + proposedKey))
                if self.speechMode:
                    pygame.time.wait(1500)
                    self.readAndWait(u"ausgewählt wurde")
                    pygame.time.wait(200)
                    self.presentation.start_auditory_cue(proposedKey)
                    
                    
                    
                self.handleNewKey(proposedKey)
                if mod(self.numDecisions, self.PAUSE_INTERVAL)==0 and not self.simulate_sbj :
                    #4th decision done, make a short pause!
                    self.on_pause()
                    
            if self.stopped or self.stopping:
                break

    def handleNewKey_mode1(self, proposedKey):
        if proposedKey == 1:
                if self.p9.foundValidWord():
                    self.suggestions = self.p9.giveSuggestions()
                else: self.suggestions = []
                self.selectiontext = u'Auswahl Menü'
                self.gotoMenu2(self.suggestions)
        else: #normal (spelling) in mode1 
            #press the number and spell it into the system
                self.p9.handleInput(str(proposedKey))
                self.selectiontext = self.convertToStringWithSpaces(self.presentation.stdAlphabetLegend[proposedKey])
                if (not self.p9.foundValidWord()) and (len(self.p9.giveKeys()) > len(self.p9.giveSuggestions()[0]) and proposedKey != 1):
                    # catch the case when there is no word fitting anymore (only accept BACK and EXIT)   
                    self.selectiontext = self.selectiontext + u' ist eine ungültige Eingabe. bitte wiederholen!'  
                    self.p9.handleInput("D") #previous command was not accepted --> remove last key!

    
    def handleNewKey_mode2(self, proposedKey):
        self.logger.debug("handleNewKey_mode2 ("+str(proposedKey) +")!!")
        if proposedKey == 1: #ENTER/Speichern
            #todo with removing the space
            #OLD Vesion: write a DOT "." and puts it in the scrolledList
            currtext = self.p9.gettext()
            if currtext[-1] == "|":
                self.p9.handleInput("D")
                self.p9.handleInput("0")
                self.p9.handleInput("1")
                if self.speechMode:
                    self.readAndWait(u"gespeichert")
                self.selectiontext = "ENTER"
                self.logger.info("ENTER EINGEGEBEN!!")
                self.handleNewLine()
                self.gotoMenu1()
            else:
                self.selectiontext = u"Eingabe ungueltig"  

                
        if proposedKey == 3: #DELETE
            self.selectiontext = u'Löschen'          
            currtext = ""
            gotoMenu2 = False
            try:
                currtext = self.p9.getRawText()
                if (currtext[-1] == " ") and (currtext[-2] != " "):
                    #delete a space and go to menu2  
                    gotoMenu2 = True
            except:
                self.logger.info(str(time.strftime('%x %X')) + "could not access last 2 digits of currtext")
                
            self.p9.handleInput("D")
            if gotoMenu2:
                #a space was deleted--> go to Menu2
                self.p9.handleInput("U") #needed to select the last word!
               
                if len(self.p9.giveSuggestions()) > 1: #there are several suggestions, reset them by deleting and adding the last code!
                    lastKeyOfWord = int(self.p9.giveKeys()[-1])
                    self.p9.handleInput("D")
                    self.p9.handleInput(str(lastKeyOfWord))
                    #print "changing the word to startup"

                if self.p9.foundValidWord():
                    self.suggestions = self.p9.giveSuggestions()
                else: suggestions = []
                self.gotoMenu2(self.suggestions)
            else: #go to Menu1 = False
                self.gotoMenu1()


        if proposedKey == 2: #BACK to menu1
            self.gotoMenu1()
            self.selectiontext = "Zurück"
                
                
        if (self.p9.foundValidWord()) and (4<=proposedKey) and \
        (proposedKey <=8) and (len(self.p9.giveSuggestions())>=(proposedKey-3)): 
            #suggestions
            tmp_save = 0
            for i in range(proposedKey-4):
                self.p9.handleInput("U")  #choose the correct suggestion by 
                tmp_save +=1
            self.selectiontext = self.p9.giveSuggestions()[tmp_save]      
            self.p9.handleInput("1")      #pressing "U" the correct number of time
            self.gotoMenu1()
            
            #it is always the first suggestion [0] since we already 
            #changed the order by 'pressing' "U".
        
        elif (len(self.p9.giveSuggestions()) < (proposedKey-3)) and (4<=proposedKey) and (proposedKey <=8): #suggestions
            self.selectiontext = "Wort nicht gefunden, nochmalige Eingabe"  
            print "this number doesnt stand for a word!"
        
        elif proposedKey == 9: #go to Menu3 --> Templates
            self.presentation.gotoMenu3()
            
            #self.stopping = True
            #quit
#        self.p9.giveSuggestions()
    
    def handleNewKey_mode3(self, proposedKey):   
        self.logger.debug("handleNewKey_mode3 ("+str(proposedKey) +")!!") 
        if proposedKey == 9:
            if self.speechMode:
                self.readAndWait(u"Vorlagen")
                pygame.time.wait(500)
                self.readAndWait(u"Eingabe Menü")
            self.gotoMenu1()
        else:
            selected_word = self.templateTexts[proposedKey]
            newLine = True
            if proposedKey==8:
                #Pause, TODO
                pass
    def handleNewLine(self):
        newText = self.p9.gettext()[:-1]#-1 because the last char is a '|'b which comes from the T9
        self.presentation.newParagraph(newText) 
        self.logger.info('new paragraph entered: \n' + newText)
        self.p9.__init__(self.dictFile)

                
    def handleNewKey(self, proposedKey):
        """
        handles the usage entered key (proposedKey), considering the mode (mode1/mode2) and the 
        status (unknown word). Unknown words are not accepted! 
        Also this method prints the spelled text in the footer!
        """ 
        self.logger.debug("handleNewKey ("+str(proposedKey) +")!!")
        self.selectiontext = ''
        self.presentation.deleteTextInFooter()  
        self.presentation.start_auditory_cue(proposedKey)
        time.sleep(0.5)
        self.presentation.printTextInHeader("Erkannt wurde " + str(proposedKey) + ": " + self.selectiontext )
        self.presentation.start_visual_cue(proposedKey)
        
        
        if self.in_mode1:
            self.handleNewKey_mode1(proposedKey)    
        elif self.in_mode2:
            self.handleNewKey_mode2(proposedKey) 
        elif self.in_mode3:
            self.handleNewKey_mode3(proposedKey)
        
        print self.selectiontext
        self.readAndWait(self.selectiontext) #self.selectiontext was modified in self.handleNewKey_modeX
        
        self.logger.info(str(time.strftime('%x %X')) + "--> proposed phrase: " + self.p9.gettext() )
        print "proposed phrase: ", self.p9.gettext()

        time.sleep(1./1000.*self.TIME_DECISION_PRESENTATION)
        
        self.presentation.stop_visual_cue(proposedKey)
        self.presentation.deleteTextInHeader()
        self.presentation.printTextInFooter(self.p9.gettext() )

                            
            
    def tick(self):
        """
        One tick of the main loop.
        
        Decides in which state the feedback currently is and calls the appropriate
        tick method.
        """
        self.presentation.manageEvents()
        if self.speechMode:
                while self.maryClient.isBusy():
                    pygame.time.wait(100)
        if self.stopping:
            pass #go back to main_loop
        elif self.paused:
            #self.logger.info("paused")
            pygame.time.delay(20)
        elif self.endOfTrial:
            self.endOfTrial_tick()
        elif self.startOfTrial:
            self.startOfTrial_tick()
            pygame.time.delay(self.PRIMING_SPELLING_OFFSET)                
            self.elapsed = self.clock.tick(self.FPS)
        else:                        
            ##new with fps
            self.elapsed = self.clock.tick(self.FPS)
            self.cummulatedTime += self.elapsed
            if self.cummulatedTime == self.elapsed and self.cummulatedTime>0:
                #start a Subtrial!
                #obtain next subtrial
                self.currSubtrial =  self.SequenceGenerator.giveTrial()
                if self.SequenceGenerator.sendMarker():
                    #the first few sequences are presented without markers 
                    self.send_marker(int(self.currSubtrial))
                self.presentation.start_substim(self.currSubtrial)
                if not self.SequenceGenerator.keepTrialing():
                    #last Subtrial given
                    self.endOfTrial_tick()
            else:
                if self.cummulatedTime >= self.STIM_TIME and ((self.cummulatedTime - self.elapsed) < self.STIM_TIME):
                    #end subtrial
                    self.presentation.stop_substim(self.currSubtrial)
                    
                    #obtain and manage the simulated response! This has to be done 
                    #somewhere else in the real experiment!
                    if self.simulate_sbj :
                        resp = self.subject.respond(self.currSubtrial)
                        self.SequenceGenerator.manageResponse(resp, self.currSubtrial)
                elif (self.cummulatedTime >= self.ISI): #and ((self.cummulatedTime - self.elapsed) < self.ISI):
                    # End subtrial
                    self.cummulatedTime = 0

        
    def send_marker(self, istim):
        self.logger.debug("send_marker ("+str(istim) +")!!")
        if (istim == self.currentTargetStim):
            self.send_parallel(int(self.DEVIANT_SHIFT+istim)) #11..19 for dev trials
        else:
            self.send_parallel(int(istim)) # 1..9 for std trials
            
      
    def on_interaction_event(self, data):
        """
        handles interaction events that are sent from e.g. Matlab 
        or in other ways over the PYFF Framework.
        Following interaction events are implemented:
         - enable spellerMode
         - receive the number of counts (calibration run) and 
           write them into the log-file and as marker (100+diff) 
        """
        self.logger.info("on_interaction_event, data" + str(data))
            
        if data.has_key(u'spellerMode'):
            if data[u'spellerMode']:
                self.logger.info(str(time.strftime('%x %X')) + "Speller Mode was activated")
                self.spellerMode = True
            else:
                self.logger.info(str(time.strftime('%x %X')) + "Speller Mode was deactivated")
                #self.spellerMode = False #maybe to debug
                
        elif data.has_key(u'numCounts') and self.paused:
            #waiting for the counts
            counts = data[u'numCounts']
            self.logger.info(time.strftime('%x %X') + ": target was " +str(self.currentTargetStim) + "subject counted " + str(counts) + " out of " + str(self.SequenceGenerator.nRandomSeqBefore + self.SequenceGenerator.numMarkerSequences))
            diff = int(counts - (self.SequenceGenerator.nRandomSeqBefore + self.SequenceGenerator.numMarkerSequences))
            self.numCounts = 0
            self.send_parallel(int(100+diff)) 
            self.presentation.deleteTextInHeader()
            
            if len(self.keysToSpell) == 0:
                #send the last marker and stopp the program 
                self.stopping = True
            self.presentation.deleteTextInHeader()
            self.paused = False
            #send 100 if counted number was correct 
            #and e.g. 98 if sbj counted 2 subtrials short!
        

    def on_control_event(self, data):
        """
        handles control events from PYFF
           here, the data of the online Classification is 
           received and transmitted to the SequenceGenerator!         
        """
        self.logger.info("on_control_event, data:" + str(data))       
        if data.has_key(u'cl_output'):
            #classification output was sent
            score_data = data[u'cl_output']
            cl_out = score_data[0]
            iSubstim = int(score_data[1]) # evt auch "Subtrial"
            self.SequenceGenerator.manageResponse(cl_out, iSubstim)
            self.onlineLogger.info(str(self.SequenceGenerator.iSubtrial) + " " + str(iSubstim) + " " + str(cl_out) )
        
         
        
    def on_quit(self):
        """ 
        quits the main loop 
        """
        self.logger.info(str(time.strftime('%x %X')) + "on_quit")
        self.stopping = True
        pygame.time.wait(100)
        while not self.stopped:
            pygame.time.wait(100)
        self.stopped = True
        
        
    def on_pause(self):
        """ 
        pause or unpause! 
        """
        if self.paused:
            self.send_parallel(int(self.PAUSE_STOPP))
            self.presentation.deleteTextInHeader()
        else: 
            self.send_parallel(int(self.PAUSE_START))
            self.presentation.printTextInHeader("Pause")
        self.paused = not self.paused
    
    def on_play(self):
        """
         starts the software:
         - loggers are set up
         - presentation class is instanciated
         - sbjsimulator is instanciated if necessary
         - DefaultParameters are loaded if (not self.manualParameters)
         - mail_loop() is called (endless loop untill program stops) 
         - pygame is closed
        """
        
        logHandler = logging.handlers.RotatingFileHandler(\
              self.LOG_FILENAME, maxBytes=0)
        self.logger.addHandler(logHandler)
        self.logger.setLevel(logging.DEBUG)
        print "logger saving to ", self.LOG_FILENAME
        self.onlineLogger = logging.getLogger("OnlineData")
        logHandler2 = logging.handlers.RotatingFileHandler(\
              str(self.LOG_FILENAME + 'online'), maxBytes=0)
        self.onlineLogger.addHandler(logHandler2)
        
        
        self.logger.info(time.strftime('%x %X') + "--> on_play")
        if self.adaptiveSequence and not self.spellerMode:
            self.logger.critical("adaptive Sequence has to be in the speller mode!")
        
        
        #Instanciate the trial-presentation (sound output) class 
        self.presentation = TrialPresentation.TrialPresentation(self.screenPos, self.sounds, self.templateTexts, self.buttonLabels)
        
        
        pygame.time.wait(2000)
        
        if self.simulate_sbj :
            SbjSimulationDataDir = os.path.abspath(self.fbPath+"\\SubjectSimulator\\")
            self.subject = SubjectSimulator.SubjectSimulator.SubjectSimulator(realSubjectID="VPll", data_dir = SbjSimulationDataDir)
            self.keysToSpell = py9.py9.getkey("im") + "14" + "11" + "2951"+ py9.py9.getkey("Auswahl")
              
        if self.startMode == 1:
            self.gotoMenu1()
        elif self.startMode == 2:
            self.gotoMenu2()
        elif self.startMode == 3:
            self.gotoMenu3()
        #some output in console and logfile for better usabilty
        ttmp = "PARAMETERS:\n spellerMode: "+ str(self.spellerMode)+ \
                         "\n simulate_sbj : "+ str(self.simulate_sbj )+ \
                         "\n keysToSpell: "+ str(self.keysToSpell)+ \
                         "\n MIN_MARKER_SEQ: " + str(self.MIN_MARKER_SEQ)+  \
                         "\n N_MARKER_SEQ: " + str(self.N_MARKER_SEQ)+ \
                         "\n MAX_NONMARKER_SEQ: " + str(self.MAX_NONMARKER_SEQ)+ \
                         "\n EARLY_STOPPING: " + str(int(self.EARLY_STOPPING))+ \
                         "\n DECISION_CRITERION: " + str(self.DECISION_CRITERION) +\
                          "\n ADAPTIVE_SEQUENCE: " + str(self.adaptiveSequence) + "\n \n" 
        self.logger.info(ttmp)
                        
        
        self.send_parallel(int(self.START_EXP))
        if self.speechMode:
            self.readAndWait("Willkommen im System")
        self.main_loop()
        self.send_parallel(int(self.END_EXP))
        self.stopped = True
        
        jpype.shutdownJVM()
        pygame.quit()
        self.logger.info("pygame was closed")
        
    def get_variables(self):
        pass
    
    def loadParameterSet(self, fname):
        """
        loads all Variables which are specified in fname.py into the T9Speller class.
        Everything is overwritten without questioning!
        ###############################
        Example:
        ex1.py contains:
            ISI = 200
            dump = True
        
        within the T9Speller class, self.loadParameterSet('ex1') leads to
            self. ISI = 200
            self.dump = True
        """        
        para = __import__(fname)

        tmp = inspect.getmembers(para)
        for i in range(len(tmp)):
            if tmp[i][0][:2] != '__':
                self.__setattr__(tmp[i][0], tmp[i][1])   
        
    
    def initSequenceGenerator(self):
        """
        initiatiate the SequenceGenerator (adaptive or standard)
        """
        if not self.adaptiveSequence:
            #standard SequenceGenerator
            self.SequenceGenerator = SequenceGenerator.SequenceGenerator( \
                minDistance=self.MIN_DISTANCE, nstim=9, minnumMarkerSequences=5, \
                decisionCriterion = self.DECISION_CRITERION, numMarkerSequences=self.N_MARKER_SEQ, \
                maxNonmarkerSeq=self.MAX_NONMARKER_SEQ, earlyStopping =self.EARLY_STOPPING)
        else:
            #adaptive SequenceGenerator
            self.SequenceGenerator = AdaptiveSequenceGenerator.AdaptiveSequenceGenerator(\
                minDistance=self.MIN_DISTANCE, nstim=9, constantShift=0.1, \
                minPresentation=5, decisionCriterion = self.DECISION_CRITERION,\
                maxNumSubtrials=(self.N_MARKER_SEQ * 9))
            print str(locals())
            print str(globals())

        try:
            self.GLOBAL_SEQ
        except:
            pass # `var1' does not exist
        else:
            pass # `var1' exists
            self.SequenceGenerator.seq = self.GLOBAL_SEQ
            self.SequenceGenerator.maxNumSubtrials = len(self.GLOBAL_SEQ)
            self.logger.warning("a manually defined sequence was taken")
            
    
    def readSuggestions(self, suggestions):
        """
        reads the choices and presents the stimuli in mode2:
            (SAVE, BACK TO SPELLING MENU, DELETE,  .... CHOICES ....  TEMPLATES)
        """
        self.readAndWait(u"Sie sind im Auswahl Menü")
        pygame.time.wait(6*self.DELAY_SPEECH_SYNTHESIS)
        ii=1
        txtInDisplay = [u"speichern", u"Eingabe Menü", u"löschen"] + self.presentation.alphabetLegend[4:9] + [u"Vorlagen Menü"]
              
        for wrd in txtInDisplay:
            if wrd != "":
                self.readAndWait(wrd)
                self.presentation.start_auditory_cue(ii)
                time.sleep(4./1000.*self.DELAY_SPEECH_SYNTHESIS)
            ii += 1;


    def readStdSpelling(self):
        """
        reads the letters of each class and presents the stimuli in mode1:
        """
        if self.speechMode:
            self.readAndWait(u"Sie sind im Eingabe Menü")
            pygame.time.wait(6*self.DELAY_SPEECH_SYNTHESIS)
            self.readAndWait(u"Auswahl Menü")
        self.presentation.start_auditory_cue(1)

        for ii in arange(2,10):
            time.sleep(5./1000.*self.DELAY_SPEECH_SYNTHESIS)
            if self.speechMode:
                txtToRead = self.convertToStringWithSpaces(self.presentation.stdAlphabetLegend[ii])            
                self.readAndWait(txtToRead)
            #time.sleep(1./1000.*self.DELAY_SPEECH_SYNTHESIS)
            self.presentation.start_auditory_cue(ii)
              
            
    def readTemplates(self):
        """
        reads the templates and presents the stimuli in mode3:
        """        
        self.readAndWait(u"Sie sind im Vorlagen Menü")
        pygame.time.wait(6*self.DELAY_SPEECH_SYNTHESIS)
        for ii in arange(1,9):
            self.readAndWait(self.templateTexts[ii-1])
            self.presentation.start_auditory_cue(ii)
            time.sleep(5./1000.*self.DELAY_SPEECH_SYNTHESIS)
        
        self.readAndWait(u"Menü")
        self.presentation.start_auditory_cue(9)
        time.sleep(10./1000.*self.DELAY_SPEECH_SYNTHESIS)
        
    def readAndWait(self, txt):
        """
        reads a given text and waits untill the reading is finished ( ~ pygame queue is empty)
        """
        if self.speechMode:
            self.maryClient.processAndPlay(txt)
            while self.maryClient.isBusy():
                time.sleep(.01)
        return
    
    def convertToStringWithSpaces(self, txt):
        """
        converts 'string' to 's t r i n g '
        """
        tmptxt = u''
        for s in txt: #read the chars of each key followed by the cue! 
            tmptxt = tmptxt + s + ' '
        return tmptxt

                    
    def gotoMenu1(self):
        """
        sets the mode parameter to mode1 and updates the GUI
        """
        self.in_mode1 = True
        self.in_mode2 = False
        self.in_mode3 = False
        self.presentation.show_mode1()
        
    def gotoMenu2(self, suggestions=[]):
        """
        sets the mode parameter to mode2 and updates the GUI
        """
        self.in_mode1 = False
        self.in_mode2 = True
        self.in_mode3 = False
        self.presentation.show_mode2(suggestions)
    
    def gotoMenu3(self):
        """
        sets the mode parameter to mode3 and updates the GUI
        """        
        self.in_mode1 = False
        self.in_mode2 = False
        self.in_mode3 = True
        self.presentation.show_mode3()
    
    def loadAllParameterFiles(self):
        flist = os.listdir(self.fbPath)
        paraList = ["StdParameters"]
        for f in flist:
            if (f[0:4] == "para") and (f[-3:] == ".py"):
                paraList.append(f[:-3])

          
        for f in paraList:
            self.loadParameterSet(f)
            self.logger.debug("Parameter File " + f + " loaded!")
            
        
if __name__ == "__main__":
    #import logging
    #import SubjectSimulator 
    #logging.basicConfig(level=logging.DEBUG)
    sim = T9Speller()
    sim.on_init()
    sim.spellerMode = True#False
    sim.simulate_sbj  = True
    sim.logger.critical("This is a Testrun which was started on console and standard parameters!!!")
    sim.on_play()   
    
    
     
