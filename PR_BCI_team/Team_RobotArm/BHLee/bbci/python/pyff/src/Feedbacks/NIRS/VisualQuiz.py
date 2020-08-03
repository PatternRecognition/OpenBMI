

import pygame
from pygame.locals import *
import pygame.key as key
import os, time
import numpy as np
import copy
import random as r
from FeedbackBase.MainloopFeedback import MainloopFeedback
from xml.dom.minidom import *
import socket


class VisualQuiz(MainloopFeedback):   
    
    RUN_START,QUIZ_APPEAR,USR_READY,TASK_START,REST_START,OUTPUT_MARKED,CRCT_MARKED,RUN_END, REST_END = 1,2,3,4,5,6,7,8,9 # user input: 10-16

    def init(self):
        
        pygame.init()

        self.geometry = [100, 100, 1440, 900] 
        self.fullscreen = False
        self.bgColor = (0., 0., 0.)
        self.quesColor=(250,250,250)
        self.quesHeight=150
        self.ansColor=(0.,0.,0.)
        self.boxColor=(255,255,100)
        self.boxSize=(300,200)
        self.boxHeight=450
        self.quesFontSize = 80
        self.plusIconSize=100
        self.ansFontSize = 60
        self.borderColor=(255,255,255)
        self.borderSize=(self.boxSize[0]+30,self.boxSize[1]+30)
        self.classBorderColor=(255,45,67)
        self.counterTime=3
        self.counterHeight=300
        self.counterFontSize=120
        self.counterColor=(255,255,100)
        self.iconFontSize= 70
        self.iconColor=(0,0,0)
        self.taskTime=1
        self.restTimeRange=[1,2]
        self.correctColor=(0,255,0)
        self.online=False 
        self.nTrials=20
        self.path = os.path.dirname(globals()["__file__"])
        #if socket.gethostname()=='hadi-Latitude-E6400':
        #    self.path='/home/hadi/bbci/python/pyff/src/Feedbacks/NIRS/'
        #else:
        #    self.path= os.path.abspath(os.path.dirname(globals()["__file__"])+)

        self.musicIconPath=os.path.join(self.path,'music.png')
        self.naviIconPath=os.path.join(self.path,'navi2.png')
        self.plusIconPath=os.path.join(self.path,'plus.png')

        # change this so that triggers and tasks are assigned to eachotehr regradless of the self.tasks order
        self.tasks=["music","navigation","math"]
        self.taskTrigDic = {'music':10,'navigation':11,'math':12}
        self.quizFile='quiz_practice'

        self._triggerResetTime  = 0.2
        
    def initStuff(self):
        self.nClasses=len(self.tasks)
        self.taskTriggers=[self.taskTrigDic[task] for task in self.tasks]
        self.keys=[K_1,K_2,K_3,K_4,K_5,K_6,K_7][0:self.nClasses]
        
        os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (self.geometry[0],
                                                        self.geometry[1])
        self.scrW,self.scrH=self.geometry[2],self.geometry[3]
        self.screen = pygame.display.set_mode((self.scrW,self.scrH))

        self.orderTasks=[]
        nEach=np.floor(self.nTrials/self.nClasses)
        for c in range(self.nClasses):
            self.orderTasks+=[c]*nEach 
        self.orderTasks+=[r.randint(0,self.nClasses-1) for n in range(np.mod(self.nTrials,self.nClasses))]
        r.shuffle(self.orderTasks)
        self.quesFont = pygame.font.Font(None, self.quesFontSize)
        self.ansFont = pygame.font.Font(None, self.ansFontSize)
        self.counterFont=pygame.font.Font(None, self.counterFontSize)
        self.iconFont= pygame.font.Font(None, self.iconFontSize)
        self.xmlPath=os.path.join(self.path,self.quizFile+'.xml')
        self.prepareSyms()
        self.loadXML()
        nQuestions=len(self.allQuestions)
        N=min([self.nTrials,nQuestions])
        self.quesOrder=range(N)
        r.shuffle(self.quesOrder)      
        
        
    
    def loadXML(self):      
        dom=parse(self.xmlPath)
        handleQuiz=dom.getElementsByTagName("quiz")
        self.allAnswers=[]
        self.allQuestions=[str(handle.getAttribute("question")) for handle in handleQuiz]  
        self.corrects=[str(handle.getAttribute("correct")) for handle in handleQuiz]
        for handle in handleQuiz:
            c=0
            anses=[]
            while str(handle.getAttribute("answer"+str(c)))!="":
                anses.append(str(handle.getAttribute("answer"+str(c)))) 
                c+=1
            self.allAnswers.append(anses)
  
    def prepareSyms(self):
        musicSurface=pygame.image.load(self.musicIconPath)
        naviSurface=pygame.image.load(self.naviIconPath)
        plusSurface=pygame.image.load(self.plusIconPath)

        
        self.musicSurface=pygame.transform.scale(musicSurface,self.boxSize)
        self.naviSurface=pygame.transform.scale(naviSurface,self.boxSize)
        self.plusSurface=pygame.transform.scale(plusSurface,(self.plusIconSize,self.plusIconSize))
    
    def showQuestion(self):
        #question
        self.screen.fill(self.bgColor)
        pygame.display.flip()
        quesSurface = self.quesFont.render(self.question, True, self.quesColor)
        quesCnt=(int(self.scrW/2),self.quesHeight)
        rect=self.screen.get_rect()
        self.screen.blit(quesSurface, quesSurface.get_rect(center=quesCnt))

        #answers
        self.boxSurfaces=[]
        self.boxCenters=[]
        boxW=self.boxSize[0]
        boxH=self.boxHeight
        self.ansSurfaces=[]
        for c in range(self.nClasses):
            self.boxSurfaces.append(pygame.Surface(self.boxSize))
            self.boxSurfaces[c].fill(self.boxColor)
            self.boxCenters.append((int((self.scrW-boxW*self.nClasses)/(self.nClasses+1)*(1+c)+boxW*(c+0.5)),boxH)) #
            self.screen.blit(self.boxSurfaces[c], self.boxSurfaces[c].get_rect(center=self.boxCenters[c]))
            self.ansSurfaces.append(self.ansFont.render(self.answers[c], True, self.ansColor))
            self.screen.blit(self.ansSurfaces[c], self.ansSurfaces[c].get_rect(center=self.boxCenters[c]))

        pygame.display.flip()

    def ask(self):
        def loadQuestion():
            self.question=self.allQuestions[self.quesOrder.pop()]
            self.questionIdx=self.allQuestions.index(self.question)
            self.correct=int(self.corrects[self.questionIdx])
            theAnswers=copy.deepcopy(self.allAnswers[self.questionIdx])
            theCrctAns=theAnswers[self.correct]
            theAnswers.remove(theCrctAns)
            self.answers=[theCrctAns]
            for n in range(self.nClasses-1):
                anotherAns=r.choice(theAnswers)
                self.answers.append(anotherAns)
                theAnswers.remove(anotherAns)    
            r.shuffle(self.answers)
            self.correct=self.answers.index(theCrctAns) 

            #self.allAnswers.remove(self.allAnswers[self.questionIdx])
            #self.corrects.remove(self.corrects[self.questionIdx])
            #self.allQuestions.remove(self.question)   
        
        def wait4Key():
            time.sleep(1)
            pygame.event.clear()
            done = False
            while not done:
                for event in pygame.event.get():
                    if (event.type == KEYUP) or (event.type == KEYDOWN):
                        done = True     
                        self.send_parallel(self.USR_READY)
        loadQuestion()
        self.showQuestion()
        self.send_parallel(self.QUIZ_APPEAR)
        wait4Key()
        
    def process(self):
        def countDown():
            for count in range(self.counterTime):
                self.cl_out_received = 0
                cntrSurface = self.counterFont.render(str(self.counterTime-count), True, self.counterColor)
                cntrCnt=(int(self.scrW/2),self.counterHeight)
                rect=self.screen.get_rect()
                self.screen.blit(cntrSurface, cntrSurface.get_rect(center=cntrCnt))
                pygame.display.update()
                time.sleep(1)
                cntrSurface.fill(self.bgColor)
                self.screen.blit(cntrSurface, cntrSurface.get_rect(center=cntrCnt))
                pygame.display.update()
        
        def showTasks():
            def genMath():
                a=r.randint(5,9)
                b=r.randint(11,19)
                return ''.join([str(a),"+",str(b)])

            self.send_parallel(self.TASK_START)

            for c in range(self.nClasses):
                self.boxSurfaces[c].fill(self.boxColor)
                self.screen.blit(self.boxSurfaces[c], self.boxSurfaces[c].get_rect(center=self.boxCenters[c]))
            
            
            mathSurface = self.iconFont.render(genMath(), True, self.iconColor)
            taskSurfaceDic={'music':self.musicSurface,'navigation':self.naviSurface,'math':mathSurface}
            self.taskSurfaces=[taskSurfaceDic[task] for task in self.tasks]

            theTask=self.orderTasks[self.trlCount-1]
            otherTasks=range(self.nClasses)
            otherTasks.remove(theTask)
            r.shuffle(otherTasks)
            otherAns=range(self.nClasses)
            otherAns.remove(self.correct)
            r.shuffle(otherAns)
            
            self.allTasks=range(self.nClasses)
            self.allTasks[self.correct]=theTask
            self.screen.blit(self.taskSurfaces[theTask], self.taskSurfaces[theTask].get_rect(center=self.boxCenters[self.correct]))
            for c in range(self.nClasses-1):
                self.screen.blit(self.taskSurfaces[otherTasks[c]], self.taskSurfaces[otherTasks[c]].get_rect(center=self.boxCenters[otherAns[c]]))
                self.allTasks[otherAns[c]]=otherTasks[c]
            pygame.display.update()
            time.sleep(self.taskTime)
            #self.showQuestion()
            #pygame.display.update()
            
        def showRest():
            self.screen.fill(self.bgColor)
            #pygame.display.update()
            #restSurface=self.restFont.render('*', True, self.restColor)
            plusCnt=(int(self.scrW/2),self.boxHeight)
            self.screen.blit(self.plusSurface, self.plusSurface.get_rect(center=plusCnt))
            pygame.display.update()
            self.send_parallel(self.REST_START)
            time.sleep(self.restTime)
            self.send_parallel(self.REST_END)
            self.showQuestion()
            
        def showClass():
            clsOnScr=self.allTasks.index(self.cl_out)
            clsCnt=self.boxCenters[clsOnScr]

            classBrdrSurface=pygame.Surface(self.borderSize)
            pygame.draw.rect(self.screen, self.classBorderColor, classBrdrSurface.get_rect(center=clsCnt),5)
            pygame.display.update()
            self.send_parallel(self.OUTPUT_MARKED)
        
        countDown()
        showTasks()
        showRest()
        if self.online:
            while not self.cl_out_received:
                time.sleep(.1)
            showClass()

    def reply(self):
        def getAns():
            pygame.event.clear()
            done = False
            while not done:
                for event in pygame.event.get():
                    if (event.type == KEYUP) or (event.type == KEYDOWN):
                        if event.key in self.keys:
                            done = True
                            self.usrAns=self.keys.index(event.key)
                            self.send_parallel(self.taskTriggers[self.allTasks[self.usrAns]])
                            choiceCnt=self.boxCenters[self.usrAns] 
            
            usrBrdrSurface=pygame.Surface(self.borderSize)
            pygame.draw.rect(self.screen, self.borderColor, usrBrdrSurface.get_rect(center=choiceCnt),5)
            pygame.display.update()

        def markCorrect():
           
            crctSurface=self.boxSurfaces[self.correct]
            crctCnt=self.boxCenters[self.correct]
            crctAns=self.ansSurfaces[self.correct]

            crctSurface.fill(self.correctColor)
            self.screen.blit(crctSurface, crctSurface.get_rect(center=crctCnt))
            self.screen.blit(crctAns, crctAns.get_rect(center=crctCnt))

            pygame.display.update()
            self.send_parallel(self.CRCT_MARKED)
            
   
        getAns()
        time.sleep(0.5)
        markCorrect()

   
    def pre_mainloop(self):
        self.initStuff() 
        self.send_parallel(self.RUN_START)
        time.sleep(1)
        self.trlCount=1  
      
    def tick(self):
        pass            

    def play_tick(self):
        if self.trlCount > self.nTrials or self.allQuestions==[]:
            self.on_stop()
            #self.on_quit()
        else:
            self.restTime= r.uniform(self.restTimeRange[0],self.restTimeRange[1])
            self.ask()
            self.process()
            self.reply()
            time.sleep(3) 
            self.trlCount +=1
        
        
    def post_mainloop(self):
        self.send_parallel(self.RUN_END)
        time.sleep(2)
        pygame.quit()


    def on_control_event(self, data):
        print data
        if data.has_key(u'cl_output'):
            # classification output was sent:
            score_data = data[u'cl_output']
            cl_out = score_data
            self.cl_out=cl_out>0
            self.cl_out_received = 1
        

if __name__ == "__main__":
    feedback = VisualQuiz()
    feedback.on_init()
    feedback.on_play()
 











