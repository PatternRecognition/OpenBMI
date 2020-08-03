#!/usr/bin/python
# -*- coding: latin-1 -*-
import jpype, pygame, os, time, wave

class MaryClient:
    specificationVersion = "0.1"
    
    """Python implementation of a MARY TTS client"""
    def __init__( self, maryDir, host='localhost', port=59125):
        self.host = host
        self.port = port
        if not jpype.isJVMStarted():
            try:
                JVMpath = jpype.getDefaultJVMPath()
                classpath = maryDir+"java\\maryclient.jar;"+maryDir+ "java\\java-diff.jar"
                #start Java virtual machine with the Mary-libs in the classpath
                jpype.startJVM(JVMpath, "-ea", "-Djava.class.path=%s" % classpath)
                jpype.java.lang.System.out.println("Java Virtual machine started")
            except:
                print "ERROR: JVM couldn't be started!" 
        else:
            java.lang.System.out.println("Java Virtual machine is already running")
        try:
            maryClientClass = jpype.JClass('de.dfki.lt.mary.client.MaryClient') #only valid for MaryTTS 3.6 version !!!!!
        except:
            print "Mary TTS could not be loaded. Check the path and version (should be 3.6.0)"
            
        
        self.tmpAudioDir = maryDir+'tmp\\'
        try:
            self.maryClient = maryClientClass(self.host, self.port)
        except:
            os.startfile(os.path.abspath(maryDir+'bin/maryserver.bat'))
            print "Wait for the Mary TTS server to start"
            time.sleep(10)
            
            self.maryClient = maryClientClass(self.host, self.port)
          
        text = "Willkommen in der Welt der Sprachsynthese!";
        self.inputType = "TEXT_DE";
        self.outputType = "AUDIO";
        self.audioType = "WAVE";
        self.defaultVoiceName = 'hmm-bits2'
        #pygame.mixer.quit()
        #pygame.mixer.init(15000, -16, 2, 512) #1024 bits is very important since we introduce a huge gitter (inconsistent delay) otherwise... 
        # sample size 15000 is important!!! In the standeard settings the speech-files are played in doublespeed or so!!  
        
        self.audioChannel = pygame.mixer.find_channel() 
        self.counter = 0
        self.FilesToRemove = []
        
    def process(self, text, outputFile = 'C:\\temp\\testJAVA.wav'):# 'C:\\temp\\testJAVA.wav'):
        out = jpype.java.io.ByteArrayOutputStream()
        self.maryClient.process(text, unicode(self.inputType, 'utf-8'), unicode(self.outputType, 'utf-8'), unicode(self.audioType, 'utf-8'),
            unicode(self.defaultVoiceName, 'utf-8'), out);
        fileStream = jpype.java.io.FileOutputStream (outputFile);
        out.writeTo(fileStream)
        out.close()
        fileStream.close()
        return os.path.abspath(outputFile)
        
        
        
    def processAndPlay(self, text):
        self.counter += 1
        out = jpype.java.io.ByteArrayOutputStream()
        try:            
            self.maryClient.process(text, unicode(self.inputType, 'utf-8'), unicode(self.outputType, 'utf-8'), unicode(self.audioType, 'utf-8'), unicode(self.defaultVoiceName, 'utf-8'), out);
        except:
            print "first trial of speech-proccessing failed, another try"
            try:
                 pygame.time.delay(10)
                 self.maryClient.process(text, unicode(self.inputType, 'utf-8'), unicode(self.outputType, 'utf-8'), unicode(self.audioType, 'utf-8'), unicode(self.defaultVoiceName, 'utf-8'), out);
            except:
                print "second trial of speech-proccessing failed, third and LAST try:"
                try:
                    pygame.time.delay(20)
                    self.maryClient.process(text, unicode(self.inputType, 'utf-8'), unicode(self.outputType, 'utf-8'), unicode(self.audioType, 'utf-8'), unicode(self.defaultVoiceName, 'utf-8'), out);
                except:
                    print "Speechprocessing failed!!"
                    return

        outputFile = self.tmpAudioDir+'bufferfile'+str(self.counter)+'.wav'
        outputFile_conv = self.tmpAudioDir+'bufferfile'+str(self.counter)+ "_conv"+'.wav'
        fileStream = jpype.java.io.FileOutputStream (outputFile);
        out.writeTo(fileStream)
        out.close()
        fileStream.close()
        
        
        input = wave.open(outputFile, 'r')
        output = wave.open(outputFile_conv, 'w')
        output.setparams(input.getparams())
        output.setframerate(44100)
        buffer = ''
        for i in range(input.getnframes()):
            buffer += input.readframes(1) * 3
        
        output.writeframes(buffer) 
        output.close()
        input.close()
        pygame.time.delay(10)
        
        snd = pygame.mixer.Sound(os.path.abspath(outputFile_conv)) 
        print outputFile_conv      
        print outputFile 
        self.audioChannel.play(snd)
        #while playing the current sound file, we can remove previously played files
        for ff in self.FilesToRemove:
                os.remove(self.tmpAudioDir + self.FilesToRemove.pop())
                pass
        return False
        
        
        
    def isBusy(self):
        if self.audioChannel.get_busy():
            #channel is busy --> sth is read
            return True
        else:
            self.FilesToRemove = os.listdir(self.tmpAudioDir)
            return False

            