# -*- coding: cp1252 -*-

import py9, msvcrt, time, os.path

class BCIpy9interface:
    """
        T9 spelling for a BCI system.
    """
    
    def __init__(self, dict):
        #dict = "EN-BR.dict"
        #dict = "DE-DE-SMALL.dict"
        #dict = "C:/svn/bbci/python/pyff/src/Feedbacks/T9Speller/DE-DE-SMALL.dict"
        #dict = "D:/svn/bbci/python/pyff/src/Feedbacks/T9Speller/py9/DE-DE-SMALL.dict"
        
        if not os.path.isfile(dict):
            print "creating dictionary (1 time only)"
            print "loads of ram required"
            import makepy9

        self.x = py9.Py9Input(dict,"")
        print  "BCIpy9interface was successfully loaded"
        
    def handleInput(self,i):
        """
            handles the Input which comes from the Speller. 
            Returns a matching word!
            i is a char and can be "0123456789D" !
        """
        self.x.sendkeys(i)
        return(self.x.gettext())
    
    def gettext(self):
        return(self.x.gettext())
    
    def getRawText(self):
        return(self.x.text())

    def giveSuggestions(self):
        return(self.x.words)

    def giveKeys(self):
        return(self.x.keys)
    
    def foundValidWord(self):
        if self.x.mode == 1: return True
        else: return False    
    
if __name__ == "__main__": #test input "hello world" which is 43556196753
 #test input in german is "mein Haus ist schön" which is 6346142371478172466
    import py9, msvcrt, time, os.path
    p9 = BCIpy9interface()
#    print(p9.handleInput(str(4)))
#    print(p9.handleInput(str(3)))
#    print(p9.handleInput(str(5)))
#    print(p9.handleInput(str(5)))
#    print(p9.handleInput(str(6)))
#    print(p9.handleInput(str(1)))
#    print(p9.handleInput(str(9)))
#    print(p9.handleInput(str(6)))
#    print(p9.handleInput(str(7)))
#    print(p9.handleInput(str(5)))
#    print(p9.handleInput(str(3)))
#    print(p9.handleInput(str(1)))
    print
    print 6346142871478172466
    print(p9.handleInput(str(6)))
    print(p9.handleInput(str(3)))
    print(p9.handleInput(str(4)))
    print(p9.handleInput(str(6)))
    print(p9.handleInput(str(1)))
    print(p9.handleInput(str(4)))
    print(p9.handleInput(str(2)))
    print(p9.handleInput(str(8)))    
    print(p9.handleInput(str(7)))
    print(p9.handleInput(str(1)))
    print(p9.handleInput(str(4)))
    print(p9.handleInput(str(7)))
    print(p9.handleInput(str(8)))
    print(p9.handleInput(str(1)))
    print(p9.handleInput(str(7)))
    print(p9.handleInput(str(2)))
    print(p9.handleInput(str(4)))
    print(p9.handleInput(str(6)))
    print(p9.handleInput(str(6)))     