'''
    To Translate minimalistic latex script into Glyph readable
'''

class Translate():
    def __init__(self, text=' ', target=None, missile=None):
        text = self.bold(text)
        text = self.spaces(text)
        text = self.rm(text)
        self.text = self.misc(text, target, missile)
        #self.final(text)

    def bold(self,text):
        """ remove \b and introduce b; in {} after \b"""    
        return text.replace('\\bf{','{b;')
        
    def spaces(self,text):
        """ replace {  } with /space{x};x is int"""   
        ls = '{'
        rs = '}'
        ms = ' '
        for i in range(1,101):
            temp = ls + ms + rs
            replace = '/space{'+str(i)+'}'
            text = text.replace(temp,replace)
            ms = ms + ' '
        
        return text
        
    def rm(self, text):
        """ removes renderings like \\rm"""
        return text.replace('\\rm','')
        
    def misc(self, text, target, missile):
        if target and missile:
            if len(target) == len(missile):
                for i in range(len(target)):
                    text = text.replace(target[i],missile[i])        
        
        return text

    def final(self):
        return self.text