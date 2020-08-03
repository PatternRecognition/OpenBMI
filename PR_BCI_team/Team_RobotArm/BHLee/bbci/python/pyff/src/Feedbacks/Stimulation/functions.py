'''
    1. Generally Useful Methods for Python
    2. Parse from setup_relax
'''
import pygame
def misc(text, target, missile):
    if target and missile:
        if len(target) == len(missile):
            for i in range(len(target)):
                text = text.replace(target[i],missile[i])  
    return text

def spaces(text):
    """ replace {  } with /space{x};x is int"""   
    ls = '{'
    rs = '}'
    ms = ' '
    for i in range(1,101):
        temp = ls + ms + rs
        replace = '/space{'+str(i*3)+'}'
        text = text.replace(temp,replace)
        ms = ms + ' '
    
    return text
    
def find(L, func, comp):
    """ find elements in a list or array """
    for index, value in enumerate(L):
        if func(value, comp):
            return index, value
        
def equals(value, comp):
    """ tests for equality """
    if value == comp:
        return True
    else:
        return False
    
def greater(value, comp):
    """ test for greater inequality """
    if value > comp:
        return True
    else:
        return False
    
def smaller(value, comp):
    """ tests for smaller inequality """
    if value < comp:
        return True
    else:
        return False
    
def parse(seq,rek):
    """ Decodes the seq string 
    
    DECODING THE 'seq' STRING SEQUENCE:
         1.P stands for index 3 
             followed by numerical is time in ms
         2.F stands for index 1 (see switch case in stim_artifactMeasurement)
             followed by another index which refers to which sound to play (see cel)
         3.A stands for 4
         4.R is for repeat and is
             followed by number of times of the repeat
             followed by () indicating the sequence to repeat R times
         
     INDICES: 
             1 : play with asyn
             2 : play with syn
             3 : pause
             4 : animation for given time
             5 : play the stimutil_countdown(functionality??) for given time
    """
    
    a=[]
    if not seq:
        if rek > 0:
            raise('parsing error')
        a = []
        rest = ''
        return a, rest
    
    if isinstance(seq[0],int):
        raise('parsing error')
        a = []
        rest = ''
        return a, rest
    
    if seq[0] == 'F':
        [aN, rest] = parse(seq[2:len(seq)], rek)
        a.extend([1,seq[1]])
        #a.append(str([1,seq[1]]))
        if aN:
            a.extend(aN)
        return a, rest
    elif seq[0] == 'f':
        [aN, rest] = parse(seq[2:len(seq)], rek)
        a.extend([2,seq[1]])
        #a.append(str([2,seq[1]]))
        if aN:
            a.extend(aN)
        return a, rest
    elif seq[0] == 'P':
        [aN, rest] = parse(seq[2:len(seq)], rek)
        a.extend([3,seq[1]])
        #a.append(str([3,seq[1]]))
        if aN:
            a.extend(aN)
        return a, rest
    elif seq[0] == 'A':
        [aN, rest] = parse(seq[2:len(seq)], rek)
        a.extend([4,seq[1]])
        #a.append(str([4,seq[1]]))
        if aN:
            a.extend(aN)
        return a, rest
    elif seq[0] == 'C':
        [aN, rest] = parse(seq[2:len(seq)], rek)
        a.extend([5,seq[1]])
        #a.append(str([5,seq[1]]))
        if aN:
            a.extend(aN)
        return a, rest
    elif seq[0] == ')':
        if rek == 0:
            raise('parsing error')
            a = []
            rest = ''
            return a, rest
        a = None
        rest = seq[1:len(seq)]
        return a, rest
    elif seq[0] == 'R':
        if (not seq[1] == '[') or (not isinstance(seq[2],int)) or (not seq[3] == ']') or (not seq[4] == '('):
            raise('parsing error')
            a = []
            rest = ''
            return a, rest
        [aN, rest] = parse(seq[5:len(seq)], rek+1)
        b = []
        #b = [aN for _ in range(seq[2])]
        for _ in range(seq[2]):
            b.extend(aN)
        a.extend(b)
        [aN,rest] = parse(rest, rek)
        a.extend(aN)
        if not rest=='':
            a.extend([rest])
        return a, rest
    else:
        raise('parsing error')
        a = []
        rest = ''
        return a, rest