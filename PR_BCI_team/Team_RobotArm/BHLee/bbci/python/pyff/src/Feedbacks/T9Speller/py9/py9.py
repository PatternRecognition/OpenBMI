# -*- coding: cp1252 -*-
"""
    Python T9 style dictionary by Bitplane feedback@bitplane.net
"""

import struct, os, string, time

# key->letter mapping constants (must be same as dict, [1] is any other char)
allkeys = ["."," ","ABCÄ","DEFÈÉ",
           "GHI","JKL","MNOÖ","PQRSß","TUVÜ","WXYZ"]

DEBUG = 4

class Py9Key:
    """
        for file access to keypress dictionary
    """
    def __init__(self):
        self.refs = [None,None,None,None,None,None,None,None,None]
        self.words = []
        self.fpos = 0L
        
        self.needsave = False
        self.last = -1

    def save(self,f):
        """
            saves node and all child nodes to the file f
            used when creating dictionary file. 
        """
        # recurse save children first so self.ref[x].fpos is always set
        for i in self.refs:
            if i:
                i.save(f)
        # now get position in file
        self.fpos = f.tell()

        # write flags (2 bytes)
        flags = 0
        for i in range(1,10):
            if self.refs[i-1] != None:
                flags = 2 ** i | flags
        f.write(struct.pack("h",flags))
        
        # write positions of children (4 bytes each)
        for i in self.refs:
            if i:
                f.write(struct.pack("i",i.fpos))
        
        # write number of words
        f.write(struct.pack("h",len(self.words)))
        
        # write list of words
        if len(self.words) > 0:
            for l in self.words:
                f.write("%s\n" % l)

    def savenode(self,f):
        """
            just saves this node to the file
            used to add or overwrite a node
        """
        # get position in file
        self.fpos = f.tell()

        # write flags (2 bytes)
        flags = 0
        for i in range(1,10):
            if self.refs[i-1] != None:
                flags = 2 ** i | flags
        if DEBUG > 5: print "writing flags", self.words, flags, self.refs
        f.write(struct.pack("h",flags))
        
        # write positions of children (4 bytes each)
        if DEBUG > 5: print "saving children",
        for i in self.refs:
            if i:
                if DEBUG > 5: print i,
                f.write(struct.pack("i",i))

        if DEBUG > 5: print "..."
        
        # write number of words
        f.write( struct.pack("h",len(self.words)))
        
        # write list of words
        if len(self.words) > 0:
            for l in self.words:
                f.write("%s\n" % l)


    def loadnode(self,f):
        """
            loads a node from an open file object
        """
        self.fpos = f.tell()
        # read flags (2 bytes)
        flags, = struct.unpack("h",f.read(2))
        c = 0
        # loop through flags
        for i in range(1,10):
            if 2 ** i & flags != 0:
                f.seek(self.fpos + 2 + c * 4)
                c += 1
                self.refs[i-1], = struct.unpack("L",f.read(4))

        # read word count
        wc, = struct.unpack("h",f.read(2))
        self.words = []
        for n in range(0,wc):
            self.words.append(f.readline()[:-1])

        if DEBUG > 5: print "***load***",self.refs,self.words
        

class Py9Dict:
    def __init__(self, strDict):
        """
            creates the Py9 dictionary class
                loads file header info:

                word count     L   (==4)
                root node pos  L   (==4)
                language       STR (>=1)
                comment        STR (>=1)
                
        """
        self.file = strDict
        f = open(strDict,"rb")
        f.seek(8)
        self.wordcount, self.rootpos = struct.unpack("LL",f.read(8))
        self.language = f.readline()[:-1]
        self.comment  = f.readline()[:-1]
        f.close()

    def getwords(self,strDigits):
        """
            returns [], or list of possible words
                len(getwords(s)[0]) = len(s):
                    node was found, contained words
                
                len(getwords(s)[0]) > len(s):
                    node found, no words found.
                    look ahead was used to return single word
                    
                len(getwords(s)[0]) < len(s):
                    node not found.
                    look behind was used to return single word
        """
        
        f = open(self.file,"rb")
        k = Py9Key()
        oldlist = []
        p = self.rootpos
        if DEBUG > 5: print "root = ", p
        # process each digit
        
        for c in strDigits:
            
            f.seek(p)
            k.__init__()
            k.loadnode(f)
            
            if k.refs[int(c)-1] != None:
                # the next node is available
                p = k.refs[int(c)-1]
                if len(k.words) > 0:
                    # save the top word
                    oldlist = [k.words[0]]
            else:
                # didn't find the word - return short word
                f.close()
                del k
                return oldlist
            
        # reset node, load node
        k.__init__()
        f.seek(p)
        k.loadnode(f)
        if len(k.words) == 0:
            # couldn't find word
            print strDigits
            if strDigits[-1] == "1":
                f.close()
                del k
                return oldlist
            else:
                while len(k.words) == 0:
                    p = 0
                    for i in k.refs:
                        if i != None:
                            p = i
                            break
                    if p == 0:
                        f.close()
                        return []
                    else:
                        f.seek(p)
                        k.__init__()
                        k.loadnode(f)

            f.close()
            return k.words
        else:
            return k.words

#    def addword(self,word):
#        """
#            adds the given word to the current dictionary
#            must not exist or raises KeyError
#        """
#
#        # messy code, bitch to debug
#        
#        if DEBUG > 5: print "root:", self.rootpos
#        key  = getkey(word)
#
#        f = open(self.file,"rb")
#
#        nodes = []
#        nodes.append(Py9Key())
#        f.seek(self.rootpos)
#        nodes[0].loadnode(f)
#        p = 0
#        
#        # process each digit
#        for c in key:
#            if DEBUG > 5: print "node",nodes[p].last
#            
#            # is it referenced?
#            if nodes[p].refs[int(c)-1] != None:
#                # load it
#                nodes.append(Py9Key())
#                f.seek(nodes[p].refs[int(c)-1])
#                p += 1
#                nodes[p].loadnode(f)
#                
#            else:
#                # create it
#                p += 1
#                nodes.append(Py9Key())
#                # node needs a save
#                nodes[p].needsave = 2
#
#            if DEBUG > 5: print "Last node: ", nodes[p-1].refs[int(key[p-1])-1]
#            if nodes[p-1].refs[int(key[p-1])-1] == None:
#                if DEBUG > 5: print "last node was new"
#                # previous node is also new
#                nodes[p-1].needsave = 2
#                if p-2 >= 0:
#                    if nodes[p-2].refs[int(key[p-2])-1] == None:
#                        if DEBUG > 5: print "ding ding"
#                        nodes[p-2].needsave = 2
#                    else:
#                        nodes[p-2].needsave = 1
#            else:
#                # previous node is old - update
#                if nodes[p-1].needsave != 2:
#                    nodes[p-1].needsave = 1
#
#
#        f.close()
#
#        # let's not add dupes
#        for q in nodes[p].words:
#            if q.lower() == word.lower():
#                del nodes
#                raise KeyError("Word '" + word + "' is already in dictionary '" + self.file +"' at position " + key)
#
#        # sort out the last ones
#        for n in range(1,len(nodes)):
#            nodes[n].last = int(key[n-1])-1
#
#        # add the word to the list
#        nodes[p].words.append(word)
#        nodes[p]
#        nodes[p].needsave = 2
#        if nodes[p-1].fpos != 0:
#            nodes[p].last = int(c)-1
#            if nodes[p-1].needsave == -1: nodes[p-1].needsave = 1
#        
#        # now work from the last digit back saving each one
#        for n in range(len(nodes)-1,-1,-1):
#            if nodes[n].needsave == 2:
#                if DEBUG > 5: print "needs save:",n
#
#                # are we moving the root node?
#                movert = self.rootpos == nodes[n].fpos                
#            
#                f = open(self.file,"r+b")
#                
#                f.seek(os.stat(self.file)[6])
#                if DEBUG > 5: print n, len(nodes)
#                if n < len(nodes)-1:
#                    if DEBUG > 5: print "ok... next is a ", nodes[n+1].last, "at", nodes[n+1].fpos
#                    if nodes[n+1].last != -1:
#                        if DEBUG > 5: print "...w: new fpos for", n , "is" , nodes[n+1].fpos
#                        nodes[n].refs[nodes[n+1].last] = nodes[n+1].fpos    
#                nodes[n].savenode(f)
#                if DEBUG > 5: print "...append gave me a fpos of ",nodes[n].fpos
#                f.close()
#                if movert:
#                    self.rootpos = nodes[n].fpos
#                    
#            elif nodes[n].needsave == 1:
#                if DEBUG > 5: print "needs update:",n, "pos=",nodes[n].fpos
#                f = open(self.file,"r+b")
#
#                if DEBUG > 5: print "...u: new fpos for", n , "is" , nodes[n+1].fpos
#                nodes[n].refs[nodes[n+1].last] = nodes[n+1].fpos
#                
#                f.seek(nodes[n].fpos)
#                nodes[n].savenode(f)
#                f.close()
#            else:
#                if DEBUG > 5: print "no edit:",n,"pos=",nodes[n].fpos
#
#        self.wordcount += 1
#        f = open(self.file,"r+b")
#        f.seek(8)
#        f.write(struct.pack("LL",self.wordcount,self.rootpos))
#        f.close()
#        if DEBUG > 5: print "root:", self.rootpos
#        del nodes
#        
    def test(self,word):
        print self.getwords(getkey(word))
                

#    def delword(self,word):
#        """
#            deletes the given word from the dictionary
#            must exist or raises exception
#        """
#        print "Py9Dict.delword() NOT IMPLEMENTED"


class Py9Input:
    """
        The input parser.
        Handles keypresses from the user and manipulates the text.
        
        Send keypresses to this control with sendkeys(), retrieve
        it's input for display with gettext() (includes cursor,
        to get the raw text use text)
        
    """
    def __init__(self,dict,defaulttxt="",defaultmode=0,
                 keydelay=0.0,numeric=False):
        """
            create a new input parser
            
            dict        = dictionary file name
            defaulttxt  = text to start with
            defaultmode = mode to start in
                          0=Predictive, 3=TXT lower, 4=TXT upper, 5=Numeric)
            keydelay    = key timeout in TXT mode
            numeric     = NOT IMPLEMENTED YET
        """
        self.dict         = Py9Dict(dict)  # dict for lookups
        self.modes        = ["Abc...","[Abc]","[a..]","abc","ABC","123"]
        self.modekeys     = [
            "0=Space, 1-9=Abc..., D=DEL, ULR=Navigate, S:abc",
            "0=Save/Space, 1-9=[Abc], D=DEL, U=Change, LR=Navigate, S:123",
            "0=Save/Reset, 1-9=[a..], D=DEL, U=Change, LR=Navigate/Save, S:[A..]",
            "0=Space, 1-9=abc, D=DEL, ULR=Navigate, S:Abc...",
            "0-Space, 1-9=ABC, D=DEL, ULR=Navigate, S:123",
            "0-9=123, D=DEL, ULR=Navigate, S:Abc..."
            ]
        self.mode         = defaultmode    # 0=navigate,1=edit word
                                           # 2=edit chars,3=lcase txt
                                           # 4=ucase txt,5=numbers
        self.pos          = 0              # curs pos       (edit chars)
        self.keys         = ""             # keys typed     (edit word)
        self.word         = ""             # word displayed (edit word/chars)
        self.words        = []             # possible words (edit word)
        self.textbefore   = "   "          # before the cursor
        self.textafter    = ""             # after the cursor
        self.lastkeypress = ""             # last key pressed   (txt input)
        self.lastkeytime  = time.clock()   # time from last key (txt input)
        self.keydelay     = keydelay       # time to chang char (txt input)
        self.numeric      = numeric        # True if this is numbers only 

    def getkey(self,strWord):
        """
            returns a string of the keystrokes required to type strWord
            (py9.getkey alias for calling from outside)
        """
        return getkey(strWord)

    def showmode(self):
        """
            returns a text logo for current input mode
        """
        return self.modes[self.mode]
    
    def showkeys(self):
        """
            returns a string telling the user what keys do what
        """
        return self.modekeys[self.mode]

    def gettext(self):
        """
            returns the current text in the control including the cursor
            for raw text use x.text()
        """
        if self.mode == 0:
            return self.textbefore + "|" + self.textafter
        elif self.mode == 1:
            return self.textbefore + "[" + self.word + "]" + self.textafter
        elif self.mode == 2:
            return self.textbefore + "\"" + self.posword() + "\"?" + self.textafter
        elif self.mode == 3:
            return self.textbefore + "()" + self.textafter
        elif self.mode == 4:
            return self.textbefore + "[]" + self.textafter
        elif self.mode == 5:
            return self.textbefore + "#" + self.textafter
    def text(self):
        """
            returns the control's text without cursor 
        """
        if self.mode == 2 or self.mode == 1:
            return self.textbefore + self.word + self.textafter
        else:
            return self.textbefore + self.textafter

    def posword(self):
        """
            returns the word with position marker set
        """
        return "%s|%c|%s" % (self.word[0:self.pos],
                             self.word[self.pos],
                             self.word[self.pos+1:len(self.word)])

    def setword(self):
        """
            changes the current word to the first valid one
        """

        if len(self.keys) == 0:
            self.mode = 0
            return

        self.pos  = 0
        
        self.words = self.dict.getwords(self.keys)
        if len(self.words) == 0:
            self.word = "." * len(self.keys)
        else:
            wl = len(self.words[0])
            kl = len(self.keys)
            if wl == kl:
                # same length
                self.word = self.words[0]
                self.mode = 1
                
            elif wl > kl:
                # long
                self.word = self.words[0][0:kl]
                self.mode = 2
            else:
                # short
                self.word = self.words[0] + "."*(kl-wl)
                self.mode = 2

    def nextword(self):
        """
            in edit mode 1, moves to the next word if possible
        """

        if self.word in self.words:

        # enter manual edit if we run out of words
            i = self.words.index(self.word)
            print i
            if i == len(self.words)-1:
                #self.mode = 2
                #self.pos  = 0
                #print "tata"
                self.mode = 1
                self.word = self.words[0] #choose first word again
            else:
                self.mode = 1
                self.word = self.words[i+1]
        else:
            # the word was not found
            self.setword()
        
#    def nextchar(self):
#        """
#            in edit mode 2, selects the next letter in the group
#        """
#        c   = self.word[self.pos]
#        lc  = c == c.lower()
#        c   = c.upper()
#        key = int(self.keys[self.pos])
#        if not c in allkeys[key]:
#            c = allkeys[key][0]
#        else:
#            i = allkeys[key].find(c) + 1
#            if i < len(allkeys[key]):
#                c = allkeys[key][i]
#            else:
#                c = allkeys[key][0]
#        if lc:
#            c = c.lower()
#        self.word = "%s%c%s" % (self.word[0:self.pos], c,
#                                  self.word[self.pos+1:len(self.word)])


    def addkeypress(self,key):
        if key == "1" and self.keys[0] != "1":
            if self.keys[-1] == "0":
                # this is punctuation only - skip the word (no save)
                self.textbefore += self.word[0:-1]
                self.keys        = self.keys[-1]
                self.keys       += key
                self.setword()
            else:
                self.keys   += key
                self.words   = self.dict.getwords(self.keys)
                self.word   += "'"
        else:
            self.keys += key
            # reset text
            self.setword()

    def sendkeys(self,keys):
        """
            sends action keys, (key = "0123456789UDLRS")
            
            UDLRS = UP, DOWN, LEFT, RIGHT, SELECT

            naviagate mode:
                0123456789 = start new word
                ULR        = Navigate
                D          = backspace
                S          = LAMO TXT MODE (3)
            
            edit mode (1+2):
                0123456789 = append keystroke


                complete word (mode=1):
                    LR         = accept
                    U          = change word
                    D          = backspace
                    S          = char edit mode (2)

                incomplete word (mode=2):
                    L          = back letter
                    R          = confirm letter
                    U          = change letter
                    D          = delete char
                    S          = change case

            letter mode (3+4): (upper+lower)
                0123456789 = numbers
                ULR        = Navigate
                D          = backspace
                S          = mode++ (4,5)
                
            number mode (4):
                0123456789 = numbers
                ULR        = Navigate
                D          = backspace
                S          = navigate mode (1)

        """
        for key in keys:
            if self.mode == 0:
                # navigate mode
                if key in "23456789":
                    # starting a new word - edit mode
                    self.mode = 1
                    self.keys = key
                    self.words = self.dict.getwords(key)
                    self.setword()
                elif key == "0":
                    self.textbefore = self.textbefore + "."
                    #print "TATA", self.keys, self.word, self.words, self.keys
                    
                elif key == "1":
                    # insert a space
                    self.textbefore = self.textbefore + " "
                    #self.nextword()
                    self.mode = 0 # starting a new word - edit mode
                    #print "Tata2", self.keys, self.word, self.words, self.keys
                elif key == "D":
                    # delete a char
                    if self.textbefore == "":
                        return
                    if int(getkey(self.textbefore[-1])) <2:
                        self.textbefore = self.textbefore[:-1]
                    else:
                        # edit word
                        self.mode       = 1
                        # move in to edit buffer
                        self.word       = self.textbefore.split(" ")[-1]
                        self.textbefore = self.textbefore[0:self.textbefore.rfind(" ")+1]
                        if self.word    == "":
                            self.mode   = 0
                        else:
                            self.keys       = getkey(self.word)
                            self.words      = self.dict.getwords(self.keys)
                        

                elif key == "S":
                    self.mode = 3
                    
                elif key in "UL":
                    if self.textbefore == "":
                        return
                    # left a char
                    if int(getkey(self.textbefore[-1])) <2:
                        # move one char
                        self.textafter = self.textbefore[-1] + self.textafter
                        self.textbefore = self.textbefore[:-1]
                    else:
                        # edit word
                        self.mode       = 1
                        # move to edit buffer
                        t               = self.textbefore.split(" ")
                        self.word       = t[-1]
                        self.textbefore = self.textbefore[0:self.textbefore.rfind(" ")+1]
                        self.keys       = getkey(self.word)
                        self.words      = self.dict.getwords(self.keys)
                
                elif key == "R":
                    # right a char
                    if self.textafter == "":
                        return
                    if int(getkey(self.textafter[0])) <2:
                        # move one char
                        self.textbefore = self.textbefore + self.textafter[0]
                        self.textafter  = self.textafter[1:]
                    else:
                        # edit word
                        self.mode       = 1
                        # move to edit buffer
                        t               = self.textafter.split(" ")
                        self.word       = t[0]
                        self.textafter  = self.textafter[len(t[0]):len(self.textafter)]
                        self.keys       = getkey(self.word)
                        self.words      = self.dict.getwords(self.keys)
            elif self.mode < 3:
                # mode 1,2
                if key == "D":
                    # down key = delete 1 char
                    self.keys = self.keys[:-1]
                    # reset text
                    self.setword()

                if self.mode == 1:
                    # edit mode 1 - edit word
                    # space/right = accept.
                    if key in "23456789":
                        # add keypress
                        self.addkeypress(key)
                        
                    elif key in "1R":
                        if not self.textbefore[-1] in " .": 
                            self.textbefore = self.textbefore + " "
                            self.mode = 1
                            
                        # save this word?
                        elif not self.word in self.words:
                            # save it
                            if DEBUG > 3: print "saving word: ", self.word
                            self.dict.addword(self.word)
                        
                        # return to navigate mode.
                        self.mode = 0
                        self.textbefore += self.word
                        if key == "1":
                            self.textbefore += " "
                            #print "Tata3", self.keys, self.word, self.words, self.keys
                    elif key == "L":
                        # save this word?
                        if not self.word in self.words:
                            # save it
                            if DEBUG > 3: print "saving word: ", self.word
                            self.dict.addword(self.word)
                            
                        # return to navigate mode.
                        self.mode = 0
                        self.textafter = self.word + self.textafter
                    elif key == "U":
                        # up is navigate to next word
                        self.nextword()

                else:
                    # edit mode 2: edit chars
                    if key in "23456789":
                        # add keypress
                        self.addkeypress(key)

                    if key == "1":
                        # whenever "1" is pressed, we leave the word the way it is
                        self.mode = 0
                        self.textbefore += self.word
                        self.textbefore += " "
                        # only move on if we are at the end
                        
#                        if self.pos == len(self.word) -1:
#                            
#                            # save this word?
#                            if not self.word in self.words:
#                                if DEBUG > 3: print "saving word: ", self.word
#                                self.dict.addword(self.word)
#                            
#                            # return to navigate mode.
#                            self.mode = 0
#                            self.textbefore += self.word + " "
#                        else:
#                            # reset the text.
#                            self.setword()
                    elif key == "S":
                        # change the case of current letter
                        c = self.word[self.pos]
                        if c.upper() == c :
                            self.word = self.word[0:self.pos] + c.lower() + self.word[self.pos+1:len(self.word)]
                        else:
                            self.word = self.word[0:self.pos] + c.upper() + self.word[self.pos+1:len(self.word)]
                    elif key == "L":
                        # move back one char
                        if self.pos > 0:
                            self.pos -= 1
                    elif key == "R":
                        # move forward one char
                        k = allkeys[int(self.keys[self.pos])]
                        if self.word[self.pos].upper() in k:
                            if self.pos == len(self.word)-1:
                                # save this word?
                                if not self.word in self.words:
                                    if DEBUG > 3: print "saving word: ", self.word
                                    self.dict.addword(self.word)
                                
                                # return to navigate mode. 
                                self.mode = 0
                                self.textbefore += self.word
                            else:
                                self.pos += 1
            

def getkey(strWord):
    """
        getkey(string) -> string of digits
        Converts a word to keypresses
    """
    r = ""
    for c in strWord:
        d = c.upper()
        dig = "1"
        for n in range(0,len(allkeys)):
            if allkeys[n].find(d) != -1:
                dig = str(n)
        r = r + dig
    return r
