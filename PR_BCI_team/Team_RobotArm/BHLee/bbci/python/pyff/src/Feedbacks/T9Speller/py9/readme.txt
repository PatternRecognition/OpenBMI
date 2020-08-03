python "T9" style dictionary for python, v0.1
for use with xbox media center and chat programs - chat via the remote!

gaz@bitplane.net


------------------------------------------------------------------------

            DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE 
                    Version 2, December 2004 

 Copyright (C) 2004 Sam Hocevar 
  14 rue de Plaisance, 75014 Paris, France 
 Everyone is permitted to copy and distribute verbatim or modified 
 copies of this license document, and changing it is allowed as long 
 as the name is changed. 

            DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE 
   TERMS AND CONDITIONS FOR COPYING, DISTRIBUTION AND MODIFICATION 

  0. You just DO WHAT THE FUCK YOU WANT TO. 

------------------------------------------------------------------------

The language files are huge, but it's organised so the number of reads depends 
on the number of buttons pressed. It's pretty fast, and uses virtually no ram or cpu. 

There are 3 classes, a database node (Py9Key), a database client (Py9Dict), and an inputparser (Py9Input). You'll only need to bother with the latter.

See demo.py for a good example (works best in a dos prompt), keys are 0-9, UDLR (navigation) and S (select mode). You'll need loads of ram the first time you run it, after it's made the DB it'll need hardly any.


  language files wanted!

  EN-GB: Downloaded from the web, derived from gnu aspell (iirc)
  NL-DU: Thanks to Breght Boschker for submitting these :)


To make your own dictionary, have a read of makePy9.py. 
Keep your wordlists though - the file format might change in future.



help(py9) follows....

------------------------

Help on module py9:

NAME
    py9 - Python T9 style dictionary by Bitplane feedback@bitplane.net

FILE
    py9.py

CLASSES
    Py9Dict
    Py9Input
    Py9Key
    
    class Py9Dict
     |  Methods defined here:
     |  
     |  __init__(self, strDict)
     |      creates the Py9 dictionary class
     |          loads file header info:
     |      
     |          word count     L   (==4)
     |          root node pos  L   (==4)
     |          language       STR (>=1)
     |          comment        STR (>=1)
     |  
     |  addword(self, word)
     |      adds the given word to the current dictionary
     |      must not exist or raises KeyError
     |  
     |  delword(self, word)
     |      deletes the given word from the dictionary
     |      must exist or raises exception
     |  
     |  getwords(self, strDigits)
     |      returns [], or list of possible words
     |          len(getwords(s)[0]) = len(s):
     |              node was found, contained words
     |          
     |          len(getwords(s)[0]) > len(s):
     |              node found, no words found.
     |              look ahead was used to return single word
     |              
     |          len(getwords(s)[0]) < len(s):
     |              node not found.
     |              look behind was used to return single word
     |  
     |  test(self, word)
    
    class Py9Input
     |  The input parser.
     |  Handles keypresses from the user and manipulates the text.
     |  
     |  Send keypresses to this control with sendkeys(), retrieve
     |  it's input for display with gettext() (includes cursor,
     |  to get the raw text use text)
     |  
     |  Methods defined here:
     |  
     |  __init__(self, dict, defaulttxt='', defaultmode=0, keydelay=0.5, numeric=False)
     |      create a new input parser
     |      
     |      dict        = dictionary file name
     |      defaulttxt  = text to start with
     |      defaultmode = mode to start in
     |                    0=Predictive, 3=TXT lower, 4=TXT upper, 5=Numeric)
     |      keydelay    = key timeout in TXT mode
     |      numeric     = NOT IMPLEMENTED YET
     |  
     |  addkeypress(self, key)
     |  
     |  getkey(self, strWord)
     |      returns a string of the keystrokes required to type strWord
     |      (py9.getkey alias for calling from outside)
     |  
     |  gettext(self)
     |      returns the current text in the control including the cursor
     |      for raw text use x.text()
     |  
     |  nextchar(self)
     |      in edit mode 2, selects the next letter in the group
     |  
     |  nextword(self)
     |      in edit mode 1, moves to the next word if possible
     |  
     |  posword(self)
     |      returns the word with position marker set
     |  
     |  sendkeys(self, keys)
     |      sends action keys, (key = "0123456789UDLRS")
     |      
     |      UDLRS = UP, DOWN, LEFT, RIGHT, SELECT
     |      
     |      naviagate mode:
     |          0123456789 = start new word
     |          ULR        = Navigate
     |          D          = backspace
     |          S          = LAMO TXT MODE (3)
     |      
     |      edit mode (1+2):
     |          0123456789 = append keystroke
     |      
     |      
     |          complete word (mode=1):
     |              LR         = accept
     |              U          = change word
     |              D          = backspace
     |              S          = char edit mode (2)
     |      
     |          incomplete word (mode=2):
     |              L          = back letter
     |              R          = confirm letter
     |              U          = change letter
     |              D          = delete char
     |              S          = change case
     |      
     |      letter mode (3+4): (upper+lower)
     |          0123456789 = numbers
     |          ULR        = Navigate
     |          D          = backspace
     |          S          = mode++ (4,5)
     |          
     |      number mode (4):
     |          0123456789 = numbers
     |          ULR        = Navigate
     |          D          = backspace
     |          S          = navigate mode (1)
     |  
     |  setword(self)
     |      changes the current word to the first valid one
     |  
     |  showkeys(self)
     |      returns a string telling the user what keys do what
     |  
     |  showmode(self)
     |      returns a text logo for current input mode
     |  
     |  text(self)
     |      returns the control's text without cursor
    
    class Py9Key
     |  for file access to keypress dictionary
     |  
     |  Methods defined here:
     |  
     |  __init__(self)
     |  
     |  loadnode(self, f)
     |      loads a node from an open file object
     |  
     |  save(self, f)
     |      saves node and all child nodes to the file f
     |      used when creating dictionary file.
     |  
     |  savenode(self, f)
     |      just saves this node to the file
     |      used to add or overwrite a node

FUNCTIONS
    getkey(strWord)
        getkey(string) -> string of digits
        Converts a word to keypresses

DATA
    DEBUG = 4
    allkeys = [' ', '.,!?"\'():;=+-/@|\xa3$%*<>[]\\^_{}~#', 'ABC\xc0\xc2\x...




