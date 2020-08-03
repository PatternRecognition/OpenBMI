# -*- coding: cp1252 -*-

import py9, msvcrt, time, os.path

if not os.path.isfile("DE-DE-SMALL.dict"): #DE-DE or EN-BR for german or english
    print "creating dictionary (1 time only)"
    print "loads of ram required (130mb+)"
    import makepy9

x = py9.Py9Input("DE-DE-SMALL.dict","any old chunk of text that's worth editing I spose")

i = ""
print x.showmode(), "---", x.showkeys()
print x.gettext()
print "? [0-9/UDLR/S/Q] >",
while i != "Q":
    time.sleep(0.05)
    i = msvcrt.getche()
    if i == "ÿ":
        i = raw_input(">")
    
    if i < "~" :
        i = i.upper()
        print "enteredKey=",i
        x.sendkeys(i)
        print
        print
        print
        print
        print
        print
        print
        print
        print
        print
        print x.showmode(), "---", x.showkeys()
        print x.keys
        print x.gettext()
        print
        print "mode"
        print x.mode
        print
        print "words", x.words
        print "? [0-9/UDLR/S/Q]",

# print the text
print x.text()

