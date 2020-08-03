# -*- coding: cp1252 -*-
import os

dictFile = os.path.abspath(os.path.dirname( os.path.abspath(__file__)) + "\\DE-DE-SMALL.WORDS.TXT")
dictFileout = os.path.abspath(os.path.dirname( os.path.abspath(__file__)) + "\\DE-DE-SMALLextracted.WORDS.TXT")
f = open(dictFile,"rt")
wrds = []
ChuckedWdrs = []
count=0
for line in f:
    count += 1
    print line
    if set([line.upper()]).issubset(map(str.upper, wrds)):
        ChuckedWdrs.append(line)
    else:
        wrds.append(line)

f2 = open(dictFileout, 'w')
for l in wrds:
    f2.write(l)
 
f2.close()
    
f.close()

print ChuckedWdrs
print 'len chucked ', len(ChuckedWdrs)
print 'len remaining ', len(wrds)


#
#

