'''
Created on 13.01.2011

@author: Rithwik
'''

import numpy
import random

def createLektion(filename='260.txt',fileNo=4,shufTimes=3):
    """ separates a given text file into individual lines"""
    
    #open file and read all lines
    data = open(filename,'r')
    datalist = data.readlines()
    
    # separate distractor list from learning list
    templist = datalist[:240]
    distractList = datalist[240:]
    #numpy.random.shuffle(distractList) #just for fun: if so then append a \n
    # and then do this
    
    # shuffle it shufTimes
    for _ in range(shufTimes):
        numpy.random.shuffle(templist)
    
    for i in range(fileNo):
        # assign the elements to the file
        fileList = []
        fileList = templist[i*60 : (i+1)*60]
        
        # remove the '\n' after the last line
        temp = fileList.pop()
        fileList.append("".join(temp.split("\n")))
        
        # write to a new file
        tempFile = []
        tempFile = open('lektion_simple_'+str(i+1)+'.txt','w')
        tempFile.writelines(fileList)
        tempFile.close()
        
    # write distractor to a new file
    distractorFile = open('lektion_distractor.txt','w')
    distractorFile.writelines(distractList)
    distractorFile.close()
    
    return

if __name__== '__main__':
    filename = '260.txt'
    fileNo = 4
    shufTimes = 3
    createLektion(filename,fileNo,shufTimes)
