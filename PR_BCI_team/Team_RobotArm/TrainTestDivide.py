# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 16:12:31 2018

@author: wam
"""
import random
import loadEEG
from numpy import newaxis
import numpy as np
def convertDivide(path, testSet,className):
    
    ## variable
    testIndex = []
    testDataX = []
    testDataY = []
    trainDataX = []
    trainDataY = []
    
    data = loadEEG.loadEEG(path)
    # data has clab, x, y, t, className
    clab = data[0]
    x = np.array(data[1])
    y = np.array(data[2])
    t = data[3]
    ## clab and the className are not supported (matlab)
    trials = x.shape[2] / len(className)
    
    ## divide into train and test set
    for i in range(len(className)):
        r = range(i * trials, (i + 1) * trials - 1)
        tempR = random.sample(r, testSet)
        if i == 0:
            testIndex = tempR
        else:
            testIndex += tempR
    teindex=0
    trindex=0
    for i in range(x.shape[2]):
        if i in testIndex:
            if teindex==0:
                teindex=+1
                testDataX=x[:,:,i]
                testDataY=y[:,i]
            elif teindex==1:
                teindex+=1
                testDataX=testDataX[:,:,newaxis]
                testDataY=testDataY[:,newaxis]
                tempX=np.array(x[:,:,i])
                tempX=tempX[:,:,newaxis]
                tempY=np.array(y[:,i])
                tempY=tempY[:,newaxis]
                testDataX=np.concatenate((testDataX,tempX),axis=2)
                testDataY=np.concatenate((testDataY,tempY),axis=1)
                
            else:
                tempX=np.array(x[:,:,i])
                tempX=tempX[:,:,newaxis]
                tempY=np.array(y[:,i])
                tempY=tempY[:,newaxis]
                testDataX=np.concatenate((testDataX,tempX),axis=2)
                testDataY=np.concatenate((testDataY,tempY),axis=1)
        else:
            if trindex==0:
                trindex+=1
                trainDataX=x[:,:,i]
                trainDataY=y[:,i]
            elif trindex==1:
                trindex+=1
                trainDataX=trainDataX[:,:,newaxis]
                trainDataY=trainDataY[:,newaxis]
                tempX=np.array(x[:,:,i])
                tempY=np.array(y[:,i])
                tempX=tempX[:,:,newaxis]
                tempY=tempY[:,newaxis]
                trainDataX=np.concatenate((trainDataX,tempX),axis=2)
                trainDataY=np.concatenate((trainDataY,tempY),axis=1)
            else:
                tempX=np.array(x[:,:,i])
                tempX=tempX[:,:,newaxis]
                tempY=np.array(y[:,i])
                tempY=tempY[:,newaxis]
                trainDataX=np.concatenate((trainDataX,tempX),axis=2)
                trainDataY=np.concatenate((trainDataY,tempY),axis=1)
    return clab, trainDataX, trainDataY, testDataX, testDataY, t