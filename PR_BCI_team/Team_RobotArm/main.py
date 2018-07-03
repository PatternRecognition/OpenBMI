import TrainTestDivide

## load data
filelist = ['20180323_jgyoon2_twist_MI', '20180220_bwyu_twist_MI']
path = ('/home/wam/Desktop/KHShim/matlab/DL/Converted/')
# variable should be changed by the patients
path = path + filelist[1]

#clab, trainDataX, trainDataY, testDataX, testDataY, t, className
# input value: path & test data for each class
className = ['Left', 'Right', 'Rest']
data=TrainTestDivide.convertDivide(path, 5, className)
clab=data[0]
trainDataX=data[1]
trainDataY=data[2]
testDataX=data[3]
testDataY=data[4]
t=data[5]


import tensorflow as tf

# hyperparameters
learningRate=0.001
numSteps=500
batchSize=128
displayStep=100

#Network Parameters
