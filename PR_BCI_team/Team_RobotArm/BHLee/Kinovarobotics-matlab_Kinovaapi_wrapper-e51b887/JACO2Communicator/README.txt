***************************************************************************
********************** JACO2 Communicator Class ***************************
***************************************************************************
This class allows you to connect MATLAB and Simulink to the Kinova JACO 2 
robot via USB.
Copyright 2017 The MathWorks, Inc.
 

******* Requirements ********

- MATLAB R2017b on Windows 10. The class has been tested on MATLAB R2017b 
running on a Windows 10 - 64 bit machine (It might work for other versions 
of Windows/MATLAB but you will need to recompile the class). 

- Kinova Driver for JACO2 installed on your machine. (If you have Kinova's 
Development Center installed and it can communicate with the robot then 
you are all set)

****** Using the class *****

If you have a Windows 10 - 64 bit machine follow these steps:
 
1) Add all the files to the MATLAB path by
Right clicking on the JACO2Communicator folder and selecting
"Add To Path/Selected Folders and Subfolders")

2) You should use the class from the same directory where the Kinova 
SDK files (shipped with the class) are.  
<YOUR_JACOCOM_FILES_PATH>/JACO2Communicator/JACO2SDK
If you want to run the class from other locations, you will need to add 
this directory to your Windows path variable.

3) Open the testJacoComm.m file and test connecting to the robot, 
getting sensor data from the robot, and moving the robot.     

If for some reason it doesn't work, you can try recompiling the MEX file by following 
these steps:

1. You need Microsoft Visual C++ 2015 Professional.  Select it by executing
in MATLAB 
    >> mex -setup C++ 
    >> mex -setup

2. Delete the JacoMexInterface.mexw64 file in
<YOUR_JACOCOM_FILES_PATH>/JACO2Communicator/Source/

3. Check the file <YOUR_JACOCOM_FILES_PATH>/JACO2Communicator/Source/compileMexInterface.m
as a reference of how to generate the MEX file


***** Optional *******

This class can also be used in Simulink. Open the "testJacoCommHarness.slx"
model to see an example of how to use it. By default it returns all sensor 
data so it's not intended for high control frequencies. 


