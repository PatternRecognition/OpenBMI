%testJacoComm - Example script that shows how to use the JACO2 
% USB communication class for MATLAB and Simulink
% BEFORE USING THIS CLASS, CHECK README.txt
% Copyright 2017 The MathWorks, Inc.


%% Be sure that you are on the directory where the SDK files are. That is
%  <YOUR_JACOCOM_FILES_PATH>/JACO2Communicator/JACO2SDK
%  If you want to run the class from other locations, 
%  you will need to add this directory to your Windows path variable.
% addpath(genpath('C:\Users\shovington\Documents\GitHub\matlab_Kinovaapi_wrapper'))
% cd 'C:\Users\shovington\Documents\GitHub\matlab_Kinovaapi_wrapper\JACO2Communicator\JACO2SDK'


%% Create an instance of the class (you can only create ONE instance)
jc = JacoComm


%% Connect to the robot
connect(jc);
% Verify the object has all the sensor data 
jc


%% When you are done using the class, close the library by executing
% >> disconnect(jc) 


%% Calibrate fingers
calibrateFingers(jc);


%% Query individual object properties
jc.JointPos
%%
jc.JointVel
%%
jc.JointTorque
%%
jc.JointTemp
%%
jc.FingerPos
%%
jc.FingerVel
%%
jc.FingerTorque
%%
jc.FingerTemp
%%
jc.EndEffectorPose
%%
jc.EndEffectorWrench
%%
jc.ProtectionZone
%%
jc.EndEffectorOffset
%%
jc.DOF
%%
jc.TrajectoryInfo


%% Methods to query joint and finger values all at once
pos = getJointAndFingerPos(jc)
%%
vel = getJointAndFingerVel(jc)
%%
torque = getJointAndFingerTorque(jc)

%%
temp = getJointAndFingerTemp(jc)


%% Set position control mode (default mode after powering on the robot)
setPositionControlMode(jc);


%% Go to home position (function returns when it reaches the home position);
goToHomePosition(jc);


%% Set torque control mode
% You can move the robot and test the gravity compensation. When you are
% finished, execute again setPositionControlMode(jc);
setTorqueControlMode(jc);


%% Start admittance mode
% You can move the robot in admitance mode. When you are
% finished, execute jc.StopForceControl;
jc.StartForceControl;


%% Send joint positions
%q_home = [4.806;2.924;1.004;4.204;1.445;1.323]; %6DOF
q_home = [4.9409;2.8396;0.0016;0.7581;4.6342;4.4963;5.0252]; %7DOF
jntCmd = q_home;
sendJointPositionCommand(jc,jntCmd);


%% Send finger positions
fCmd = 6000*ones(3,1);
sendFingerPositionCommand(jc,fCmd);
%%
fCmd = 0*ones(3,1);
sendFingerPositionCommand(jc,fCmd);


%% Send joint and finger position
%q_home = [4.806;2.924;1.004;4.204;1.445;1.323]; %6DOF
q_home = [4.9409;2.8396;0.0016;0.7581;4.6342;4.4963;5.0252]; %7DOF
jntCmd = q_home;
fCmd = 0*ones(3,1);
sendJointAndFingerPositionCommand(jc,jntCmd,fCmd);


%% Send joint velocities
%jntVelCmd = [0;0;0;0;0;0.8]; %6DOF
jntVelCmd = [0;0;0;0;0;0;0.8]; %7DOF
for i=1:200
    sendJointVelocityCommand(jc,jntVelCmd);
end


%% send joint torques
setTorqueControlMode(jc);
%jntTorqueCmd = [1.5;0.0;0;0;0;0.8]; %6DOF
jntTorqueCmd = [0;0;0;0;0;0;-0.8]; %7DOF
for i=1:100
   sendJointTorqueCommand(jc,jntTorqueCmd); 
   pause(0.01)
end


%% Send cartesian positions
setPositionControlMode(jc);
CartPosCmd = [0.05;0;0;0;0;0];
sendCartesianPositionCommand(jc,CartPosCmd)


%% Send cartesian velocity
for i=1:100
CartVelCmd = [0;0;-0.1;0;0;0];
sendCartesianVelocityCommand(jc,CartVelCmd);
end

%% Set End Effector Offset
offsetCmd = [0;0;0;0];
jc.setEndEffectorOffset(offsetCmd);


%% Run gravity calibration (will block until it finishes, not interruptible!)
% runGravityCalibration(jc);


%% Set a protection zone with a speed limit
% See API documentation to know what to send in Zone Cmd
% Search for setProtectionZone
ZoneCmd = [0;-0.52;0;0;0;0;...
           0;0;0;...
           0.5;0;0;...
           0.5;-0.52;0;...
           0.63;...
           0.01;0.01];
jc.setProtectionZone(ZoneCmd)

%% Delete all protection zones
jc.EraseAllProtectionZones


%% Disconnect from robot and unload library
setPositionControlMode(jc);
disconnect(jc);



