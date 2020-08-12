function Grasping_and_Twisting

%% Grasping and twisting

clear; close; clc;
%% initiate
bbci_acquire_bv('close');
startup_bbci;

jc = JacoComm;
connect(jc);
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
%% 팔 관절 각도 값과 손가락 관절 각도 값을 리턴
pos = getJointAndFingerPos(jc);
%%
%% 팔 관절 속도 값과 손가락 관절 속도 값을 리턴
vel = getJointAndFingerVel(jc);
%%
%% 팔 관절 토크 값과 손가락 관절 토크 값을 리턴
torque = getJointAndFingerTorque(jc);
temp = getJointAndFingerTemp(jc);

setPositionControlMode(jc);
goToHomePosition(jc);

current_pos=jc.EndEffectorPose;
home_pos=jc.EndEffectorPose;
previous_pos=current_pos;


setPositionControlMode(jc);
fCmd = 0*ones(3,1);
sendFingerPositionCommand(jc,fCmd);