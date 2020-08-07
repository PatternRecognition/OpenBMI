%% 이 프로그램은 로봇팔 각도로는 라디안, 길이로는 미터를 사용
%% 로봇 손가락의 경우 0~6800 사이의 값을 가진다.  
%% Joint position control이나 cartesian position control이나 
%% Joint는 절대 좌표를 기준으로 움직이나 cartesian은 상대좌표로 움직임 
%% 둘다 목표 지점에 닿으면 바로 되돌아온다는 문제가 있음
%% 코드에 손을 대지 않는다면 적당한 cartesian velocity/joint velocity를 써서 움직이는 방안이 가장 좋을듯.
%% 반복되는 움직임을 위한 모듈화가 내일은 진행되어야겠다.
clc; clear; close;

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
%home_pos=[0 0 0 0 0 0];
%% Desired_pos가 작업범위 바깥이라면 반드시 제외시켜야 한다.
home_pos=jc.EndEffectorPose;
current_pos=jc.EndEffectorPose;
prev_pos=current_pos;

setPositionControlMode(jc);
fCmd = 0*ones(3,1);
sendFingerPositionCommand(jc,fCmd);

desired_pos=[0.7; -0.25; 0.1; home_pos(4); home_pos(5); home_pos(6)];
moveToCP(jc,desired_pos);


pause(1);

setPositionControlMode(jc);
fCmd = 6000*ones(3,1);
sendFingerPositionCommand(jc,fCmd);


desired_pos=[0.7; -0.2; 0.2; home_pos(4); home_pos(5); home_pos(6)];
moveToCP(jc,desired_pos);


Wrist rotation
 jntVelCmd = [0;0;0;0;0;0;0.8]; %7DOF
 for i=1:260
     sendJointVelocityCommand(jc,jntVelCmd);
 end

 jntVelCmd = [0;0;0;0;0;0;-0.8]; %7DOF
 for i=1:260
     sendJointVelocityCommand(jc,jntVelCmd);
 end
 
 
 desired_pos=[0.7; -0.25; 0.1; home_pos(4); home_pos(5); home_pos(6)];
moveToCP(jc,desired_pos);

pause(1);

setPositionControlMode(jc);
fCmd = 0*ones(3,1);
sendFingerPositionCommand(jc,fCmd);



desired_pos=[0.2; -0.2; 0.4; home_pos(4); home_pos(5); home_pos(6)];
moveToCP(jc,desired_pos);
length=sqrt((desired_pos(1)-current_pos(1)).^2+(desired_pos(2)-current_pos(2)).^2+(desired_pos(3)-current_pos(3)).^2);
CartVel=0.2;
direction=CartVel*[desired_pos(1)-current_pos(1),desired_pos(2)-current_pos(2),desired_pos(3)-current_pos(3)]/length;
time=round(length*200/CartVel);

time=400;

%% i값과 속도값을 적절히 조절하는 것이 관건 
%% Send cartesian velocity
%% i=200 은 1초에 해당
%% 앞의 셋은 미터지만 뒤는 라디안. 5ms마다 입력이 들어가므로 절대 0.01을 넘는 값을 인가하지 말것
%% 뒤의 셋의 경우 대개 small actuator의 한계값과 관련이 있어 

error=sqrt((desired_pos(1)-current_pos(1)).^2+(desired_pos(2)-current_pos(2)).^2+(desired_pos(3)-current_pos(3)).^2);

while error>0.01
    temp_pos=jc.EndEffectorPose;
    error=sqrt((desired_pos(1)-temp_pos(1)).^2+(desired_pos(2)-temp_pos(2)).^2+(desired_pos(3)-temp_pos(3)).^2);
    CartVel=0.2;
    direction=CartVel*[desired_pos(1)-temp_pos(1),desired_pos(2)-temp_pos(2),desired_pos(3)-temp_pos(3)]/error;
    CartVelCmd = [direction(1);direction(2);direction(3);0;0;0];
    sendCartesianVelocityCommand(jc,CartVelCmd);
end
for i=1:time
temp_pos=jc.EndEffectorPose;
CartVelCmd = [direction(1);direction(2);direction(3);0;0;0];
sendCartesianVelocityCommand(jc,CartVelCmd);
end

current_pos=jc.EndEffectorPose;

%% 이후에 방향 제어라든지 물건을 집는다던지 하는 동작을 수행하여야 한다.
setPositionControlMode(jc);
fCmd = 0*ones(3,1);
sendFingerPositionCommand(jc,fCmd);

pause(3);

setPositionControlMode(jc);
goToHomePosition(jc);
