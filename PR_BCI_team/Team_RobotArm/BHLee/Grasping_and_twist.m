function Grasping_and_twist

%% Grasping and twisting Function with EEG recording
clear; close; clc;
%% Initializing EEG recording device
bbci_acquire_bv('close');
startup_bbci;

%% RoboticArm communication channel opened
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
pos = getJointAndFingerPos(jc);
vel = getJointAndFingerVel(jc);

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

%% Offline model training
global EEG_MAT_DIR
EEG_MAT_DIR = '';

dd = 'MotorImagery Converted Data\';
filelist= {'20191113_demo_bhkwon'};

%%
Bandpass_Filter = [8 40];

% ----------------------------------------------------------------------
% Offline Classifier Training
[cnt, mrk, mnt] = eegfile_loadMatlab([dd filelist{1}]);

ival = [0 3000];

cnt_filt = proc_filtButter(cnt, 5, Bandpass_Filter);
epo = cntToEpo(cnt_filt,mrk,ival);

[fv, Out.csp_w]=  proc_multicsp(epo, 3);
fv = proc_variance(fv); fv= proc_logarithm(fv);

fv.classifier_param = {'RLDAshrink','prior', nan, 'store_prior', 1, 'store_means', 1, ...
    'store_cov', 1, 'store_invcov', 1, 'scaling', 1};

proc = {'wr_multiClass','policy','one-vs-all','coding','hamming'}; % one-vs-all all-pairs

Out.C = trainClassifier(fv, proc);
Out.out_eeg = applyClassifier(fv, 'wr_multiClass', Out.C);

%%
% ----------------------------------------------------------------------
% Onling Initialization
params = struct;
state = bbci_acquire_bv('init', params);
EEG_data = [];
mnt = getElectrodePositions(state.clab);

epo.clab = state.clab;
epo.fs = state.fs;
epo.title = filelist{1};
Ans = zeros(1,3);
%%
% Get EEG data
py.SharedDemo_6_Vision_LBH.getPerspectiveMine();

coord=py.SharedDemo_6_Vision_LBH.mainRunning(1);
coord=double(py.array.array('d',py.numpy.nditer(coord)));
coord=uint8(coord/5);
pause(1);

home_pos=jc.EndEffectorPose;
current_pos=jc.EndEffectorPose;
prev_pos=current_pos;

targetX = coord(1);
targetY = coord(2);
targetX = double(targetX);
targetY = double(targetY);
x_origin = (102 - targetX)/100
y_origin = (5+targetY)/100

desired_pos=[x_origin; -y_origin; 0.1; home_pos(4); home_pos(5); home_pos(6)];
moveToCP(jc,desired_pos);

while 1
    %% cue
    [A,AFs] = audioread('grasp.mp3');
    sound(A,AFs);
    pause(2);
    
    eog_ch = [1 ,31];
    eog_th = 18;
    time_window = 20;
    
    eog_test(eog_ch, eog_th, time_window);
    disp('Receiving brain signal')
    
    pause(2);
    
    %% Initializing EEG recording
    bbci_acquire_bv('close');
    EEG_MAT_DIR = '';
    params=struct;
    state=bbci_acquire_bv('init',params);
    EEGData=[];
    mnt=getElectrodePositions(state.clab);
    epo.clab = state.clab;
    epo.fs = state.fs;
    epo.title = filelist{1};
    pause('on')
    
    [B,BFs] = audioread('censor-beep-4.wav');
    sound(B,BFs);
    pause(4);
    
    data = bbci_acquire_bv(state);
    EEG_data = [EEG_data; data];
    
    % Sampling Rate: 250Hz, get data every 3 sec
    if size(EEG_data,1) >= 750
        epo.x = EEG_data;
        
        % EEG filtering
        Wps= [42 49]/epo.fs*2;
        [n, Ws]= cheb2ord(Wps(1), Wps(2),3, 40);
        [filt.b, filt.a]= cheby2(n, 50, Ws);
        epo = proc_filt(epo, filt.b, filt.a);
        
        % Feature Extraction and Classification
        Classification_Result = MotorImagery_Online_Fn(epo, Bandpass_Filter, Out);
        
        disp('Signal processing');
        pause(2);
        disp('Decoding signal');
        pause(2);
        
        
        disp('Grasp');
        
        disp('robotic arm activation');
        pause(2)
        
        setPositionControlMode(jc);
        fCmd = 4000*ones(3,1);
        sendFingerPositionCommand(jc,fCmd);
        
        pause(1);
        
        desired_pos=[0.6; -0.2; 0.15; home_pos(4); home_pos(5); home_pos(6)];
        moveToCP(jc,desired_pos);
        
        pause(1);
        
    end
    
    [B,BFs] = audioread('drinkwater.mp3');
    sound(B,BFs);
    pause(2);
    
    eog_ch = [1 ,31];
    eog_th = 18;
    time_window = 20;
    
    
    
    eog_test(eog_ch, eog_th, time_window);
    disp('Receiving brain signal')
    
    bbci_acquire_bv('close');
    EEG_MAT_DIR = '';
    params=struct;
    state=bbci_acquire_bv('init',params);
    EEGData=[];
    data=[];
    mnt=getElectrodePositions(state.clab);
    epo.clab = state.clab;
    epo.fs = state.fs;
    epo.title = filelist{1};
    
    [B,BFs] = audioread('censor-beep-4.wav');
    sound(B,BFs);
    pause(4);
    
    data = bbci_acquire_bv(state);
    EEG_data = [EEG_data; data];
    
    if size(EEG_data,1) >= 750
        epo.x = EEG_data;
        
        % EEG filtering
        Wps= [42 49]/epo.fs*2;
        [n, Ws]= cheb2ord(Wps(1), Wps(2),3, 40);
        [filt.b, filt.a]= cheby2(n, 50, Ws);
        epo = proc_filt(epo, filt.b, filt.a);
        
        % Feature Extraction and Classification
        Classification_Result = MotorImagery_Online_Fn(epo, Bandpass_Filter, Out);
        
        disp('Signal processing');
        pause(2);
        disp('Decoding signal');
        pause(2);
        
        desired_pos=[0.6; 0.15; 0.15; home_pos(4); home_pos(5); home_pos(6)];
        moveToCP(jc,desired_pos);
        
        jntVelCmd = [0;0;0;0;0;0;0.2]; %7DOF
        for i=1:300
            sendJointVelocityCommand(jc,jntVelCmd);
        end
        
        pause(5);
        
        jntVelCmd = [0;0;0;0;0;0;-0.2]; %7DOF
        for i=1:260
            sendJointVelocityCommand(jc,jntVelCmd);
        end
        
        pause(1);
        
        desired_pos=[x_origin; -y_origin; 0.08; home_pos(4); home_pos(5); home_pos(6)];
        moveToCP(jc,desired_pos);
        
        pause(1);
        
        setPositionControlMode(jc);
        fCmd = 0*ones(3,1);
        sendFingerPositionCommand(jc,fCmd);
        
        desired_pos=[x_origin; -y_origin; 0.2; home_pos(4); home_pos(5); home_pos(6)];
        moveToCP(jc,desired_pos);
        
        pause(1);
        
        goToHomePosition(jc);
        
        break;
    end
    
    else
        disp('Decoding Error!');
        disp('Receiving brain signal again....');
        
        pause(2);
        
        desired_pos=[x_origin; -y_origin; 0.07; home_pos(4); home_pos(5); home_pos(6)];
        moveToCP(jc,desired_pos);
        
        pause(1);
        
        setPositionControlMode(jc);
        fCmd = 0*ones(3,1);
        sendFingerPositionCommand(jc,fCmd);
        
        pause(1);
        
        desired_pos=[x_origin; -y_origin; 0.2; home_pos(4); home_pos(5); home_pos(6)];
        moveToCP(jc,desired_pos);
        
        goToHomePosition(jc);
        
        pause(1);
        
        break
        
        bbci_acquire_bv('close'); 
end
end




