%% Modified by LBH V2 20200622
%% Modified by AHJ 20200728
%% initiate
bbci_acquire_bv('close');
startup_bbci;

jc = JacoComm;
connect(jc);
calibrateFingers(jc);

%% Query individual object properties, robotic arm initialization
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

% %% Load EEG_Mat file
% global EEG_MAT_DIR
% EEG_MAT_DIR = '';
% 
% dd = 'MI_ConvertedData\';
% filelist= {'20180605_msoh_reaching_MI.mat'};
% 
% %% Pretraining CSP with LDA
% Bandpass_Filter = [8 24]; % Bandpass filter range
% 
% % Offline Classifier Training
% [cnt, mrk, mnt] = eegfile_loadMatlab([dd filelist{1}]); % Loading training data
% 
% ival = [0 3000]; % Set interval
% 
% cnt_filt = proc_filtButter(cnt, 5, Bandpass_Filter); % Bandpass filtering in cnt data
% epo = cntToEpo(cnt_filt,mrk,ival); % Transform: cnt -> epo
% 
% [fv, Out.csp_w]=  proc_multicsp(epo, 6); % Perform CSP training #proc_multicsp(input, output)
% fv = proc_variance(fv); fv= proc_logarithm(fv); % Calculating variance and applying logarithm (CSP)
% 
% fv.classifier_param = {'RLDAshrink','prior', nan, 'store_prior', 1, 'store_means', 1, ...
%     'store_cov', 1, 'store_invcov', 1, 'scaling', 1};  % Extract classifier parameters
% 
% proc = {'wr_multiClass','policy','one-vs-all','coding','hamming'}; % one-vs-all all-pairs
% 
% Out.C = trainClassifier(fv, proc); % Classification results
% Out.out_eeg = applyClassifier(fv, 'wr_multiClass', Out.C); % Show output
% %% Online Initialization
% params = struct;
% state = bbci_acquire_bv('init', params);
% EEG_data = [];
% mnt = getElectrodePositions(state.clab);
% 
% % Add information in variable epo
% epo.clab = state.clab;
% epo.fs = state.fs;
% epo.title = filelist{1};
% 
% %% Get EEG Data
% global jaw

while 1
    %% cue
    Block = [1,2,3]; % cue sound
    for cue=1:4    
        if cue==1
        [A,AFs] = audioread('ball.wav');
        Ans = 1;
        sound(A,AFs);
        pause(1);
    elseif cue==2
        [A,BFs] = audioread('bottle.wav');
        Ans = 2;
        sound(A,AFs);
        pause(1);        
    elseif cue==3
        [A,CFs] = audioread('cup.wav');
        Ans = 3;
        sound(A,AFs);
        pause(1);        
    elseif cue==4
        [A,AFs] = audioread('censor-beep-4.wav');
        Ans = 4;
        sound(A,AFs);
        pause(1);        
        end
    
    
    Ans;
    
%     eog_ch = [1];
%     eog_th = 18; %임계값
%     time_window = 20;
%     eog_test(eog_ch, eog_th, time_window);
%     disp('Receiving brain signal')
%     % jaw
%     if jaw == 1
%     decision = Ans; 
%     % Connect and initialize EEG recording
%     bbci_acquire_bv('close');
%     EEG_MAT_DIR = '';
%     params=struct;
%     state=bbci_acquire_bv('init',params);
%     EEGData=[];
%     data=[];
%     mnt=getElectrodePositions(state.clab);
%     epo.clab = state.clab;
%     epo.fs = state.fs;
%     epo.title = filelist{1};
%     pause(1)
    
    
    
%     % 뇌파 저장 시작
%     
%     data = bbci_acquire_bv(state);
%     EEG_data = [EEG_data; data];
%     
%     % Sampling Rate: 250Hz, get data every 3 sec
%     if size(EEG_data,1) >= 750 % 250 *3 = 750
%         epo.x = EEG_data;
%         
%         % EEG filtering`
%         Wps= [42 49]/epo.fs*2;
%         [n, Ws]= cheb2ord(Wps(1), Wps(2),3, 40);
%         [filt.b, filt.a]= cheby2(n, 50, Ws);
%         epo = proc_filt(epo, filt.b, filt.a);
%         
%         % Feature Extraction and Classification
%         Classification_Result = MotorImagery_Online_Fn(epo, Bandpass_Filter, Out); %Classification result
%         
%         disp('Signal processing');
%         pause(2);
%         disp('Decoding signal');
        
        %% Decision
        switch Ans
            case 1
                disp('Ball');
                disp('robotic arm activation');
                
                % Robotarm control
                home_pos=jc.EndEffectorPose;
                current_pos=jc.EndEffectorPose;
                prev_pos=current_pos;
                
                x = 0.1;
                y = 0.5;
                
                desired_pos=[x; -y; 0.1; home_pos(4); home_pos(5); home_pos(6)];
                moveToCP(jc,desired_pos);
                
                pause(0.5);
                
                desired_pos=[x; -y; 0.2; home_pos(4); home_pos(5); home_pos(6)];
                moveToCP(jc,desired_pos);
                pause(0.5);
                
                goToHomePosition(jc);
                    
                pause(0.5);
                
            case 2
                disp('bottle');
                disp('robotic arm activation');
                
                % Robotarm control
                home_pos=jc.EndEffectorPose;
                current_pos=jc.EndEffectorPose;
                prev_pos=current_pos;
                
                x = 0.4;
                y = 0.35;
                
                desired_pos=[x; -y; 0.1; home_pos(4); home_pos(5); home_pos(6)];
                moveToCP(jc,desired_pos);
                
                pause(0.5);
                
                desired_pos=[x; -y; 0.2; home_pos(4); home_pos(5); home_pos(6)];
                moveToCP(jc,desired_pos);
                pause(0.5);
                
                goToHomePosition(jc);
                    
                pause(0.5);
                
            case 3
                disp('Cup');
                disp('robotic arm activation');
                
                % Robotarm control
                home_pos=jc.EndEffectorPose;
                current_pos=jc.EndEffectorPose;
                prev_pos=current_pos;
                
                x = 0.3;
                y = 0.3;
                
                desired_pos=[x; -y; 0.1; home_pos(4); home_pos(5); home_pos(6)];
                moveToCP(jc,desired_pos);
                
                pause(0.5);
                
                desired_pos=[x; -y; 0.2; home_pos(4); home_pos(5); home_pos(6)];
                moveToCP(jc,desired_pos);
                pause(0.5);
                
                goToHomePosition(jc);
                    
                pause(0.5);
                
            case 4
                disp('Backward');
                disp('robotic arm activation');
                
                % Robotarm control
                home_pos=jc.EndEffectorPose;
                current_pos=jc.EndEffectorPose;
                prev_pos=current_pos;
                
                x = 0.1;
                y = 0.7;
                
                desired_pos=[x; -y; 0.1; home_pos(4); home_pos(5); home_pos(6)];
                moveToCP(jc,desired_pos);
                
                pause(0.5);
                
                desired_pos=[x; -y; 0.2; home_pos(4); home_pos(5); home_pos(6)];
                moveToCP(jc,desired_pos);
                pause(0.5);
                
                goToHomePosition(jc);
                    
                pause(0.5);
                       
            otherwise
                % Classification error
                disp('Decoding error!');
                [A,AFs] = audioread('censor-beep-4.wav');
                sound(A,AFs);
        end
        
        %% Reset the parameters
        EEG_data = [];
        data=[];
        pause(1);
        
    end
    
end
%bbci_acquire_bv('close');
%end