%% CALIBRATION - multiClass grasping actions (realMove data only, EMG pattern classification approach) 
clear all; close all; clc;
tic

%% Load Data
dd = 'C:\Users\Doyeunlee\Desktop\Analysis\01_Raw data\';
filelist = 'session1_sub1_multigrasp_MI_EMG'; % this code is only for realMove data 

% for image dataset building
% order_count = 0; % from 0 ~  

%% Parameter setting for EEG
filt_band_eeg = [0.1 40]; % [0.3 3]Hz, [0.3 35]Hz -> reference: Decoding natural reach-and-grasp actions from human EEG 
% ival_rest = [-7000 -3000];
ival = [0 4000]; % for MOTOR IMAGERY % 250Hz sampling? Should besame for EEG and EMG 
% subChannel_eeg = [11 12 13 15 16 17 18 20 21 22 23 43 44 45 47 48 49 51 52 53]; %20 out of 60 channels
subChannel_eeg = [6 7 8 9 11 12 13 15 16 17 18 20 21 22 23 25 26 27 28 39 40 10 ...
    43 44 45 47 48 49 51 52 53 55 56 57]; %34 out of 60 channels
main_class = {'Cylindrical','Spherical','Lumbrical'};

% Parameter setting for EMG
filt_band_emg = [10 225]; % EMG [10 225]Hz  
subChannel_emg = [65:70]; % 4 EMG corresponding to hand and wrist movements only (grasp actions) 
% Total 71ch(60 EEG, 4 EOG, 7 EMG) <- 6 EMG channels [65:70] = 6 EMG 

%% load thres.mat; % threshold data of EEG and EMG signals, respectively.
ival_thres = [-6000 -3000]; 
bline_list = [0.75 1.0 1.25 1.5];

for bline_count = 1:size(bline_list, 2)
    bline_val = bline_list(bline_count); 
    [thres_EEG, thres_EMG] = thresCalib(dd, filelist, ival_thres, filt_band_eeg, filt_band_emg, ...
        subChannel_eeg, subChannel_emg, main_class); 

    %% converted data file to cnt - EMG, EEG all same  
    [cnt, mrk, mnt]=eegfile_loadMatlab([dd filelist], 'fs', 500); % Load cnt, mrk, mnt variables to Matlab (grasp data) 
    % separate cnt into EEG's and EMG's 
    %for EEG 
    cnt_eeg = proc_commonAverageReference(cnt, subChannel_eeg); 
    cnt_eeg = proc_selectChannels(cnt_eeg, subChannel_eeg); % Channel Selection
    cnt_emg = proc_selectChannels(cnt, subChannel_emg); % Channel Selection
    
    epo_emg = cntToEpo(cnt_emg, mrk, ival);
    epo_emg = proc_selectClasses(epo_emg, {'Cylindrical','Spherical','Lumbrical','Rest'});
    
    epoCylindrical=proc_selectClasses(epo_emg,{'Cylindrical'});
    epoSpherical = proc_selectClasses(epo_emg,{'Spherical'});
    epoLumbrical = proc_selectClasses(epo_emg,{'Lumbrical'});
    epoRest = proc_selectClasses(epo_emg,{'Rest'});
    
    
    % for EMG 
%     cnt_emg = proc_commonAverageReference(cnt, subChannel_emg); 
%     cnt_emg = proc_selectChannels(cnt_emg, subChannel_emg); 

    %% IIR Filter (4th butterworth filtter -> reference: Decoding natural reach-and-grasp actions) 
    %for EEG 
    cnt_eeg = proc_filtButter(cnt_eeg, 3, filt_band_eeg);  % Band-pass filtering
    
    % FIR FIlter - Option 1 
%     [y1, D1] = bandpass_computeGPU(cnt.x, filt_band, cnt.fs, 'ImpulseResponse', 'fir'); 
%     cnt.x = gather(y1);

    % FIR Filter - Option 2
%     Fs = cnt_eeg.fs;  % Sampling Frequency
%     N    = 16;     % Order
%     Fc1  = filt_band_eeg(1);    % First Cutoff Frequency
%     Fc2  = filt_band_eeg(2);    % Second C utoff Frequency
%     flag = 'scale';  % Sampling Flag
%     % Create the window vector for the design algorithm.
%     win = blackman(N+1);
%     % Calculate the coefficients using the FIR1 function.
%     b  = fir1(N, [Fc1 Fc2]/(Fs/2), 'bandpass', win, flag);
%     Hd = dfilt.dffir(b);
%     cnt_eeg.x = filter(Hd, cnt_eeg.x);
    
    %for EMG
%     cnt_emg = proc_filtButter(cnt_emg, 5, filt_band_emg);  % Band-pass filtering 

    %% MNT_ADAPTMONTAGE - Adapts an electrode montage to another electrode set
    mnt = mnt_adaptMontage(mnt, cnt); %MNT_ADAPTMONTAGE - Adapts an electrode montage to another electrode set

    %% Noise elimination 
    % cnt_eeg.x = medfilt1(cnt_eeg.x); % median values, to remove the spike noise on data 
    % cnt_emg.x = medfilt1(cnt_emg.x); 
    
    %% Normalization, rescale, smoothing filter 
%     cnt_eeg.x = normalize(cnt_eeg.x); 
%     cnt_emg.x = normalize(cnt_emg.x); 
%     cnt_eeg.x = hampel(cnt_eeg.x, 2000);
%     cnt_emg.x = hampel(cnt_emg.x, 2000); 

    %% cnt to epoch    
%     w_size = 1010; %(ms), window size 400, step size 150, overlap size = 250(ms) -> 25 segments  
%     step_size = 130;
%     overlap_size = w_size - step_size; % 250ms 
%     org_ival = ival; % [0 4000]= 4000ms
%     seg_length = abs(org_ival(2) - org_ival(1));
%     n_segment = ((seg_length - w_size)/(w_size - overlap_size)) + 1; % 25 segments
%     
%     h = 0;
%     for h = 1:ch

%     k = 0; 
%     for k = 1:size(main_class, 2) % repeat 3 classes 
%             i = 0; 
%             for i = 1:n_segment % repeat 25 segments
%                 s_point = ((i-1)*step_size) + org_ival(1);
%                 end_point = s_point + w_size; 
%                 sub_ival = [s_point end_point];

                % make epoch for EMG 
    
    classes=size(epo_emg.className,2);
    trial=50;
    for ii =1:classes
            if strcmp(epo_emg.className{ii},'Rest')
                epoRest=proc_selectClasses(epo_emg,{epo_emg.className{ii}});
                epoRest.x=datasample(epoRest.x,trial,3,'Replace',false);
                epoRest.y=datasample(epoRest.y,trial,2,'Replace',false);
            else
                epo_check(ii)=proc_selectClasses(epo_emg,{epo_emg.className{ii}});
                % random sampling
                epo_check(ii).x=datasample(epo_check(ii).x,trial,3,'Replace',false);
                epo_check(ii).y=datasample(epo_check(ii).y,trial,2,'Replace',false);
            end
        end
        if classes<4
            epo_check(size(epo_check,2)+1)=epoRest;
        end
        
        % concatenate the classes
        for ii=1:size(epo_check,2)
            if ii==1
                concatEpo=epo_check(ii);
            else
                concatEpo=proc_appendEpochs(concatEpo,epo_check(ii));
            end
        end
    % make epoch for EEG
%     epo_eeg = cntToEpo(cnt_eeg, mrk, ival); 
%     epo_eeg = proc_selectClasses(epo_eeg, main_class{k}); 

    for j = 1:50 % number of trials 
        % create muscle patterns 
        temp.muscle_fv = epo_emg.x(:, :, j); 
        temp.ch1 = temp.muscle_fv(:, 1); temp.ch2 = temp.muscle_fv(:, 2);
        temp.ch3 = temp.muscle_fv(:, 3); temp.ch4 = temp.muscle_fv(:, 4);
        temp.ch5 = temp.muscle_fv(:, 5); temp.ch6 = temp.muscle_fv(:, 6);

        temp.rms_fv(1) = rms(temp.ch1); temp.rms_fv(2) = rms(temp.ch2);
        temp.rms_fv(3) = rms(temp.ch3); temp.rms_fv(4) = rms(temp.ch4);
        temp.rms_fv(5) = rms(temp.ch5); temp.rms_fv(6) = rms(temp.ch6);


        % build EEG epoches per each class
%         max_size = size(epo_eeg.x(:, :, j), 1); 
%         temp.x{j}(1:max_size, :, i) = epo_eeg.x(:, :, j);

        ch_num = 0; 
%         for ch_num = 1:6 % number of channels 
%             if temp.rms_fv(ch_num) > bline_val * thres_EMG{ch_num} % ???% of rest state rms(EMG value)
%                 temp.emg_ptrn(i, ch_num, j) = 1;
%             else
%                 temp.emg_ptrn(i, ch_num, j) = 0; 
% 
%             end
%         end
    end % for j   
%             end % for i 
%     emg_ptrn{k} = temp.emg_ptrn; 
%     temp.x_all{k} = temp.x; 

%     end % for k 
end

X = abs(temp.muscle_fv);
X_1 = movmean(X, 30);
plot(X_1);
hold on;
xlim([0 4000]);
X = abs(temp.rms_fv);
X_1 = movmean(X, 30);
plot(X_1);

    