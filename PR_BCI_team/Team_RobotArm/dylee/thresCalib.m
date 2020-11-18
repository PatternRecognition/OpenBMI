function [thres1, thres2] = thresCalib(dd, filelist, ival, filt_band_eeg, filt_band_emg, subChannel_eeg, subChannel_emg, main_class)
%% Parameter setting for EEG
% filt_band_eeg = [8 30]; % [0.3 3]Hz, [0.3 35]Hz -> reference: Decoding natural reach-and-grasp actions from human EEG 
% REST threshold: [-6000 -3000], activation state threshold: [0 4000] 
% ival = [0 4000]; % for MOTOR IMAGERY % 250Hz sampling? Should be same for EEG and EMG 
% subChannel_eeg = [11 12 13 15 16 17 18 20 21 22 23 43 44 45 47 48 49 51 52 53]; %20 out of 60 channels
% main_class = {'Cylindrical', 'Spherical', 'Lumbrical'};

% Parameter setting for EMG 
% filt_band_emg = [10 225]; % EMG [10 225]Hz  
% subChannel_emg = [71]; % Total 71ch(60 EEG, 4 EOG, 7 EMG) <- 6 EMG channels 

%% converted data file to cnt - EMG, EEG all same  
[cnt, mrk, mnt]=eegfile_loadMatlab([dd filelist], 'fs', 500); % Load cnt, mrk, mnt variables to Matlab (grasp data) 
% separate cnt into EEG's and EMG's 
%for EEG 
cnt_eeg = proc_commonAverageReference(cnt, subChannel_eeg); 
cnt_eeg = proc_selectChannels(cnt_eeg, subChannel_eeg); % Channel Selection
% for EMG 
cnt_emg = proc_commonAverageReference(cnt, subChannel_emg); 
cnt_emg = proc_selectChannels(cnt_emg, subChannel_emg); 

%% IIR Filter (5th butterworth filtter -> reference: Decoding natural reach-and-grasp actions) 
%for EEG 
cnt_eeg = proc_filtButter(cnt_eeg, 5, filt_band_eeg);  % Band-pass filtering 
%for EMG
cnt_emg = proc_filtButter(cnt_emg, 5, filt_band_emg);  % Band-pass filtering 

%% cnt to epo
% make epoch for EMG 
epo_emg = cntToEpo(cnt_emg, mrk, ival);
epo_emg = proc_selectClasses(epo_emg, main_class); 
% make epoch for EEG
epo_eeg = cntToEpo(cnt_eeg, mrk, ival); 
epo_eeg = proc_selectClasses(epo_eeg, main_class); 

%% create threshold 
i = 0; 
for i = 1:size(subChannel_emg, 2) % number of selected EMG channels 
    temp.x = (epo_emg.x(:, :, i))';
%     temp.x = abs(temp.x); 
    thres_EMG{i} = rms(temp.x(i, :)); 

end

i = 0; 
for i = 1:size(subChannel_eeg, 2) % number of selected EEG channels 
    temp.x = (epo_eeg.x(:, :, i))';
%     temp.x = abs(temp.x); 
    thres_EEG{i} = rms(temp.x(i, :)); 

end
thres1 = thres_EEG; thres2 = thres_EMG; 


end
    

