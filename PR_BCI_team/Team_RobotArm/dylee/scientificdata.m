%% Scientific Data
% ylabel: channel/ xlabel: time(s)_day1, day2, day3
% Subject N (dslim), 1 single(trial_reaching_realMove_Up)

clc; close all; clear all;

%% file -> day1, day2, day3
% input data: please convert include the EMG signal data
dd = 'C:\Users\Doyeunlee\Desktop\Analysis\rawdata\scientific data_EMG\';
filelist={'GIGA_20190710_dslim_reaching_realMove'};

% ival: -5 ~ 4.5s (0~4s: realMove/ -5~0s: preparing/ 4~4.5s: after)
ival = [-500 4500];
% channel: total 71 channels (EEG channel: 60, EOG channel: 4 channel, EMG
% channel: 7)
chans = [1 71];


for i = 1:length(filelist)
    [cnt, mrk, mnt] = eegfile_loadMatlab([dd filelist{i}]);
    
    epo = cntToEpo(cnt,mrk,ival);
    
    % class: Up
    epoUp=proc_selectClasses(epo,{'Up'});
    
    % single trial-> Up class 
    % epoUp.x --> time x channel x trial
    epoUp.x = epoUp.x(:,:,[35 40]);
    
    % Show in GUI: time x channels    
    showGuiEEG(epoUp, mrk, chans);
    
end

