%% SSVEP validation
clear all; clc; close all;
%% initialization
DATADIR = 'WHERE\IS\DATA';
%% ERP
SSVEPDATA = 'EEG_SSVEP.mat';
STRUCTINFO = {'EEG_SSVEP_train', 'EEG_SSVEP_test'};
SESSIONS = {'session1', 'session2'};
TOTAL_SUBJECTS = 54;

%% INITIALIZATION
FS = 100;

%init
params = {'time', 4;...
'freq' , 60./[5, 7, 9, 11];...
'fs' , FS;...
'band' ,[0.5 40];...
'channel_index', [23:32]; ...
'time_interval' ,[0 4000]; ...
'marker',  {'1','up';'2', 'left';'3', 'right';'4', 'down'}; ...
};

%% validation
for sessNum = 1:length(SESSIONS)
    session = SESSIONS{sessNum};
    fprintf('\n%s validation\n',session);
    for subNum = 1:TOTAL_SUBJECTS
        subject = sprintf('s%d',subNum);
        fprintf('LOAD %s ...\n',subject);
        
        data = importdata(fullfile(DATADIR,session,subject,SSVEPDATA));
        
        CNT{1} = prep_resample(data.(STRUCTINFO{1}), FS,{'Nr', 0});
        CNT{2} = prep_resample(data.(STRUCTINFO{2}), FS,{'Nr', 0});
        ACC.SSVEP(subNum,sessNum) = ssvep_performance(CNT, params);
        fprintf('%d = %f\n',subNum, ACC.SSVEP(subNum,sessNum));
        clear CNT
    end
end



