clear all; clc; close all;
OpenBMI; % Edit the variable BMI if necessary
global BMI; % check BMI directories

%% PARADIGM MODULE
opt={'Num_trial',10,'Time_STI',2,'Time_ISI',4,'Time_Jitter',0.1,'Num_Screen',1, ...
    'Screen_Size',[1920 1080],'Screen_Type','full', 'Beep', 'on'}
BMI.IO_ADDR=hex2dec('C010'); %check trigger
addpath(genpath('C:\Psychtoolbox'));
paradigm_MI(opt);         