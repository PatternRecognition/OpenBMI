clear all; clc; close all;
%%
Screen('Preference', 'SkipSyncTests', 1);
n_seq = 10; 
screens = Screen('Screens'); 
n_scr = 3; 
stimulus_time=0.05; 
interval_time=0.135; 
rs_time = 0;
port = '2FF8';
scr_size = 'full';
frequency = 8000;

% classifier 
chanSel=1:32;
TimeIval=[-200 800];
baselineTime=[-200 0];
selectedTime=[0 800];
numFeature=10;
selectedFreq=[0.5 40];

% task
train_task = [15, 13, 4, 1, 25, 2, 19, 13, 22, 7, ... 
    3, 29, 12, 33, 11, 14, 5, 21, 18, 26];
test_task = [22, 29, 14, 17, 10, 19, 13, 1, 21, 23, ...
    31, 28, 31, 5, 9, 11, 15, 36, 5, 1];

% chanSel=[1 2 4 11 12 14 16 17 26 31];
eog_ch = [6 ,10];
eog_th = 50;
%% Subjects info
sub_num = 7;
sub_name = 'hsjeong';
path = 'C:\Users\KHJ-work\Documents\MATLAB\Application_Demo\erp';
files = {'train', 'test'};
datDir=fullfile(path, sprintf('subject%d_%s',sub_num, sub_name),'session2');

%% EOG Test
eog_ch = [6 ,10];
eog_th = 24;
time_window = 20;
%%
eog_test(eog_ch, eog_th, time_window);
%% Practice
soundImagery(0, n_scr, port);
%% sound generator for practice
soundGeneration(.5);
%% plotting for practice
file = 'practice';
marker = {'1', 'Listening' ;'2', 'Imagery'};
p300_plotting(fullfile(datDir, file), marker);
%% Session 1 2 3 Train
copy_task = train_task;
pa_opt = {'port', port;'text', copy_task; 'nSequence',n_seq; 'screenNum',n_scr;...
    'sti_Times',stimulus_time; 'sti_Interval',interval_time;'screenSize', scr_size;...
    'resting', rs_time; 'frequency', frequency;'online', false};
norm_train(pa_opt);
%% Session 1 2 3 plotting for train/test
% file = 'train';
file = 'test';
marker = {'1', 'non-target' ;'2', 'target'};
p300_plotting(fullfile(datDir, file), marker);
%% Session 1 2 3 Test
copy_task = test_task;
pa_opt = {'port', port;'text', copy_task; 'nSequence',n_seq; 'screenNum',n_scr;...
    'sti_Times',stimulus_time; 'sti_Interval',interval_time;'screenSize', scr_size;...
    'resting', rs_time; 'frequency', frequency;'online', false};
norm_train(pa_opt);
%% ÀüÀÚÀü Train
txt = 'KOREA_UNIVERSITY';
pa_opt = {'port', port;'text',txt; 'nSequence',5; 'screenNum',n_scr;...
    'sti_Times',stimulus_time; 'sti_Interval',interval_time;'screenSize', scr_size;...
    'resting', rs_time; 'frequency', frequency;'online', false};
application_train(pa_opt);
%% plotting for practice
file = 'dot_train';
marker = {'1', 'non-target' ;'2', 'target'};
p300_plotting(fullfile(datDir, file), marker);
%% TEST
txt='NEURAL_NET';
pa_opt = {'port', port;'text',txt; 'nSequence',n_seq; 'screenNum',n_scr;...
    'sti_Times',stimulus_time; 'sti_Interval',interval_time;'screenSize', scr_size;...
    'resting', rs_time; 'frequency', frequency;'online', true};
application_demo(pa_opt);
%%
clfs = make_classifiers(fullfile(datDir, 'dot_train'), chanSel);
%% Online scripts
on_opt = {'segTime',TimeIval;'baseTime',baselineTime; 'selTime',selectedTime;...
    'nFeature',numFeature;'channel',chanSel;'clf_param',clfs; ...
    'selectedFreq', selectedFreq; 'eog_ch', eog_ch; 'eog_th',eog_th;'eog_window', time_window};
res =p300_online_threeclasses(on_opt);
%%
p300_plotting(fullfile(datDir, files{1}), {'1', 'non';'2', 'tar'});