%% Main Experiments parameters
clear all; clc; close all;
Screen('Preference', 'SkipSyncTests', 1);


n_seq = 6;
screens = Screen('Screens');   
n_scr =1; 
stimulus_time=0.05; 
interval_time=0.135; 
rs_time = 20; 
port = 'DFF8';
scr_size = [1000 550];
frequency = 8000;
chanSel=1:16;
TimeIval=[-200 800];
baselineTime=[-200 0];
selectedTime=[0 800];
numFeature=10;
selectedFreq=[0.5 40];
train_task = [11, 15, 18, 5, 1, 21, 14, 9, 22, 5, 18, 19, 9, 20, 25]; 
test_task = [11, 15, 18, 5, 1, 21, 14, 9, 22, 5, 18, 19, 9, 20, 25]; %KOREAUNIVERSITY 쓰려고
field={'x','t','fs','y_dec','y_logic','y_class','class', 'chan'};

% Subjects info
sub_num = 0;
sub_name = 'practicesp';
path = 'C:\Users\cvpr\Desktop\sp_experiment_0715\Experiment190801Start';
datDir=fullfile(path, sprintf('subject%d_%s',sub_num, sub_name));
files = {'training'};

%% 패러다임1 (RASP paradigm) 프로그램
fNum = 1;
copy_task = train_task; 
paradigm1({'port', port;'text',copy_task; 'nSequence',n_seq;...
    'screenNum',n_scr;'sti_Times', stimulus_time; 'sti_Interval',interval_time;...
    'screenSize', scr_size; 'resting', rs_time; 'frequency', frequency;'online', false});
%% 패러다임2 (CTA paradigm) 프로그램
test_num=1; 
fNum = 2;
copy_task = test_task;
total_run = length(copy_task)*n_seq*12;
paradigm2({'port', port;'text',copy_task; 'nSequence',n_seq;...
    'screenNum',n_scr;'sti_Times',stimulus_time; 'sti_Interval',interval_time;...
    'screenSize', scr_size; 'resting', rs_time; 'frequency', frequency; 'total_run', total_run; 'online', true});
%% 분석1 (RASP paradigm analysis) 프로그램
clfs = make_classifiers(fullfile(datDir, files{1}), chanSel); 
res = analysis1({'segTime',TimeIval;'baseTime',baselineTime;...
    'selTime',selectedTime;'nFeature',numFeature;'channel',chanSel;...
    'clf_param',clfs; 'selectedFreq', selectedFreq} );
%% 분석2 (CTA paradigm analysis) 프로그램
copy_task = test_task;
total_run = length(copy_task)*n_seq*12;
clfs = make_classifiers(fullfile(datDir, files{1}), chanSel);
res = analysis2({'segTime',TimeIval;'baseTime',baselineTime;...
    'selTime',selectedTime;'nFeature',numFeature;'channel',chanSel;...
    'clf_param',clfs; 'selectedFreq', selectedFreq; 'total_run', total_run;} );
