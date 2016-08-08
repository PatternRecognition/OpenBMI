clc 
clear all;

OpenBMI('C:\Users\Administrator\Desktop\BCI_Toolbox\OpenBMI')
global BMI;
BMI.EEG_DIR=['C:\Users\Administrator\Desktop\BCI_Toolbox\DemoData'];

%% Artifact recording
% Eyes movement
artifact_eyesMove({'soundDirectory','sound';'repeatTimes',10;'blankTime',1.5;'durationTime',2})
% Eyes blink, clench teeth, lift shoulders, move head (2m :30)
artifact_theRest({'blankTime',3;'durationTime',5})
% Eyes open/closed (6m 16) too long
eyesOpenClosed({'soundDirectory','sound'; 'repeatTimes',10; 'blankTime',1.5; 'durationTime',14})

%% Continous real-movement
Makeparadigm_realmovement({'time_sti',60,'time_isi',10,'num_trial',4,'num_class',3,'time_jitter',0.1})

%% Continous motor-imagery
Makeparadigm_realmovement({'time_sti',60,'time_isi',5,'num_trial',4,'num_class',2,'time_jitter',0.1})

% Uncued motor-imagery - 수정 필요
Makeparadigm_MI_uncue({'time_sti',60,'time_isi',10,'num_trial',4,'num_class',2,'time_jitter',0.1})


%% Discrete MI with no-feedback
Makeparadigm_MI({'time_sti',4,'time_isi',2,'time_rest',3,'num_trial',50,'num_class',3,'time_jitter',0.1,'num_screen',2});

%% Accuracy, and parameters
BMI.EEG_DIR=['C:\Users\Administrator\Desktop\BCI_Toolbox\DemoData'];
file=fullfile(BMI.EEG_DIR, '20160704_smkim_Run1NoFeedback');
band=[8 13];
fs=500;
[LOSS, CSP, LDA]=MI_calibration(file, band,fs);
% select the binary class, foot vs (right or left): 2=right(-) vs foot(+), 3=left(-) vs foot(+) 
FOOTfb=3;

% for only left-right
pseudoOnline(BMI.EEG_DIR, '\mhlee_160701_calib_long','\mhlee_160701_calib_short4', band, fs)

%% Calibration for feedback experiment
% matlab1: psychtoolbox
[weight(1), bias(1)]=MI_setting();
[weight(2), bias(2)]=MI_setting();
% matlab2: executed with second matlab
Feedback_Client(CSP{1,1}, LDA{1,1}, band, {'buffer_size', 5000; 'data_size', 1500; 'feedback_freq', 100/1000;'TCPIP','on'});
Feedback_Client(CSP{FOOTfb,1}, LDA{FOOTfb,1}, band, {'buffer_size', 5000; 'data_size', 1000; 'feedback_freq', 100/1000;'TCPIP', 'on'});

%% Discrete MI with feedback
% matlab1: psychtoolbox
Makeparadigm_MI_feedback(weight, bias, FOOTfb, {'time_sti',4;'time_isi',3;'time_rest',3;'num_trial',30;'num_class',3;'time_jitter',0.1;'screenNumber',2});
% matlab2: executed with second matlab
Feedback_Client(CSP, LDA, band, {'buffer_size', 5000;  'data_size', 1500; 'feedback_freq', 100/1000; 'TCPIP', 'on'} );
% CSP{1,1}, LDA{1,1} not cell, binary class output
% CSP, LDA, cell type, 3 class output 

