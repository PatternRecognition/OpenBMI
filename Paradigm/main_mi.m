OpenBMI('C:\Users\OpenBMI_Lab\Desktop\OpenBMI')
global BMI;
BMI.EEG_DIR=['C:\Users\OpenBMI_Lab\Desktop\OpenBMI\data'];


%% Artifact recording
artifact_eyesMove({'soundDirectory',sound_dir; 'repeatTimes',10; 'blankTime',1.5; 'durationTime',1})
artifact_theRest({'blankTime',3;'durationTime',5})
eyesOpenClosed({'soundDirectory',in.soundDirectory; 'repeatTimes',10; 'blankTime',1.5; 'durationTime',10})


%% MI without feedback
Makeparadigm_MI({'time_cross',1.5;'time_sti',60;'time_blank',5;'num_trial',1;'num_class',3});


%% Accuracy, and parameters
file=fullfile(BMI.EEG_DIR, '\160928_smkim'); % file name
band=[8 13];
fs=100;
interval=[750 3500];
channel_index=[11:20,22:26,46:52,55:57];
[LOSS, CSP, LDA]=MI_calibration_2(file, band, fs, interval,3,{'nClass',3;'erd',0;'channel',channel_index});

% select the binary class, foot vs (right or left): 2=right(-) vs foot(+), 3=left(-) vs foot(+)
FOOTfb=3;


%% Calibration for feedback experiment
% matlab1: psychtoolbox
[weight(1), bias(1)]=MI_setting(1);
[weight(2), bias(2)]=MI_setting(2);
% matlab2: executed with second matlab
Feedback_Client(CSP{1,1}, LDA{1,1}, band, {'buffer_size',5000; 'data_size',1500; 'feedback_freq',100/1000; 'TCPIP','on'; 'channel',channel_index});
Feedback_Client(CSP{FOOTfb,1}, LDA{FOOTfb,1}, band, {'buffer_size',5000; 'data_size',1000; 'feedback_freq',100/1000; 'TCPIP','on'; 'channel',channel_index});


%% Discrete MI with feedback
% matlab1: psychtoolbox
Makeparadigm_MI_feedback(weight, bias, FOOTfb, {'time_sti',4;'time_blank',3;'time_cross',3;'num_trial',50;'num_class',3;'time_jitter',0.1});
% matlab2: executed with second matlab
Feedback_Client(CSP, LDA, band, {'buffer_size',5000; 'data_size',1500; 'feedback_freq',100/1000; 'TCPIP','on'; 'channel',channel_index});


%% P300 speller
% training
seq1=Makeparadigm_speller({'text','MACHINE_LEARNING'},-1);
% test
seq2=Makeparadigm_speller({'text','OPENBMI_SPELLER'},-1);
% save variables
str=sprintf('%s\\p300_sequence_order',BMI.EEG_DIR);
save(str,'seq1','seq2')


%% SSVEP
Makeparadigm_SSVEP ({'time_sti',5;'num_trial',10;'time_rest',3;'freq',[7.5 10 12 15 20];'boxSize',150;'betweenBox',200});

