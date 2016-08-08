%% Artifact recording
clear all
% p=genpath('C:\toolbox');
% addpath(p)

%% Setting
sound_dir='sound\';
% image_dir='C:\Users\CVPR\Desktop\CVPR\StarLab\Experiment\cnt_pilot\image\';

%% Eyes movement
% Left, right, up, and down
artifact_eyesMove({'soundDirectory',sound_dir;'repeatTimes',10;'blankTime',1.5;'durationTime',2})

%% Eyes blink, clench teeth, lift shoulders, move head
artifact_theRest({'blankTime',3;'durationTime',5})

%% Eyes open/closed
eyesOpenClosed({'soundDirectory',sound_dir; 'repeatTimes',10; 'blankTime',1.5; 'durationTime',14})