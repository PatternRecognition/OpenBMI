clc; clear all; close all;
%% data location
dd='/home/wam/Desktop/eegData/Twist/';
filelist={'20180323_jgyoon2_twist_MI','20180220_bwyu_twist_MI'};

channel_layout=[8 9 10 11 13 14 15 18 19 20 21 43 44 47 48 49 50 52 53 54];
%% hyperparameter
order = 2;
band={[8 15]};
ival=[0 3000];
%% save the eeg data
for i=1:length(filelist)
    loadD=loadEEG([dd filelist{i}], order, band{1}, ival, channel_layout);
    clab=loadD.clab;
    fs=loadD.fs;
    x=loadD.x;
    y=loadD.y;
    t=loadD.t;
    className=loadD.className;
    save(['./Converted/' filelist{i} '/clab'],'clab');
    save(['./Converted/' filelist{i} '/fs'],'fs');
    save(['./Converted/' filelist{i} '/x'],'x');
    save(['./Converted/' filelist{i} '/y'],'y');
    save(['./Converted/' filelist{i} '/t'],'t');
    save(['./Converted/' filelist{i} '/className'],'className');
end
%% get all data from the directory
clear all; close all; clc;
dirData = dir('./Converted/*.mat');
%% python or dnn
data={};
for i =1:size(dirData,1)
    load(['./Converted/' dirData(i).name]);
    data{i}=loadD;
end
