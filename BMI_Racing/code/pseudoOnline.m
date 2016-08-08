function []=pseudoOnline(file, train,test, band, ff)

% clear all;%close all;clc
% restoredefaultpath
% 
% p=genpath('C:\Users\BCI_STAR_LAB\Desktop\OpenBMI');
% addpath(p)


%% Calibration
file1=fullfile(file, train);
[LOSS, CSP, LDA]=MI_calibration(file1, band, ff);

%% Feedback
% Load data
marker= {'1','right';'2','left'};
[EEG.data, EEG.marker, EEG.info]=Load_EEG(fullfile(file, test),{'device','brainVision';'marker', marker;'fs', ff});

CNT=opt_eegStruct({EEG.data, EEG.marker, EEG.info}, {'x','t','fs','y_dec','y_logic','y_class','class', 'chan'});

% 실제 마커
Truth=zeros(1,length(CNT.x));
right_idx=round(EEG.marker.t(EEG.marker.y_dec==1));
for k=1:length(right_idx)
    Truth(right_idx(k):right_idx(k)+(ff*4-1))=-1;
end
left_idx=round(EEG.marker.t(EEG.marker.y_dec==2));
for k=1:length(right_idx)
    Truth(left_idx(k):left_idx(k)+(ff*4-1))=1;
end


% pseudo
windowSize=ff*3;
stepSize=ff*0.5;

t=1:stepSize:length(CNT.x);
t(t+windowSize>length(CNT.x))=[];
cf_out=zeros(1,round((length(CNT.x)-windowSize)/stepSize));

%% if cf_out bigger t, fine, less problem 
t=t(1:length(cf_out));
%%

for i=1:(length(CNT.x)-windowSize)/stepSize
    
    Dat=CNT.x(stepSize*i:(stepSize*i+windowSize),:);
    fDat=prep_filter(Dat, {'frequency', band;'fs',ff });

% fDat.x=Dat;
% fDat.fs=ff;
% fDat=prep_resample(fDat,100);

    tm=func_projection(fDat, CSP{1,1});
    ft=func_featureExtraction(tm, {'feature','logvar'});
    [cf_out(i)]=func_predict(ft, LDA{1,1});
    
end

figure()
plot(t,cf_out)
hold on
bar(Truth)



% label=ones(size(EEG.marker.y_dec));
str=sprintf('window size = %d ms',windowSize);
title(str)