clear all; close all; clc;

load('BTB.mat')
global BTB
BTB.MatDir='D:\';

subjectList = {'1'};
saveFile = 'D:\';
file = 'D:\';

for s = 1 : length(subjectList)
    [cnt, mrk, mnt] = eegfile_loadMatlab(strcat(file, '\', subjectList{s}));
    EEG file load
    
    %% Band-pass Filter and Artefact Removal (ICA)
    [b, a] = butter(4, [0.5 50] / 100, 'bandpass');
    y = filter(b, a, cnt.x);
    cnt.x=cnt.x(:,1:34);
    cnt.clab=cnt.clab(:,1:34);
    
    cnt.x=fastica(cnt.x');
    cnt.x=cnt.x';
    
    %     Band-Pass Filter Visualization
    %     f_sz=ceil(length(cnt.x)/2);
    %     f=100*linspace(0,1,f_sz);
    %     f_X=fft(cnt.x);
    %     f_y=fft(y);
    %     subplot(2,1,1)
    %     stem(f,abs(f_X(1:f_sz)));
    %     title('Original signal');
    %     xlabel('frequency');
    %     xlim([0 50]);
    %     ylabel('power');
    %     subplot(2,1,2)
    %     stem(f,abs(f_y(1:f_sz)));
    %     xlim([0 50]);
    %
    %     title('Application of the beta (13-30Hz) bandpass filter')
    %     xlabel('frequency');
    %     ylabel('power');
    
    %    ICA Visualization
    %     fasticag(cnt.x');
    %     f=get(gcf,'child')
    %     for i=1:34
    %         f(i).YLim=[-20 20];
    %     end
    
    %% Synchronization of KSS
    [interKSS] = kss_interpolation(cnt,mrk)
    % Interpolation of the KSS score for marker augmentation