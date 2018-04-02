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
    % Interpolation of KSS scores
    interKSS = zeros(length(cnt.x), 1) + 1;
    % Decision of KSS score of start and end point based on posterior and
    % prior KSS scores within range 1 ~ 9 (-1 and +1)
    
    interKSS(1) = mrk.kss.toe(1) - 1; interKSS(end) = mrk.kss.toe(end) + 1;
    interKSS(find(interKSS < 1)) = 1; interKSS(find(interKSS > 9)) = 9;
    
    for i = 1 : length(mrk.kss.toe)
        if i == length(mrk.kss.toe)
            interKSS(1 : mrk.kss.pos(1)) = linspace(interKSS(1), mrk.kss.toe(1), length(1 : length(mrk.kss.pos(1))));
            interKSS(mrk.kss.pos(i) : end) = linspace(mrk.kss.toe(i), interKSS(end), length(mrk.kss.pos(i) : length(interKSS)));
        else
            interKSS(mrk.kss.pos(i) : mrk.kss.pos(i + 1)) = linspace(mrk.kss.toe(i), mrk.kss.toe(i + 1), length(mrk.kss.pos(i) : mrk.kss.pos(i + 1)));
        end
    end
    
    %% Synchronization of Response Time for Deviation
    % Calculation of response time between start Devi and end Devi
    clear rt rtPos;
    for i = 1 : length(mrk.pos) / 2
        rt(i) = mrk.pos(i * 2) - mrk.pos(i * 2 - 1);
        rtPos(i) = mrk.pos(i * 2);
    end
    % Interpolation of response times
    interRT = zeros(length(cnt.x), 1);
    % Decision of deviation duration based on averaged values in specific
    % range (First half and last half samples for start and end point)
    interRT(1) = mean(rt(1 : round(length(rt) / 2)));
    interRT(end) = mean(rt(round(length(rt) / 2) : end));
    for i = 1 : length(rt)
        if i == length(rt)
            interRT(1 : rtPos(1)) = linspace(interRT(1), rt(1), length(1 : rtPos(1)));
            interRT(rtPos(i) : end) = linspace(rt(i), interRT(end), length(rtPos(i) : length(interRT)));
        else
            interRT(rtPos(i) : rtPos(i + 1)) = linspace(rt(i), rt(i + 1), length(rtPos(i) : rtPos(i + 1)));
        end
    end
    
    %% Segmentation of bio-signals with 1 seconds length of epoch
    % Epo included in exception condition will be abandoned during process
    epoch = segmentationSleep_R(cnt, mrk, interKSS, interRT);
    kssIdx = find(strcmp(epoch.mClab, 'KSS'));
    kss = squeeze(mean(epoch.misc(:, kssIdx, :), 1));

    %% Regional Channel Selection
    channel_f=epoch.x(:,1:7,:);
    channel_o=epoch.x(:,28:32,:);
    channel_p=epoch.x(:,23:27,:);
    channel_c=epoch.x(:,[8:11,13:15,18:21],:);
    channel_t=epoch.x(:,[12,16:17,22],:);
    
    channel_f1=epoch.x(:,[1,3:4],:);
    channel_f2=epoch.x(:,[2,6:7],:);
    channel_f3=epoch.x(:,5,:);
    channel_o1=epoch.x(:,29,:);
    channel_o2=epoch.x(:,31,:);
    channel_o3=epoch.x(:,30,:);
    channel_p1=epoch.x(:,[18:19,23:24,28],:);
    channel_p2=epoch.x(:,[20:21,26:27,32],:);
    channel_p3=epoch.x(:,25,:);
    channel_c1=epoch.x(:,[8:9,13],:);
    channel_c2=epoch.x(:,[10:11,15],:);
    channel_c3=epoch.x(:,14,:);
    channel_t1=epoch.x(:,[12,17],:);
    channel_t2=epoch.x(:,[16,22],:);
    
    %% Feature Extraction - Power spectrum analysis for EEG signals
    % Two kinds of spectrum features were extracted (Averaged and
    % whole channel values of power spectrum density
    % PSD Feature Extraction
    [cca_delta, avg_delta] = spectrumAnalysisSleep_R(epoch.x, [0.5 4], cnt.fs);
    [cca_theta, avg_theta] = spectrumAnalysisSleep_R(epoch.x, [4 8], cnt.fs);
    [cca_alpha, avg_alpha] = spectrumAnalysisSleep_R(epoch.x, [8 13], cnt.fs);
    [cca_beta, avg_beta] = spectrumAnalysisSleep_R(epoch.x, [13 30], cnt.fs);
    [cca_gamma, avg_gamma] = spectrumAnalysisSleep_R(epoch.x, [30 50], cnt.fs);
    
