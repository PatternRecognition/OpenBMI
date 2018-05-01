clear all; close all; clc;

load('BTB.mat')
global BTB
BTB.MatDir='D:\ICMI\Data\Fatigue\MATLAB';
subjectList = {'JHHWANG_20180407_fatigue'};
saveFile = 'D:\ICMI\Figure';
file = 'D:\ICMI\Data\Fatigue\MATLAB';
s=1;
[cnt, mrk, mnt] = eegfile_loadMatlab(strcat(file, '\', subjectList{s}));
clear psd_feature_cca psd_feature_cca_f psd_feature_cca_c psd_feature_cca_t psd_feature_cca_o psd_feature_cca_p psd_feature_cca_f1 psd_feature_cca_f2 psd_feature_cca_f3 psd_feature_cca_c1 psd_feature_cca_c2 psd_feature_cca_c3 psd_feature_cca_p1 psd_feature_cca_p2 psd_feature_cca_p3 psd_feature_cca_t1 psd_feature_cca_t2 psd_feature_cca_o1 psd_feature_cca_o2 psd_feature_cca_o3;

%% Band-pass Filter and Arteface Removal (ICA)

[b, a] = butter(4, [0.5 50] / 100, 'bandpass');
y = filter(b, a, cnt.x);
cnt.x=cnt.x(:,1:64);
cnt.clab=cnt.clab(:,1:64);

%% Loading Physiological data

FileName = subjectList{s};
FolderName = 'D:\ICMI\Data\Phsiological Signals\Fatigue\matlab';
File = fullfile(FolderName, FileName);
phy = load(File);
phy.cnt = [phy.EKG, phy.SC, phy.Abd, phy.Thor];

%% Synchronization of KSS and EEG

interKSS = zeros(length(cnt.x), 1) + 1;
interKSS(1) = mrk.toe(1) - 1; interKSS(end) = mrk.toe(end) + 1;
interKSS(find(interKSS < 1)) = 1; interKSS(find(interKSS > 9)) = 9;

for i = 1 : length(mrk.toe)
    if i == length(mrk.toe)
        interKSS(1 : mrk.pos(1)) = linspace(interKSS(1), mrk.toe(1), length(1 : length(mrk.pos(1))));
        interKSS(mrk.pos(i) : end) = linspace(mrk.toe(i), interKSS(end), length(mrk.pos(i) : length(interKSS)));
    else
        interKSS(mrk.pos(i) : mrk.pos(i + 1)) = linspace(mrk.toe(i), mrk.toe(i + 1), length(mrk.pos(i) : mrk.pos(i + 1)));
    end
end



interKSS_BIO = zeros(length(phy.cnt), 1) + 1;
interKSS_BIO(1) = mrk.toe(1) - 1; interKSS_BIO(end) = mrk.toe(end) + 1;
interKSS_BIO(find(interKSS_BIO < 1)) = 1; interKSS_BIO(find(interKSS_BIO > 9)) = 9;

for i = 1 : length(mrk.toe)
    if i == length(mrk.toe)
        interKSS_BIO(1 : mrk.pos(1)) = linspace(interKSS_BIO(1), mrk.toe(1), length(1 : length(mrk.pos(1))));
        interKSS_BIO(mrk.pos(i) : end) = linspace(mrk.toe(i), interKSS_BIO(end), length(mrk.pos(i) : length(interKSS_BIO)));
    else
        interKSS_BIO(mrk.pos(i) : mrk.pos(i + 1)) = linspace(mrk.toe(i), mrk.toe(i + 1), length(mrk.pos(i) : mrk.pos(i + 1)));
    end
end
%% Segmentation of EEG signals with 1 seconds length of epoch

epoch = segmentationFatigue_R(cnt, mrk, interKSS);
    kssIdx = find(strcmp(epoch.mClab, 'KSS'));
    kss = squeeze(mean(epoch.misc(:, kssIdx, :), 1));
%% Segmentation of Biosignal with 30 seconds length + 29 overlab of epoch

epoch_biosignal = segmentationFatigue_RS(phy, mrk, interKSS_BIO);
    kssIdx_biosignal = find(strcmp(epoch_biosignal.mClab, 'KSS_BIO'));
    kss_biosignal = squeeze(mean(epoch.misc(:, kssIdx, :), 1));
    
resp_feature1 = mean(epoch_biosignal.x(:,3,1:1811));
% abd 에포크 별 평균치
resp_feature2 = mean(epoch_biosignal.x(:,4,1:1811));
% thor 에포크 별 평균치
resp_feature3 = resp_feature1 - resp_feature2;

eda_feature1 = mean(epoch_biosignal.x(:,2,1:1811));