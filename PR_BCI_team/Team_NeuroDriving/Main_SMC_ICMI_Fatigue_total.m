clear all; close all; clc;

load('BTB.mat')

subjectList = {'HSRYU_20180406_fatigue','JHHWANG_20180407_fatigue','GSPARK_20180413_fatigue','DNJO_20180414_fatigue','KTKIM_20180420_fatigue','DJLEE_20180421_fatigue'};
file = 'F:\Matlab\Data\Pilot\Fatigue\MATLAB';
savefile = 'F:\Matlab\Plot\Fatigue';
% ,'JHHWANG_20180407_fatigue','GSPARK_20180413_fatigue','DNJO_20180414_fatigue','KTKIM_20180420_fatigue','DJLEE_20180421_fatigue'
for s = 1 : length(subjectList)
    [cnt, mrk, mnt] = eegfile_loadMatlab(strcat(file, '\', subjectList{s}));
    clear psd_feature_cca psd_feature_cca_f psd_feature_cca_c psd_feature_cca_t psd_feature_cca_o psd_feature_cca_p psd_feature_cca_f1 psd_feature_cca_f2 psd_feature_cca_f3 psd_feature_cca_c1 psd_feature_cca_c2 psd_feature_cca_c3 psd_feature_cca_p1 psd_feature_cca_p2 psd_feature_cca_p3 psd_feature_cca_t1 psd_feature_cca_t2 psd_feature_cca_o1 psd_feature_cca_o2 psd_feature_cca_o3;
    
    %% Band-pass Filter and Arteface Removal (ICA)
    [b, a] = butter(4, [0.5 40] / (cnt.fs/2), 'bandpass');
    y = filter(b, a, cnt.x);
    cnt.x=y(:,1:64);
    cnt.clab=cnt.clab(:,1:64);
    
    cnt.x=fastica(cnt.x');
    cnt.x=cnt.x';
    
    %% Synchronization of KSS
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
    
    %% Segmentation of bio-signals with 1 seconds length of epoch
    epoch = segmentationFatigue_1(cnt, mrk, interKSS);
    kssIdx = find(strcmp(epoch.mClab, 'KSS'));
    kss = squeeze(mean(epoch.misc(:, kssIdx, :), 1));
    
    %% Regional Channel Selection
    channel_f=epoch.x(:,[1:5,31:36],:);
    channel_c=epoch.x(:,[6:9,11:13,16:19,39:40,43:46,48:50],:);
    channel_p=epoch.x(:,[21:25,52:55],:);
    channel_t=epoch.x(:,[10,14:15,20,37:38,41:42,47,51],:);
    channel_o=epoch.x(:,[26:30,56:60],:);
    
    channel_f1=epoch.x(:,[1,3,31,33],:);
    channel_f2=epoch.x(:,[2,5,32,36],:);
    channel_f3=epoch.x(:,[4,34:35],:);
    channel_c1=epoch.x(:,[6:7,11,16,17,39,43,44,48],:);
    channel_c2=epoch.x(:,[8:9,13,18,19,40,45,46,50],:);
    channel_c3=epoch.x(:,[12,49],:);
    channel_p1=epoch.x(:,[21:22,52,53],:);
    channel_p2=epoch.x(:,[24:25,54,55],:);
    channel_p3=epoch.x(:,[23],:);
    channel_t1=epoch.x(:,[10,15,37,38,47],:);
    channel_t2=epoch.x(:,[14,20,41,42,51],:);
    channel_o1=epoch.x(:,[26,27,56,57],:);
    channel_o2=epoch.x(:,[29,30,59,60],:);
    channel_o3=epoch.x(:,[28,58],:);
    
    %% Feature Extraction - Power spectrum analysis for EEG signals
    
    % PSD Feature Extraction
    [cca.delta, avg.delta] = Power_spectrum(epoch.x, [0.5 4], cnt.fs);
    [cca.theta, avg.theta] = Power_spectrum(epoch.x, [4 8], cnt.fs);
    [cca.alpha, avg.alpha] = Power_spectrum(epoch.x, [8 13], cnt.fs);
    [cca.beta, avg.beta] = Power_spectrum(epoch.x, [13 30], cnt.fs);
    [cca.gamma, avg.gamma] = Power_spectrum(epoch.x, [30 40], cnt.fs);
    avg.delta = avg.delta'; avg.theta=avg.theta'; avg.alpha=avg.alpha'; avg.beta = avg.beta'; avg.gamma = avg.gamma';
    
    cca.PSD = [cca.delta cca.theta cca.alpha cca.beta cca.gamma];
    avg.PSD = [avg.delta avg.theta avg.alpha avg.beta avg.gamma];

    % RPL Feature Extraction
    cca.RPL.delta = cca.delta ./ (cca.delta + cca.theta + cca.alpha + cca.beta + cca.gamma) ;
    avg.RPL.delta = avg.delta ./ sum(avg.PSD,2) ;
    cca.RPL.theta = cca.theta ./ (cca.delta + cca.theta + cca.alpha + cca.beta + cca.gamma) ;
    avg.RPL.theta = avg.theta ./ sum(avg.PSD,2) ;
    cca.RPL.alpha = cca.alpha ./ (cca.delta + cca.theta + cca.alpha + cca.beta + cca.gamma) ;
    avg.RPL.alpha = avg.alpha ./ sum(avg.PSD,2) ;
    cca.RPL.beta = cca.beta ./ (cca.delta + cca.theta + cca.alpha + cca.beta + cca.gamma) ;
    avg.RPL.beta = avg.beta ./ sum(avg.PSD,2) ;
    cca.RPL.gamma = cca.gamma ./ (cca.delta + cca.theta + cca.alpha + cca.beta + cca.gamma) ;
    avg.RPL.gamma = avg.gamma ./ sum(avg.PSD,2) ;
    
    cca.RPL.PSD = [cca.RPL.delta, cca.RPL.theta, cca.RPL.alpha, cca.RPL.beta, cca.RPL.gamma];
    avg.RPL.PSD = [avg.RPL.delta, avg.RPL.theta, avg.RPL.alpha, avg.RPL.beta, avg.RPL.gamma];
    
    % Z-SCORE Feature Extraction
    cca.z.PSD = [zscore(cca.delta) zscore(cca.theta) zscore(cca.alpha) zscore(cca.beta) zscore(cca.gamma)];
    avg.z.PSD = [zscore(avg.delta) zscore(avg.theta) zscore(avg.alpha) zscore(avg.beta) zscore(avg.gamma)];
    
    
    % Regional Feature Extraction
    [cca.delta_f,avg.delta_f] = Power_spectrum(channel_f, [0.5 4], cnt.fs);
    [cca.theta_f,avg.theta_f] = Power_spectrum(channel_f, [4 8], cnt.fs);
    [cca.alpha_f,avg.alpha_f] = Power_spectrum(channel_f, [8 13], cnt.fs);
    [cca.beta_f,avg.beta_f] = Power_spectrum(channel_f, [13 30], cnt.fs);
    [cca.gamma_f,avg.gamma_f] = Power_spectrum(channel_f, [30 40], cnt.fs);
    avg.delta_f = avg.delta_f'; avg.theta_f=avg.theta_f'; avg.alpha_f=avg.alpha_f'; avg.beta_f = avg.beta_f'; avg.gamma_f = avg.gamma_f';
    
    cca.Regional_f = [cca.delta_f cca.theta_f cca.alpha_f cca.beta_f cca.gamma_f];
    avg.Regional_f = [avg.delta_f avg.theta_f avg.alpha_f avg.beta_f avg.gamma_f];
    
    cca.RPL.delta_f = cca.delta_f ./ (cca.delta_f + cca.theta_f + cca.alpha_f + cca.beta_f + cca.gamma_f) ;
    avg.RPL.delta_f = avg.delta_f ./ sum(avg.Regional_f,2) ;
    cca.RPL.theta_f = cca.theta_f ./ (cca.delta_f + cca.theta_f + cca.alpha_f + cca.beta_f + cca.gamma_f) ;
    avg.RPL.theta_f = avg.theta_f ./ sum(avg.Regional_f,2) ;
    cca.RPL.alpha_f = cca.alpha_f ./ (cca.delta_f + cca.theta_f + cca.alpha_f + cca.beta_f + cca.gamma_f) ;
    avg.RPL.alpha_f = avg.alpha_f ./ sum(avg.Regional_f,2) ;
    cca.RPL.beta_f = cca.beta_f ./ (cca.delta_f + cca.theta_f + cca.alpha_f + cca.beta_f + cca.gamma_f) ;
    avg.RPL.beta_f = avg.beta_f ./ sum(avg.Regional_f,2) ;
    cca.RPL.gamma_f = cca.gamma_f ./ (cca.delta_f + cca.theta_f + cca.alpha_f + cca.beta_f + cca.gamma_f) ;
    avg.RPL.gamma_f = avg.gamma_f ./ sum(avg.Regional_f,2) ;    
    
    cca.RPL.Regional_f = [cca.RPL.delta_f cca.RPL.theta_f cca.RPL.alpha_f cca.RPL.beta_f cca.RPL.gamma_f]; 
    avg.RPL.Regional_f = [avg.RPL.delta_f avg.RPL.theta_f avg.RPL.alpha_f avg.RPL.beta_f avg.RPL.gamma_f];
    
    cca.z.Regional_f = [zscore(cca.delta_f) zscore(cca.theta_f) zscore(cca.alpha_f) zscore(cca.beta_f) zscore(cca.gamma_f)];
    avg.z.Regional_f = [zscore(avg.delta_f) zscore(avg.theta_f) zscore(avg.alpha_f) zscore(avg.beta_f) zscore(avg.gamma_f)];   
    
    
    
    [cca.delta_c,avg.delta_c] = Power_spectrum(channel_c, [0.5 4], cnt.fs);
    [cca.theta_c,avg.theta_c] = Power_spectrum(channel_c, [4 8], cnt.fs);
    [cca.alpha_c,avg.alpha_c] = Power_spectrum(channel_c, [8 13], cnt.fs);
    [cca.beta_c,avg.beta_c] = Power_spectrum(channel_c, [13 30], cnt.fs);
    [cca.gamma_c,avg.gamma_c] = Power_spectrum(channel_c, [30 40], cnt.fs);
    avg.delta_c = avg.delta_c'; avg.theta_c=avg.theta_c'; avg.alpha_c=avg.alpha_c'; avg.beta_c = avg.beta_c'; avg.gamma_c = avg.gamma_c';
    
    cca.Regional_c = [cca.delta_c cca.theta_c cca.alpha_c cca.beta_c cca.gamma_c];
    avg.Regional_c = [avg.delta_c avg.theta_c avg.alpha_c avg.beta_c avg.gamma_c];
    
    cca.RPL.delta_c = cca.delta_c ./ (cca.delta_c + cca.theta_c + cca.alpha_c + cca.beta_c + cca.gamma_c) ;
    avg.RPL.delta_c = avg.delta_c ./ sum(avg.Regional_c,2) ;
    cca.RPL.theta_c = cca.theta_c ./ (cca.delta_c + cca.theta_c + cca.alpha_c + cca.beta_c + cca.gamma_c) ;
    avg.RPL.theta_c = avg.theta_c ./ sum(avg.Regional_c,2) ;
    cca.RPL.alpha_c = cca.alpha_c ./ (cca.delta_c + cca.theta_c + cca.alpha_c + cca.beta_c + cca.gamma_c) ;
    avg.RPL.alpha_c = avg.alpha_c ./ sum(avg.Regional_c,2) ;
    cca.RPL.beta_c = cca.beta_c ./ (cca.delta_c + cca.theta_c + cca.alpha_c + cca.beta_c + cca.gamma_c) ;
    avg.RPL.beta_c = avg.beta_c ./ sum(avg.Regional_c,2) ;
    cca.RPL.gamma_c = cca.gamma_c ./ (cca.delta_c + cca.theta_c + cca.alpha_c + cca.beta_c + cca.gamma_c) ;
    avg.RPL.gamma_c = avg.gamma_c ./ sum(avg.Regional_c,2) ;     
    
    cca.RPL.Regional_c = [cca.RPL.delta_c cca.RPL.theta_c cca.RPL.alpha_c cca.RPL.beta_c cca.RPL.gamma_c]; 
    avg.RPL.Regional_c = [avg.RPL.delta_c avg.RPL.theta_c avg.RPL.alpha_c avg.RPL.beta_c avg.RPL.gamma_c];
    
    cca.z.Regional_c = [zscore(cca.delta_c) zscore(cca.theta_c) zscore(cca.alpha_c) zscore(cca.beta_c) zscore(cca.gamma_c)];
    avg.z.Regional_c = [zscore(avg.delta_c) zscore(avg.theta_c) zscore(avg.alpha_c) zscore(avg.beta_c) zscore(avg.gamma_c)];   
    
    
    [cca.delta_p,avg.delta_p] = Power_spectrum(channel_p, [0.5 4], cnt.fs);
    [cca.theta_p,avg.theta_p] = Power_spectrum(channel_p, [4 8], cnt.fs);
    [cca.alpha_p,avg.alpha_p] = Power_spectrum(channel_p, [8 13], cnt.fs);
    [cca.beta_p,avg.beta_p] = Power_spectrum(channel_p, [13 30], cnt.fs);
    [cca.gamma_p,avg.gamma_p] = Power_spectrum(channel_p, [30 40], cnt.fs);
    avg.delta_p = avg.delta_p'; avg.theta_p=avg.theta_p'; avg.alpha_p=avg.alpha_p'; avg.beta_p = avg.beta_p'; avg.gamma_p = avg.gamma_p';
    
    cca.Regional_p = [cca.delta_p cca.theta_p cca.alpha_p cca.beta_p cca.gamma_p];
    avg.Regional_p = [avg.delta_p avg.theta_p avg.alpha_p avg.beta_p avg.gamma_p];

    cca.RPL.delta_p = cca.delta_p ./ (cca.delta_p + cca.theta_p + cca.alpha_p + cca.beta_p + cca.gamma_p) ;
    avg.RPL.delta_p = avg.delta_p ./ sum(avg.Regional_p,2) ;
    cca.RPL.theta_p = cca.theta_p ./ (cca.delta_p + cca.theta_p + cca.alpha_p + cca.beta_p + cca.gamma_p) ;
    avg.RPL.theta_p = avg.theta_p ./ sum(avg.Regional_p,2) ;
    cca.RPL.alpha_p = cca.alpha_p ./ (cca.delta_p + cca.theta_p + cca.alpha_p + cca.beta_p + cca.gamma_p) ;
    avg.RPL.alpha_p = avg.alpha_p ./ sum(avg.Regional_p,2) ;
    cca.RPL.beta_p = cca.beta_p ./ (cca.delta_p + cca.theta_p + cca.alpha_p + cca.beta_p + cca.gamma_p) ;
    avg.RPL.beta_p = avg.beta_p ./ sum(avg.Regional_p,2) ;
    cca.RPL.gamma_p = cca.gamma_p ./ (cca.delta_p + cca.theta_p + cca.alpha_p + cca.beta_p + cca.gamma_p) ;
    avg.RPL.gamma_p = avg.gamma_p ./ sum(avg.Regional_p,2) ;     
    
    cca.RPL.Regional_p = [cca.RPL.delta_p cca.RPL.theta_p cca.RPL.alpha_p cca.RPL.beta_p cca.RPL.gamma_p]; 
    avg.RPL.Regional_p = [avg.RPL.delta_p avg.RPL.theta_p avg.RPL.alpha_p avg.RPL.beta_p avg.RPL.gamma_p];
    
    cca.z.Regional_p = [zscore(cca.delta_p) zscore(cca.theta_p) zscore(cca.alpha_p) zscore(cca.beta_p) zscore(cca.gamma_p)];
    avg.z.Regional_p = [zscore(avg.delta_p) zscore(avg.theta_p) zscore(avg.alpha_p) zscore(avg.beta_p) zscore(avg.gamma_p)];   
    
    
    [cca.delta_t,avg.delta_t] = Power_spectrum(channel_t, [0.5 4], cnt.fs);
    [cca.theta_t,avg.theta_t] = Power_spectrum(channel_t, [4 8], cnt.fs);
    [cca.alpha_t,avg.alpha_t] = Power_spectrum(channel_t, [8 13], cnt.fs);
    [cca.beta_t,avg.beta_t] = Power_spectrum(channel_t, [13 30], cnt.fs);
    [cca.gamma_t,avg.gamma_t] = Power_spectrum(channel_t, [30 40], cnt.fs);
    avg.delta_t = avg.delta_t'; avg.theta_t=avg.theta_t'; avg.alpha_t=avg.alpha_t'; avg.beta_t = avg.beta_t'; avg.gamma_t = avg.gamma_t';
    
    cca.Regional_t = [cca.delta_t cca.theta_t cca.alpha_t cca.beta_t cca.gamma_t];
    avg.Regional_t = [avg.delta_t avg.theta_t avg.alpha_t avg.beta_t avg.gamma_t];

    cca.RPL.delta_t = cca.delta_t ./ (cca.delta_t + cca.theta_t + cca.alpha_t + cca.beta_t + cca.gamma_t) ;
    avg.RPL.delta_t = avg.delta_t ./ sum(avg.Regional_t,2) ;
    cca.RPL.theta_t = cca.theta_t ./ (cca.delta_t + cca.theta_t + cca.alpha_t + cca.beta_t + cca.gamma_t) ;
    avg.RPL.theta_t = avg.theta_t ./ sum(avg.Regional_t,2) ;
    cca.RPL.alpha_t = cca.alpha_t ./ (cca.delta_t + cca.theta_t + cca.alpha_t + cca.beta_t + cca.gamma_t) ;
    avg.RPL.alpha_t = avg.alpha_t ./ sum(avg.Regional_t,2) ;
    cca.RPL.beta_t = cca.beta_t ./ (cca.delta_t + cca.theta_t + cca.alpha_t + cca.beta_t + cca.gamma_t) ;
    avg.RPL.beta_t = avg.beta_t ./ sum(avg.Regional_t,2) ;
    cca.RPL.gamma_t = cca.gamma_t ./ (cca.delta_t + cca.theta_t + cca.alpha_t + cca.beta_t + cca.gamma_t) ;
    avg.RPL.gamma_t = avg.gamma_t ./ sum(avg.Regional_t,2) ;     
    
    cca.RPL.Regional_t = [cca.RPL.delta_t cca.RPL.theta_t cca.RPL.alpha_t cca.RPL.beta_t cca.RPL.gamma_t]; 
    avg.RPL.Regional_t = [avg.RPL.delta_t avg.RPL.theta_t avg.RPL.alpha_t avg.RPL.beta_t avg.RPL.gamma_t];
    
    cca.z.Regional_t = [zscore(cca.delta_t) zscore(cca.theta_t) zscore(cca.alpha_t) zscore(cca.beta_t) zscore(cca.gamma_t)];
    avg.z.Regional_t = [zscore(avg.delta_t) zscore(avg.theta_t) zscore(avg.alpha_t) zscore(avg.beta_t) zscore(avg.gamma_t)];   
    
    
    [cca.delta_o,avg.delta_o] = Power_spectrum(channel_o, [0.5 4], cnt.fs);
    [cca.theta_o,avg.theta_o] = Power_spectrum(channel_o, [4 8], cnt.fs);
    [cca.alpha_o,avg.alpha_o] = Power_spectrum(channel_o, [8 13], cnt.fs);
    [cca.beta_o,avg.beta_o] = Power_spectrum(channel_o, [13 30], cnt.fs);
    [cca.gamma_o,avg.gamma_o] = Power_spectrum(channel_o, [30 40], cnt.fs);
    avg.delta_o = avg.delta_o'; avg.theta_o=avg.theta_o'; avg.alpha_o=avg.alpha_o'; avg.beta_o = avg.beta_o'; avg.gamma_o = avg.gamma_o';
    
    cca.Regional_o = [cca.delta_o cca.theta_o cca.alpha_o cca.beta_o cca.gamma_o];
    avg.Regional_o = [avg.delta_o avg.theta_o avg.alpha_o avg.beta_o avg.gamma_o];
    
    cca.RPL.delta_o = cca.delta_o ./ (cca.delta_o + cca.theta_o + cca.alpha_o + cca.beta_o + cca.gamma_o) ;
    avg.RPL.delta_o = avg.delta_o ./ sum(avg.Regional_o,2) ;
    cca.RPL.theta_o = cca.theta_o ./ (cca.delta_o + cca.theta_o + cca.alpha_o + cca.beta_o + cca.gamma_o) ;
    avg.RPL.theta_o = avg.theta_o ./ sum(avg.Regional_o,2) ;
    cca.RPL.alpha_o = cca.alpha_o ./ (cca.delta_o + cca.theta_o + cca.alpha_o + cca.beta_o + cca.gamma_o) ;
    avg.RPL.alpha_o = avg.alpha_o ./ sum(avg.Regional_o,2) ;
    cca.RPL.beta_o = cca.beta_o ./ (cca.delta_o + cca.theta_o + cca.alpha_o + cca.beta_o + cca.gamma_o) ;
    avg.RPL.beta_o = avg.beta_o ./ sum(avg.Regional_o,2) ;
    cca.RPL.gamma_o = cca.gamma_o ./ (cca.delta_o + cca.theta_o + cca.alpha_o + cca.beta_o + cca.gamma_o) ;
    avg.RPL.gamma_o = avg.gamma_o ./ sum(avg.Regional_o,2) ;     
    
    cca.RPL.Regional_o = [cca.RPL.delta_o cca.RPL.theta_o cca.RPL.alpha_o cca.RPL.beta_o cca.RPL.gamma_o]; 
    avg.RPL.Regional_o = [avg.RPL.delta_o avg.RPL.theta_o avg.RPL.alpha_o avg.RPL.beta_o avg.RPL.gamma_o];
    
    cca.z.Regional_o = [zscore(cca.delta_o) zscore(cca.theta_o) zscore(cca.alpha_o) zscore(cca.beta_o) zscore(cca.gamma_o)];
    avg.z.Regional_o = [zscore(avg.delta_o) zscore(avg.theta_o) zscore(avg.alpha_o) zscore(avg.beta_o) zscore(avg.gamma_o)];   
    
    
    
    
    % Regional Vertical Feature Extraction
    [cca.delta_f1,avg.delta_f1] = Power_spectrum(channel_f1, [0.5 4], cnt.fs);
    [cca.theta_f1,avg.theta_f1] = Power_spectrum(channel_f1, [4 8], cnt.fs);
    [cca.alpha_f1,avg.alpha_f1] = Power_spectrum(channel_f1, [8 13], cnt.fs);
    [cca.beta_f1,avg.beta_f1] = Power_spectrum(channel_f1, [13 30], cnt.fs);
    [cca.gamma_f1,avg.gamma_f1] = Power_spectrum(channel_f1, [30 40], cnt.fs);
    avg.delta_f1 = avg.delta_f1'; avg.theta_f1=avg.theta_f1'; avg.alpha_f1=avg.alpha_f1'; avg.beta_f1 = avg.beta_f1'; avg.gamma_f1 = avg.gamma_f1';
    
    cca.Regional_f1 = [cca.delta_f1 cca.theta_f1 cca.alpha_f1 cca.beta_f1 cca.gamma_f1];
    avg.Regional_f1 = [avg.delta_f1 avg.theta_f1 avg.alpha_f1 avg.beta_f1 avg.gamma_f1];
    
    cca.RPL.delta_f1 = cca.delta_f1 ./ (cca.delta_f1 + cca.theta_f1 + cca.alpha_f1 + cca.beta_f1 + cca.gamma_f1) ;
    avg.RPL.delta_f1 = avg.delta_f1 ./ sum(avg.Regional_f1,2) ;
    cca.RPL.theta_f1 = cca.theta_f1 ./ (cca.delta_f1 + cca.theta_f1 + cca.alpha_f1 + cca.beta_f1 + cca.gamma_f1) ;
    avg.RPL.theta_f1 = avg.theta_f1 ./ sum(avg.Regional_f1,2) ;
    cca.RPL.alpha_f1 = cca.alpha_f1 ./ (cca.delta_f1 + cca.theta_f1 + cca.alpha_f1 + cca.beta_f1 + cca.gamma_f1) ;
    avg.RPL.alpha_f1 = avg.alpha_f1 ./ sum(avg.Regional_f1,2) ;
    cca.RPL.beta_f1 = cca.beta_f1 ./ (cca.delta_f1 + cca.theta_f1 + cca.alpha_f1 + cca.beta_f1 + cca.gamma_f1) ;
    avg.RPL.beta_f1 = avg.beta_f1 ./ sum(avg.Regional_f1,2) ;
    cca.RPL.gamma_f1 = cca.gamma_f1 ./ (cca.delta_f1 + cca.theta_f1 + cca.alpha_f1 + cca.beta_f1 + cca.gamma_f1) ;
    avg.RPL.gamma_f1 = avg.gamma_f1 ./ sum(avg.Regional_f1,2) ;    
    
    cca.RPL.Regional_f1 = [cca.RPL.delta_f1 cca.RPL.theta_f1 cca.RPL.alpha_f1 cca.RPL.beta_f1 cca.RPL.gamma_f1]; 
    avg.RPL.Regional_f1 = [avg.RPL.delta_f1 avg.RPL.theta_f1 avg.RPL.alpha_f1 avg.RPL.beta_f1 avg.RPL.gamma_f1];
    
    cca.z.Regional_f1 = [zscore(cca.delta_f1) zscore(cca.theta_f1) zscore(cca.alpha_f1) zscore(cca.beta_f1) zscore(cca.gamma_f1)];
    avg.z.Regional_f1 = [zscore(avg.delta_f1) zscore(avg.theta_f1) zscore(avg.alpha_f1) zscore(avg.beta_f1) zscore(avg.gamma_f1)];   
    
    
    [cca.delta_f2,avg.delta_f2] = Power_spectrum(channel_f2, [0.5 4], cnt.fs);
    [cca.theta_f2,avg.theta_f2] = Power_spectrum(channel_f2, [4 8], cnt.fs);
    [cca.alpha_f2,avg.alpha_f2] = Power_spectrum(channel_f2, [8 13], cnt.fs);
    [cca.beta_f2,avg.beta_f2] = Power_spectrum(channel_f2, [13 30], cnt.fs);
    [cca.gamma_f2,avg.gamma_f2] = Power_spectrum(channel_f2, [30 40], cnt.fs);
    avg.delta_f2 = avg.delta_f2'; avg.theta_f2=avg.theta_f2'; avg.alpha_f2=avg.alpha_f2'; avg.beta_f2 = avg.beta_f2'; avg.gamma_f2 = avg.gamma_f2';
    
    cca.Regional_f2 = [cca.delta_f2 cca.theta_f2 cca.alpha_f2 cca.beta_f2 cca.gamma_f2];
    avg.Regional_f2 = [avg.delta_f2 avg.theta_f2 avg.alpha_f2 avg.beta_f2 avg.gamma_f2];
    
    cca.RPL.delta_f2 = cca.delta_f2 ./ (cca.delta_f2 + cca.theta_f2 + cca.alpha_f2 + cca.beta_f2 + cca.gamma_f2) ;
    avg.RPL.delta_f2 = avg.delta_f2 ./ sum(avg.Regional_f2,2) ;
    cca.RPL.theta_f2 = cca.theta_f2 ./ (cca.delta_f2 + cca.theta_f2 + cca.alpha_f2 + cca.beta_f2 + cca.gamma_f2) ;
    avg.RPL.theta_f2 = avg.theta_f2 ./ sum(avg.Regional_f2,2) ;
    cca.RPL.alpha_f2 = cca.alpha_f2 ./ (cca.delta_f2 + cca.theta_f2 + cca.alpha_f2 + cca.beta_f2 + cca.gamma_f2) ;
    avg.RPL.alpha_f2 = avg.alpha_f2 ./ sum(avg.Regional_f2,2) ;
    cca.RPL.beta_f2 = cca.beta_f2 ./ (cca.delta_f2 + cca.theta_f2 + cca.alpha_f2 + cca.beta_f2 + cca.gamma_f2) ;
    avg.RPL.beta_f2 = avg.beta_f2 ./ sum(avg.Regional_f2,2) ;
    cca.RPL.gamma_f2 = cca.gamma_f2 ./ (cca.delta_f2 + cca.theta_f2 + cca.alpha_f2 + cca.beta_f2 + cca.gamma_f2) ;
    avg.RPL.gamma_f2 = avg.gamma_f2 ./ sum(avg.Regional_f2,2) ;    
    
    cca.RPL.Regional_f2 = [cca.RPL.delta_f2 cca.RPL.theta_f2 cca.RPL.alpha_f2 cca.RPL.beta_f2 cca.RPL.gamma_f2]; 
    avg.RPL.Regional_f2 = [avg.RPL.delta_f2 avg.RPL.theta_f2 avg.RPL.alpha_f2 avg.RPL.beta_f2 avg.RPL.gamma_f2];
    
    cca.z.Regional_f2 = [zscore(cca.delta_f2) zscore(cca.theta_f2) zscore(cca.alpha_f2) zscore(cca.beta_f2) zscore(cca.gamma_f2)];
    avg.z.Regional_f2 = [zscore(avg.delta_f2) zscore(avg.theta_f2) zscore(avg.alpha_f2) zscore(avg.beta_f2) zscore(avg.gamma_f2)];       
    
    
    [cca.delta_f3,avg.delta_f3] = Power_spectrum(channel_f3, [0.5 4], cnt.fs);
    [cca.theta_f3,avg.theta_f3] = Power_spectrum(channel_f3, [4 8], cnt.fs);
    [cca.alpha_f3,avg.alpha_f3] = Power_spectrum(channel_f3, [8 13], cnt.fs);
    [cca.beta_f3,avg.beta_f3] = Power_spectrum(channel_f3, [13 30], cnt.fs);
    [cca.gamma_f3,avg.gamma_f3] = Power_spectrum(channel_f3, [30 40], cnt.fs);
    avg.delta_f3 = avg.delta_f3'; avg.theta_f3=avg.theta_f3'; avg.alpha_f3=avg.alpha_f3'; avg.beta_f3 = avg.beta_f3'; avg.gamma_f3 = avg.gamma_f3';
    
    cca.Regional_f3 = [cca.delta_f3 cca.theta_f3 cca.alpha_f3 cca.beta_f3 cca.gamma_f3];
    avg.Regional_f3 = [avg.delta_f3 avg.theta_f3 avg.alpha_f3 avg.beta_f3 avg.gamma_f3];
    
    cca.RPL.delta_f3 = cca.delta_f3 ./ (cca.delta_f3 + cca.theta_f3 + cca.alpha_f3 + cca.beta_f3 + cca.gamma_f3) ;
    avg.RPL.delta_f3 = avg.delta_f3 ./ sum(avg.Regional_f3,2) ;
    cca.RPL.theta_f3 = cca.theta_f3 ./ (cca.delta_f3 + cca.theta_f3 + cca.alpha_f3 + cca.beta_f3 + cca.gamma_f3) ;
    avg.RPL.theta_f3 = avg.theta_f3 ./ sum(avg.Regional_f3,2) ;
    cca.RPL.alpha_f3 = cca.alpha_f3 ./ (cca.delta_f3 + cca.theta_f3 + cca.alpha_f3 + cca.beta_f3 + cca.gamma_f3) ;
    avg.RPL.alpha_f3 = avg.alpha_f3 ./ sum(avg.Regional_f3,2) ;
    cca.RPL.beta_f3 = cca.beta_f3 ./ (cca.delta_f3 + cca.theta_f3 + cca.alpha_f3 + cca.beta_f3 + cca.gamma_f3) ;
    avg.RPL.beta_f3 = avg.beta_f3 ./ sum(avg.Regional_f3,2) ;
    cca.RPL.gamma_f3 = cca.gamma_f3 ./ (cca.delta_f3 + cca.theta_f3 + cca.alpha_f3 + cca.beta_f3 + cca.gamma_f3) ;
    avg.RPL.gamma_f3 = avg.gamma_f3 ./ sum(avg.Regional_f3,2) ;    
    
    cca.RPL.Regional_f3 = [cca.RPL.delta_f3 cca.RPL.theta_f3 cca.RPL.alpha_f3 cca.RPL.beta_f3 cca.RPL.gamma_f3]; 
    avg.RPL.Regional_f3 = [avg.RPL.delta_f3 avg.RPL.theta_f3 avg.RPL.alpha_f3 avg.RPL.beta_f3 avg.RPL.gamma_f3];
    
    cca.z.Regional_f3 = [zscore(cca.delta_f3) zscore(cca.theta_f3) zscore(cca.alpha_f3) zscore(cca.beta_f3) zscore(cca.gamma_f3)];
    avg.z.Regional_f3 = [zscore(avg.delta_f3) zscore(avg.theta_f3) zscore(avg.alpha_f3) zscore(avg.beta_f3) zscore(avg.gamma_f3)];           
    
    
    [cca.delta_c1,avg.delta_c1] = Power_spectrum(channel_c1, [0.5 4], cnt.fs);
    [cca.theta_c1,avg.theta_c1] = Power_spectrum(channel_c1, [4 8], cnt.fs);
    [cca.alpha_c1,avg.alpha_c1] = Power_spectrum(channel_c1, [8 13], cnt.fs);
    [cca.beta_c1,avg.beta_c1] = Power_spectrum(channel_c1, [13 30], cnt.fs);
    [cca.gamma_c1,avg.gamma_c1] = Power_spectrum(channel_c1, [30 40], cnt.fs);
    avg.delta_c1 = avg.delta_c1'; avg.theta_c1=avg.theta_c1'; avg.alpha_c1=avg.alpha_c1'; avg.beta_c1 = avg.beta_c1'; avg.gamma_c1 = avg.gamma_c1';
    
    cca.Regional_c1 = [cca.delta_c1 cca.theta_c1 cca.alpha_c1 cca.beta_c1 cca.gamma_c1];
    avg.Regional_c1 = [avg.delta_c1 avg.theta_c1 avg.alpha_c1 avg.beta_c1 avg.gamma_c1];
    
    cca.RPL.delta_c1 = cca.delta_c1 ./ (cca.delta_c1 + cca.theta_c1 + cca.alpha_c1 + cca.beta_c1 + cca.gamma_c1) ;
    avg.RPL.delta_c1 = avg.delta_c1 ./ sum(avg.Regional_c1,2) ;
    cca.RPL.theta_c1 = cca.theta_c1 ./ (cca.delta_c1 + cca.theta_c1 + cca.alpha_c1 + cca.beta_c1 + cca.gamma_c1) ;
    avg.RPL.theta_c1 = avg.theta_c1 ./ sum(avg.Regional_c1,2) ;
    cca.RPL.alpha_c1 = cca.alpha_c1 ./ (cca.delta_c1 + cca.theta_c1 + cca.alpha_c1 + cca.beta_c1 + cca.gamma_c1) ;
    avg.RPL.alpha_c1 = avg.alpha_c1 ./ sum(avg.Regional_c1,2) ;
    cca.RPL.beta_c1 = cca.beta_c1 ./ (cca.delta_c1 + cca.theta_c1 + cca.alpha_c1 + cca.beta_c1 + cca.gamma_c1) ;
    avg.RPL.beta_c1 = avg.beta_c1 ./ sum(avg.Regional_c1,2) ;
    cca.RPL.gamma_c1 = cca.gamma_c1 ./ (cca.delta_c1 + cca.theta_c1 + cca.alpha_c1 + cca.beta_c1 + cca.gamma_c1) ;
    avg.RPL.gamma_c1 = avg.gamma_c1 ./ sum(avg.Regional_c1,2) ;     
    
    cca.RPL.Regional_c1 = [cca.RPL.delta_c1 cca.RPL.theta_c1 cca.RPL.alpha_c1 cca.RPL.beta_c1 cca.RPL.gamma_c1]; 
    avg.RPL.Regional_c1 = [avg.RPL.delta_c1 avg.RPL.theta_c1 avg.RPL.alpha_c1 avg.RPL.beta_c1 avg.RPL.gamma_c1];
    
    cca.z.Regional_c1 = [zscore(cca.delta_c1) zscore(cca.theta_c1) zscore(cca.alpha_c1) zscore(cca.beta_c1) zscore(cca.gamma_c1)];
    avg.z.Regional_c1 = [zscore(avg.delta_c1) zscore(avg.theta_c1) zscore(avg.alpha_c1) zscore(avg.beta_c1) zscore(avg.gamma_c1)];               
    
    [cca.delta_c2,avg.delta_c2] = Power_spectrum(channel_c2, [0.5 4], cnt.fs);
    [cca.theta_c2,avg.theta_c2] = Power_spectrum(channel_c2, [4 8], cnt.fs);
    [cca.alpha_c2,avg.alpha_c2] = Power_spectrum(channel_c2, [8 13], cnt.fs);
    [cca.beta_c2,avg.beta_c2] = Power_spectrum(channel_c2, [13 30], cnt.fs);
    [cca.gamma_c2,avg.gamma_c2] = Power_spectrum(channel_c2, [30 40], cnt.fs);
    avg.delta_c2 = avg.delta_c2'; avg.theta_c2=avg.theta_c2'; avg.alpha_c2=avg.alpha_c2'; avg.beta_c2 = avg.beta_c2'; avg.gamma_c2 = avg.gamma_c2';
    
    cca.Regional_c2 = [cca.delta_c2 cca.theta_c2 cca.alpha_c2 cca.beta_c2 cca.gamma_c2];
    avg.Regional_c2 = [avg.delta_c2 avg.theta_c2 avg.alpha_c2 avg.beta_c2 avg.gamma_c2];
    
    cca.RPL.delta_c2 = cca.delta_c2 ./ (cca.delta_c2 + cca.theta_c2 + cca.alpha_c2 + cca.beta_c2 + cca.gamma_c2) ;
    avg.RPL.delta_c2 = avg.delta_c2 ./ sum(avg.Regional_c2,2) ;
    cca.RPL.theta_c2 = cca.theta_c2 ./ (cca.delta_c2 + cca.theta_c2 + cca.alpha_c2 + cca.beta_c2 + cca.gamma_c2) ;
    avg.RPL.theta_c2 = avg.theta_c2 ./ sum(avg.Regional_c2,2) ;
    cca.RPL.alpha_c2 = cca.alpha_c2 ./ (cca.delta_c2 + cca.theta_c2 + cca.alpha_c2 + cca.beta_c2 + cca.gamma_c2) ;
    avg.RPL.alpha_c2 = avg.alpha_c2 ./ sum(avg.Regional_c2,2) ;
    cca.RPL.beta_c2 = cca.beta_c2 ./ (cca.delta_c2 + cca.theta_c2 + cca.alpha_c2 + cca.beta_c2 + cca.gamma_c2) ;
    avg.RPL.beta_c2 = avg.beta_c2 ./ sum(avg.Regional_c2,2) ;
    cca.RPL.gamma_c2 = cca.gamma_c2 ./ (cca.delta_c2 + cca.theta_c2 + cca.alpha_c2 + cca.beta_c2 + cca.gamma_c2) ;
    avg.RPL.gamma_c2 = avg.gamma_c2 ./ sum(avg.Regional_c2,2) ;     
    
    cca.RPL.Regional_c2 = [cca.RPL.delta_c2 cca.RPL.theta_c2 cca.RPL.alpha_c2 cca.RPL.beta_c2 cca.RPL.gamma_c2]; 
    avg.RPL.Regional_c2 = [avg.RPL.delta_c2 avg.RPL.theta_c2 avg.RPL.alpha_c2 avg.RPL.beta_c2 avg.RPL.gamma_c2];
    
    cca.z.Regional_c2 = [zscore(cca.delta_c2) zscore(cca.theta_c2) zscore(cca.alpha_c2) zscore(cca.beta_c2) zscore(cca.gamma_c2)];
    avg.z.Regional_c2 = [zscore(avg.delta_c2) zscore(avg.theta_c2) zscore(avg.alpha_c2) zscore(avg.beta_c2) zscore(avg.gamma_c2)];                   
    
    [cca.delta_c3,avg.delta_c3] = Power_spectrum(channel_c3, [0.5 4], cnt.fs);
    [cca.theta_c3,avg.theta_c3] = Power_spectrum(channel_c3, [4 8], cnt.fs);
    [cca.alpha_c3,avg.alpha_c3] = Power_spectrum(channel_c3, [8 13], cnt.fs);
    [cca.beta_c3,avg.beta_c3] = Power_spectrum(channel_c3, [13 30], cnt.fs);
    [cca.gamma_c3,avg.gamma_c3] = Power_spectrum(channel_c3, [30 40], cnt.fs);
    avg.delta_c3 = avg.delta_c3'; avg.theta_c3=avg.theta_c3'; avg.alpha_c3=avg.alpha_c3'; avg.beta_c3 = avg.beta_c3'; avg.gamma_c3 = avg.gamma_c3';
    
    cca.Regional_c3 = [cca.delta_c3 cca.theta_c3 cca.alpha_c3 cca.beta_c3 cca.gamma_c3];
    avg.Regional_c3 = [avg.delta_c3 avg.theta_c3 avg.alpha_c3 avg.beta_c3 avg.gamma_c3];
    
    cca.RPL.delta_c3 = cca.delta_c3 ./ (cca.delta_c3 + cca.theta_c3 + cca.alpha_c3 + cca.beta_c3 + cca.gamma_c3) ;
    avg.RPL.delta_c3 = avg.delta_c3 ./ sum(avg.Regional_c3,2) ;
    cca.RPL.theta_c3 = cca.theta_c3 ./ (cca.delta_c3 + cca.theta_c3 + cca.alpha_c3 + cca.beta_c3 + cca.gamma_c3) ;
    avg.RPL.theta_c3 = avg.theta_c3 ./ sum(avg.Regional_c3,2) ;
    cca.RPL.alpha_c3 = cca.alpha_c3 ./ (cca.delta_c3 + cca.theta_c3 + cca.alpha_c3 + cca.beta_c3 + cca.gamma_c3) ;
    avg.RPL.alpha_c3 = avg.alpha_c3 ./ sum(avg.Regional_c3,2) ;
    cca.RPL.beta_c3 = cca.beta_c3 ./ (cca.delta_c3 + cca.theta_c3 + cca.alpha_c3 + cca.beta_c3 + cca.gamma_c3) ;
    avg.RPL.beta_c3 = avg.beta_c3 ./ sum(avg.Regional_c3,2) ;
    cca.RPL.gamma_c3 = cca.gamma_c3 ./ (cca.delta_c3 + cca.theta_c3 + cca.alpha_c3 + cca.beta_c3 + cca.gamma_c3) ;
    avg.RPL.gamma_c3 = avg.gamma_c3 ./ sum(avg.Regional_c3,2) ;     
    
    cca.RPL.Regional_c3 = [cca.RPL.delta_c3 cca.RPL.theta_c3 cca.RPL.alpha_c3 cca.RPL.beta_c3 cca.RPL.gamma_c3]; 
    avg.RPL.Regional_c3 = [avg.RPL.delta_c3 avg.RPL.theta_c3 avg.RPL.alpha_c3 avg.RPL.beta_c3 avg.RPL.gamma_c3];
    
    cca.z.Regional_c3 = [zscore(cca.delta_c3) zscore(cca.theta_c3) zscore(cca.alpha_c3) zscore(cca.beta_c3) zscore(cca.gamma_c3)];
    avg.z.Regional_c3 = [zscore(avg.delta_c3) zscore(avg.theta_c3) zscore(avg.alpha_c3) zscore(avg.beta_c3) zscore(avg.gamma_c3)];                   
    
    [cca.delta_p1,avg.delta_p1] = Power_spectrum(channel_p1, [0.5 4], cnt.fs);
    [cca.theta_p1,avg.theta_p1] = Power_spectrum(channel_p1, [4 8], cnt.fs);
    [cca.alpha_p1,avg.alpha_p1] = Power_spectrum(channel_p1, [8 13], cnt.fs);
    [cca.beta_p1,avg.beta_p1] = Power_spectrum(channel_p1, [13 30], cnt.fs);
    [cca.gamma_p1,avg.gamma_p1] = Power_spectrum(channel_p1, [30 40], cnt.fs);
    avg.delta_p1 = avg.delta_p1'; avg.theta_p1=avg.theta_p1'; avg.alpha_p1=avg.alpha_p1'; avg.beta_p1 = avg.beta_p1'; avg.gamma_p1 = avg.gamma_p1';
    
    cca.Regional_p1 = [cca.delta_p1 cca.theta_p1 cca.alpha_p1 cca.beta_p1 cca.gamma_p1];
    avg.Regional_p1 = [avg.delta_p1 avg.theta_p1 avg.alpha_p1 avg.beta_p1 avg.gamma_p1];
    
    cca.RPL.delta_p1 = cca.delta_p1 ./ (cca.delta_p1 + cca.theta_p1 + cca.alpha_p1 + cca.beta_p1 + cca.gamma_p1) ;
    avg.RPL.delta_p1 = avg.delta_p1 ./ sum(avg.Regional_p1,2) ;
    cca.RPL.theta_p1 = cca.theta_p1 ./ (cca.delta_p1 + cca.theta_p1 + cca.alpha_p1 + cca.beta_p1 + cca.gamma_p1) ;
    avg.RPL.theta_p1 = avg.theta_p1 ./ sum(avg.Regional_p1,2) ;
    cca.RPL.alpha_p1 = cca.alpha_p1 ./ (cca.delta_p1 + cca.theta_p1 + cca.alpha_p1 + cca.beta_p1 + cca.gamma_p1) ;
    avg.RPL.alpha_p1 = avg.alpha_p1 ./ sum(avg.Regional_p1,2) ;
    cca.RPL.beta_p1 = cca.beta_p1 ./ (cca.delta_p1 + cca.theta_p1 + cca.alpha_p1 + cca.beta_p1 + cca.gamma_p1) ;
    avg.RPL.beta_p1 = avg.beta_p1 ./ sum(avg.Regional_p1,2) ;
    cca.RPL.gamma_p1 = cca.gamma_p1 ./ (cca.delta_p1 + cca.theta_p1 + cca.alpha_p1 + cca.beta_p1 + cca.gamma_p1) ;
    avg.RPL.gamma_p1 = avg.gamma_p1 ./ sum(avg.Regional_p1,2) ;     
    
    cca.RPL.Regional_p1 = [cca.RPL.delta_p1 cca.RPL.theta_p1 cca.RPL.alpha_p1 cca.RPL.beta_p1 cca.RPL.gamma_p1]; 
    avg.RPL.Regional_p1 = [avg.RPL.delta_p1 avg.RPL.theta_p1 avg.RPL.alpha_p1 avg.RPL.beta_p1 avg.RPL.gamma_p1];
    
    cca.z.Regional_p1 = [zscore(cca.delta_p1) zscore(cca.theta_p1) zscore(cca.alpha_p1) zscore(cca.beta_p1) zscore(cca.gamma_p1)];
    avg.z.Regional_p1 = [zscore(avg.delta_p1) zscore(avg.theta_p1) zscore(avg.alpha_p1) zscore(avg.beta_p1) zscore(avg.gamma_p1)];               
    
    [cca.delta_p2,avg.delta_p2] = Power_spectrum(channel_p2, [0.5 4], cnt.fs);
    [cca.theta_p2,avg.theta_p2] = Power_spectrum(channel_p2, [4 8], cnt.fs);
    [cca.alpha_p2,avg.alpha_p2] = Power_spectrum(channel_p2, [8 13], cnt.fs);
    [cca.beta_p2,avg.beta_p2] = Power_spectrum(channel_p2, [13 30], cnt.fs);
    [cca.gamma_p2,avg.gamma_p2] = Power_spectrum(channel_p2, [30 40], cnt.fs);
    avg.delta_p2 = avg.delta_p2'; avg.theta_p2=avg.theta_p2'; avg.alpha_p2=avg.alpha_p2'; avg.beta_p2 = avg.beta_p2'; avg.gamma_p2 = avg.gamma_p2';
    
    cca.Regional_p2 = [cca.delta_p2 cca.theta_p2 cca.alpha_p2 cca.beta_p2 cca.gamma_p2];
    avg.Regional_p2 = [avg.delta_p2 avg.theta_p2 avg.alpha_p2 avg.beta_p2 avg.gamma_p2];
    
    cca.RPL.delta_p2 = cca.delta_p2 ./ (cca.delta_p2 + cca.theta_p2 + cca.alpha_p2 + cca.beta_p2 + cca.gamma_p2) ;
    avg.RPL.delta_p2 = avg.delta_p2 ./ sum(avg.Regional_p2,2) ;
    cca.RPL.theta_p2 = cca.theta_p2 ./ (cca.delta_p2 + cca.theta_p2 + cca.alpha_p2 + cca.beta_p2 + cca.gamma_p2) ;
    avg.RPL.theta_p2 = avg.theta_p2 ./ sum(avg.Regional_p2,2) ;
    cca.RPL.alpha_p2 = cca.alpha_p2 ./ (cca.delta_p2 + cca.theta_p2 + cca.alpha_p2 + cca.beta_p2 + cca.gamma_p2) ;
    avg.RPL.alpha_p2 = avg.alpha_p2 ./ sum(avg.Regional_p2,2) ;
    cca.RPL.beta_p2 = cca.beta_p2 ./ (cca.delta_p2 + cca.theta_p2 + cca.alpha_p2 + cca.beta_p2 + cca.gamma_p2) ;
    avg.RPL.beta_p2 = avg.beta_p2 ./ sum(avg.Regional_p2,2) ;
    cca.RPL.gamma_p2 = cca.gamma_p2 ./ (cca.delta_p2 + cca.theta_p2 + cca.alpha_p2 + cca.beta_p2 + cca.gamma_p2) ;
    avg.RPL.gamma_p2 = avg.gamma_p2 ./ sum(avg.Regional_p2,2) ;     
    
    cca.RPL.Regional_p2 = [cca.RPL.delta_p2 cca.RPL.theta_p2 cca.RPL.alpha_p2 cca.RPL.beta_p2 cca.RPL.gamma_p2]; 
    avg.RPL.Regional_p2 = [avg.RPL.delta_p2 avg.RPL.theta_p2 avg.RPL.alpha_p2 avg.RPL.beta_p2 avg.RPL.gamma_p2];
    
    cca.z.Regional_p2 = [zscore(cca.delta_p2) zscore(cca.theta_p2) zscore(cca.alpha_p2) zscore(cca.beta_p2) zscore(cca.gamma_p2)];
    avg.z.Regional_p2 = [zscore(avg.delta_p2) zscore(avg.theta_p2) zscore(avg.alpha_p2) zscore(avg.beta_p2) zscore(avg.gamma_p2)];               
    
    
    [cca.delta_p3,avg.delta_p3] = Power_spectrum(channel_p3, [0.5 4], cnt.fs);
    [cca.theta_p3,avg.theta_p3] = Power_spectrum(channel_p3, [4 8], cnt.fs);
    [cca.alpha_p3,avg.alpha_p3] = Power_spectrum(channel_p3, [8 13], cnt.fs);
    [cca.beta_p3,avg.beta_p3] = Power_spectrum(channel_p3, [13 30], cnt.fs);
    [cca.gamma_p3,avg.gamma_p3] = Power_spectrum(channel_p3, [30 40], cnt.fs);
    avg.delta_p3 = avg.delta_p3'; avg.theta_p3=avg.theta_p3'; avg.alpha_p3=avg.alpha_p3'; avg.beta_p3 = avg.beta_p3'; avg.gamma_p3 = avg.gamma_p3';
    
    cca.Regional_p3 = [cca.delta_p3 cca.theta_p3 cca.alpha_p3 cca.beta_p3 cca.gamma_p3];
    avg.Regional_p3 = [avg.delta_p3 avg.theta_p3 avg.alpha_p3 avg.beta_p3 avg.gamma_p3];
    
    cca.RPL.delta_p3 = cca.delta_p3 ./ (cca.delta_p3 + cca.theta_p3 + cca.alpha_p3 + cca.beta_p3 + cca.gamma_p3) ;
    avg.RPL.delta_p3 = avg.delta_p3 ./ sum(avg.Regional_p3,2) ;
    cca.RPL.theta_p3 = cca.theta_p3 ./ (cca.delta_p3 + cca.theta_p3 + cca.alpha_p3 + cca.beta_p3 + cca.gamma_p3) ;
    avg.RPL.theta_p3 = avg.theta_p3 ./ sum(avg.Regional_p3,2) ;
    cca.RPL.alpha_p3 = cca.alpha_p3 ./ (cca.delta_p3 + cca.theta_p3 + cca.alpha_p3 + cca.beta_p3 + cca.gamma_p3) ;
    avg.RPL.alpha_p3 = avg.alpha_p3 ./ sum(avg.Regional_p3,2) ;
    cca.RPL.beta_p3 = cca.beta_p3 ./ (cca.delta_p3 + cca.theta_p3 + cca.alpha_p3 + cca.beta_p3 + cca.gamma_p3) ;
    avg.RPL.beta_p3 = avg.beta_p3 ./ sum(avg.Regional_p3,2) ;
    cca.RPL.gamma_p3 = cca.gamma_p3 ./ (cca.delta_p3 + cca.theta_p3 + cca.alpha_p3 + cca.beta_p3 + cca.gamma_p3) ;
    avg.RPL.gamma_p3 = avg.gamma_p3 ./ sum(avg.Regional_p3,2) ;     
    
    cca.RPL.Regional_p3 = [cca.RPL.delta_p3 cca.RPL.theta_p3 cca.RPL.alpha_p3 cca.RPL.beta_p3 cca.RPL.gamma_p3]; 
    avg.RPL.Regional_p3 = [avg.RPL.delta_p3 avg.RPL.theta_p3 avg.RPL.alpha_p3 avg.RPL.beta_p3 avg.RPL.gamma_p3];
    
    cca.z.Regional_p3 = [zscore(cca.delta_p3) zscore(cca.theta_p3) zscore(cca.alpha_p3) zscore(cca.beta_p3) zscore(cca.gamma_p3)];
    avg.z.Regional_p3 = [zscore(avg.delta_p3) zscore(avg.theta_p3) zscore(avg.alpha_p3) zscore(avg.beta_p3) zscore(avg.gamma_p3)];               
    
    
    [cca.delta_t1,avg.delta_t1] = Power_spectrum(channel_t1, [0.5 4], cnt.fs);
    [cca.theta_t1,avg.theta_t1] = Power_spectrum(channel_t1, [4 8], cnt.fs);
    [cca.alpha_t1,avg.alpha_t1] = Power_spectrum(channel_t1, [8 13], cnt.fs);
    [cca.beta_t1,avg.beta_t1] = Power_spectrum(channel_t1, [13 30], cnt.fs);
    [cca.gamma_t1,avg.gamma_t1] = Power_spectrum(channel_t1, [30 40], cnt.fs);
    avg.delta_t1 = avg.delta_t1'; avg.theta_t1=avg.theta_t1'; avg.alpha_t1=avg.alpha_t1'; avg.beta_t1 = avg.beta_t1'; avg.gamma_t1 = avg.gamma_t1';
    
    cca.Regional_t1 = [cca.delta_t1 cca.theta_t1 cca.alpha_t1 cca.beta_t1 cca.gamma_t1];
    avg.Regional_t1 = [avg.delta_t1 avg.theta_t1 avg.alpha_t1 avg.beta_t1 avg.gamma_t1];
    
    cca.RPL.delta_t1 = cca.delta_t1 ./ (cca.delta_t1 + cca.theta_t1 + cca.alpha_t1 + cca.beta_t1 + cca.gamma_t1) ;
    avg.RPL.delta_t1 = avg.delta_t1 ./ sum(avg.Regional_t1,2) ;
    cca.RPL.theta_t1 = cca.theta_t1 ./ (cca.delta_t1 + cca.theta_t1 + cca.alpha_t1 + cca.beta_t1 + cca.gamma_t1) ;
    avg.RPL.theta_t1 = avg.theta_t1 ./ sum(avg.Regional_t1,2) ;
    cca.RPL.alpha_t1 = cca.alpha_t1 ./ (cca.delta_t1 + cca.theta_t1 + cca.alpha_t1 + cca.beta_t1 + cca.gamma_t1) ;
    avg.RPL.alpha_t1 = avg.alpha_t1 ./ sum(avg.Regional_t1,2) ;
    cca.RPL.beta_t1 = cca.beta_t1 ./ (cca.delta_t1 + cca.theta_t1 + cca.alpha_t1 + cca.beta_t1 + cca.gamma_t1) ;
    avg.RPL.beta_t1 = avg.beta_t1 ./ sum(avg.Regional_t1,2) ;
    cca.RPL.gamma_t1 = cca.gamma_t1 ./ (cca.delta_t1 + cca.theta_t1 + cca.alpha_t1 + cca.beta_t1 + cca.gamma_t1) ;
    avg.RPL.gamma_t1 = avg.gamma_t1 ./ sum(avg.Regional_t1,2) ;     

    cca.RPL.Regional_t1 = [cca.RPL.delta_t1 cca.RPL.theta_t1 cca.RPL.alpha_t1 cca.RPL.beta_t1 cca.RPL.gamma_t1]; 
    avg.RPL.Regional_t1 = [avg.RPL.delta_t1 avg.RPL.theta_t1 avg.RPL.alpha_t1 avg.RPL.beta_t1 avg.RPL.gamma_t1];
    
    cca.z.Regional_t1 = [zscore(cca.delta_t1) zscore(cca.theta_t1) zscore(cca.alpha_t1) zscore(cca.beta_t1) zscore(cca.gamma_t1)];
    avg.z.Regional_t1 = [zscore(avg.delta_t1) zscore(avg.theta_t1) zscore(avg.alpha_t1) zscore(avg.beta_t1) zscore(avg.gamma_t1)];                   
    
    
    [cca.delta_t2,avg.delta_t2] = Power_spectrum(channel_t2, [0.5 4], cnt.fs);
    [cca.theta_t2,avg.theta_t2] = Power_spectrum(channel_t2, [4 8], cnt.fs);
    [cca.alpha_t2,avg.alpha_t2] = Power_spectrum(channel_t2, [8 13], cnt.fs);
    [cca.beta_t2,avg.beta_t2] = Power_spectrum(channel_t2, [13 30], cnt.fs);
    [cca.gamma_t2,avg.gamma_t2] = Power_spectrum(channel_t2, [30 40], cnt.fs);
    avg.delta_t2 = avg.delta_t2'; avg.theta_t2=avg.theta_t2'; avg.alpha_t2=avg.alpha_t2'; avg.beta_t2 = avg.beta_t2'; avg.gamma_t2 = avg.gamma_t2';
    
    cca.Regional_t2 = [cca.delta_t2 cca.theta_t2 cca.alpha_t2 cca.beta_t2 cca.gamma_t2];
    avg.Regional_t2 = [avg.delta_t2 avg.theta_t2 avg.alpha_t2 avg.beta_t2 avg.gamma_t2];
    
    cca.RPL.delta_t2 = cca.delta_t2 ./ (cca.delta_t2 + cca.theta_t2 + cca.alpha_t2 + cca.beta_t2 + cca.gamma_t2) ;
    avg.RPL.delta_t2 = avg.delta_t2 ./ sum(avg.Regional_t2,2) ;
    cca.RPL.theta_t2 = cca.theta_t2 ./ (cca.delta_t2 + cca.theta_t2 + cca.alpha_t2 + cca.beta_t2 + cca.gamma_t2) ;
    avg.RPL.theta_t2 = avg.theta_t2 ./ sum(avg.Regional_t2,2) ;
    cca.RPL.alpha_t2 = cca.alpha_t2 ./ (cca.delta_t2 + cca.theta_t2 + cca.alpha_t2 + cca.beta_t2 + cca.gamma_t2) ;
    avg.RPL.alpha_t2 = avg.alpha_t2 ./ sum(avg.Regional_t2,2) ;
    cca.RPL.beta_t2 = cca.beta_t2 ./ (cca.delta_t2 + cca.theta_t2 + cca.alpha_t2 + cca.beta_t2 + cca.gamma_t2) ;
    avg.RPL.beta_t2 = avg.beta_t2 ./ sum(avg.Regional_t2,2) ;
    cca.RPL.gamma_t2 = cca.gamma_t2 ./ (cca.delta_t2 + cca.theta_t2 + cca.alpha_t2 + cca.beta_t2 + cca.gamma_t2) ;
    avg.RPL.gamma_t2 = avg.gamma_t2 ./ sum(avg.Regional_t2,2) ;     
    
    cca.RPL.Regional_t2 = [cca.RPL.delta_t2 cca.RPL.theta_t2 cca.RPL.alpha_t2 cca.RPL.beta_t2 cca.RPL.gamma_t2]; 
    avg.RPL.Regional_t2 = [avg.RPL.delta_t2 avg.RPL.theta_t2 avg.RPL.alpha_t2 avg.RPL.beta_t2 avg.RPL.gamma_t2];
    
    cca.z.Regional_t2 = [zscore(cca.delta_t2) zscore(cca.theta_t2) zscore(cca.alpha_t2) zscore(cca.beta_t2) zscore(cca.gamma_t2)];
    avg.z.Regional_t2 = [zscore(avg.delta_t2) zscore(avg.theta_t2) zscore(avg.alpha_t2) zscore(avg.beta_t2) zscore(avg.gamma_t2)];                   
    
    
    
    
    
    [cca.delta_o1,avg.delta_o1] = Power_spectrum(channel_o1, [0.5 4], cnt.fs);
    [cca.theta_o1,avg.theta_o1] = Power_spectrum(channel_o1, [4 8], cnt.fs);
    [cca.alpha_o1,avg.alpha_o1] = Power_spectrum(channel_o1, [8 13], cnt.fs);
    [cca.beta_o1,avg.beta_o1] = Power_spectrum(channel_o1, [13 30], cnt.fs);
    [cca.gamma_o1,avg.gamma_o1] = Power_spectrum(channel_o1, [30 40], cnt.fs);
    avg.delta_o1 = avg.delta_o1'; avg.theta_o1=avg.theta_o1'; avg.alpha_o1=avg.alpha_o1'; avg.beta_o1 = avg.beta_o1'; avg.gamma_o1 = avg.gamma_o1';
    
    cca.Regional_o1 = [cca.delta_o1 cca.theta_o1 cca.alpha_o1 cca.beta_o1 cca.gamma_o1];
    avg.Regional_o1 = [avg.delta_o1 avg.theta_o1 avg.alpha_o1 avg.beta_o1 avg.gamma_o1];

    cca.RPL.delta_o1 = cca.delta_o1 ./ (cca.delta_o1 + cca.theta_o1 + cca.alpha_o1 + cca.beta_o1 + cca.gamma_o1) ;
    avg.RPL.delta_o1 = avg.delta_o1 ./ sum(avg.Regional_o1,2) ;
    cca.RPL.theta_o1 = cca.theta_o1 ./ (cca.delta_o1 + cca.theta_o1 + cca.alpha_o1 + cca.beta_o1 + cca.gamma_o1) ;
    avg.RPL.theta_o1 = avg.theta_o1 ./ sum(avg.Regional_o1,2) ;
    cca.RPL.alpha_o1 = cca.alpha_o1 ./ (cca.delta_o1 + cca.theta_o1 + cca.alpha_o1 + cca.beta_o1 + cca.gamma_o1) ;
    avg.RPL.alpha_o1 = avg.alpha_o1 ./ sum(avg.Regional_o1,2) ;
    cca.RPL.beta_o1 = cca.beta_o1 ./ (cca.delta_o1 + cca.theta_o1 + cca.alpha_o1 + cca.beta_o1 + cca.gamma_o1) ;
    avg.RPL.beta_o1 = avg.beta_o1 ./ sum(avg.Regional_o1,2) ;
    cca.RPL.gamma_o1 = cca.gamma_o1 ./ (cca.delta_o1 + cca.theta_o1 + cca.alpha_o1 + cca.beta_o1 + cca.gamma_o1) ;
    avg.RPL.gamma_o1 = avg.gamma_o1 ./ sum(avg.Regional_o1,2) ;     

    cca.RPL.Regional_o1 = [cca.RPL.delta_o1 cca.RPL.theta_o1 cca.RPL.alpha_o1 cca.RPL.beta_o1 cca.RPL.gamma_o1]; 
    avg.RPL.Regional_o1 = [avg.RPL.delta_o1 avg.RPL.theta_o1 avg.RPL.alpha_o1 avg.RPL.beta_o1 avg.RPL.gamma_o1];
    
    cca.z.Regional_o1 = [zscore(cca.delta_o1) zscore(cca.theta_o1) zscore(cca.alpha_o1) zscore(cca.beta_o1) zscore(cca.gamma_o1)];
    avg.z.Regional_o1 = [zscore(avg.delta_o1) zscore(avg.theta_o1) zscore(avg.alpha_o1) zscore(avg.beta_o1) zscore(avg.gamma_o1)];                   

    
    
    
    [cca.delta_o2,avg.delta_o2] = Power_spectrum(channel_o2, [0.5 4], cnt.fs);
    [cca.theta_o2,avg.theta_o2] = Power_spectrum(channel_o2, [4 8], cnt.fs);
    [cca.alpha_o2,avg.alpha_o2] = Power_spectrum(channel_o2, [8 13], cnt.fs);
    [cca.beta_o2,avg.beta_o2] = Power_spectrum(channel_o2, [13 30], cnt.fs);
    [cca.gamma_o2,avg.gamma_o2] = Power_spectrum(channel_o2, [30 40], cnt.fs);
    avg.delta_o2 = avg.delta_o2'; avg.theta_o2=avg.theta_o2'; avg.alpha_o2=avg.alpha_o2'; avg.beta_o2 = avg.beta_o2'; avg.gamma_o2 = avg.gamma_o2';
    
    cca.Regional_o2 = [cca.delta_o2 cca.theta_o2 cca.alpha_o2 cca.beta_o2 cca.gamma_o2];
    avg.Regional_o2 = [avg.delta_o2 avg.theta_o2 avg.alpha_o2 avg.beta_o2 avg.gamma_o2];
    
    cca.RPL.delta_o2 = cca.delta_o2 ./ (cca.delta_o2 + cca.theta_o2 + cca.alpha_o2 + cca.beta_o2 + cca.gamma_o2) ;
    avg.RPL.delta_o2 = avg.delta_o2 ./ sum(avg.Regional_o2,2) ;
    cca.RPL.theta_o2 = cca.theta_o2 ./ (cca.delta_o2 + cca.theta_o2 + cca.alpha_o2 + cca.beta_o2 + cca.gamma_o2) ;
    avg.RPL.theta_o2 = avg.theta_o2 ./ sum(avg.Regional_o2,2) ;
    cca.RPL.alpha_o2 = cca.alpha_o2 ./ (cca.delta_o2 + cca.theta_o2 + cca.alpha_o2 + cca.beta_o2 + cca.gamma_o2) ;
    avg.RPL.alpha_o2 = avg.alpha_o2 ./ sum(avg.Regional_o2,2) ;
    cca.RPL.beta_o2 = cca.beta_o2 ./ (cca.delta_o2 + cca.theta_o2 + cca.alpha_o2 + cca.beta_o2 + cca.gamma_o2) ;
    avg.RPL.beta_o2 = avg.beta_o2 ./ sum(avg.Regional_o2,2) ;
    cca.RPL.gamma_o2 = cca.gamma_o2 ./ (cca.delta_o2 + cca.theta_o2 + cca.alpha_o2 + cca.beta_o2 + cca.gamma_o2) ;
    avg.RPL.gamma_o2 = avg.gamma_o2 ./ sum(avg.Regional_o2,2) ;     

    cca.RPL.Regional_o2 = [cca.RPL.delta_o2 cca.RPL.theta_o2 cca.RPL.alpha_o2 cca.RPL.beta_o2 cca.RPL.gamma_o2]; 
    avg.RPL.Regional_o2 = [avg.RPL.delta_o2 avg.RPL.theta_o2 avg.RPL.alpha_o2 avg.RPL.beta_o2 avg.RPL.gamma_o2];
        
    cca.z.Regional_o2 = [zscore(cca.delta_o2) zscore(cca.theta_o2) zscore(cca.alpha_o2) zscore(cca.beta_o2) zscore(cca.gamma_o2)];
    avg.z.Regional_o2 = [zscore(avg.delta_o2) zscore(avg.theta_o2) zscore(avg.alpha_o2) zscore(avg.beta_o2) zscore(avg.gamma_o2)];                   
    
    
    
    [cca.delta_o3,avg.delta_o3] = Power_spectrum(channel_o3, [0.5 4], cnt.fs);
    [cca.theta_o3,avg.theta_o3] = Power_spectrum(channel_o3, [4 8], cnt.fs);
    [cca.alpha_o3,avg.alpha_o3] = Power_spectrum(channel_o3, [8 13], cnt.fs);
    [cca.beta_o3,avg.beta_o3] = Power_spectrum(channel_o3, [13 30], cnt.fs);
    [cca.gamma_o3,avg.gamma_o3] = Power_spectrum(channel_o3, [30 40], cnt.fs);
    avg.delta_o3 = avg.delta_o3'; avg.theta_o3=avg.theta_o3'; avg.alpha_o3=avg.alpha_o3'; avg.beta_o3 = avg.beta_o3'; avg.gamma_o3 = avg.gamma_o3';
    
    cca.Regional_o3 = [cca.delta_o3 cca.theta_o3 cca.alpha_o3 cca.beta_o3 cca.gamma_o3];
    avg.Regional_o3 = [avg.delta_o3 avg.theta_o3 avg.alpha_o3 avg.beta_o3 avg.gamma_o3];

    cca.RPL.delta_o3 = cca.delta_o3 ./ (cca.delta_o3 + cca.theta_o3 + cca.alpha_o3 + cca.beta_o3 + cca.gamma_o3) ;
    avg.RPL.delta_o3 = avg.delta_o3 ./ sum(avg.Regional_o3,2) ;
    cca.RPL.theta_o3 = cca.theta_o3 ./ (cca.delta_o3 + cca.theta_o3 + cca.alpha_o3 + cca.beta_o3 + cca.gamma_o3) ;
    avg.RPL.theta_o3 = avg.theta_o3 ./ sum(avg.Regional_o3,2) ;
    cca.RPL.alpha_o3 = cca.alpha_o3 ./ (cca.delta_o3 + cca.theta_o3 + cca.alpha_o3 + cca.beta_o3 + cca.gamma_o3) ;
    avg.RPL.alpha_o3 = avg.alpha_o3 ./ sum(avg.Regional_o3,2) ;
    cca.RPL.beta_o3 = cca.beta_o3 ./ (cca.delta_o3 + cca.theta_o3 + cca.alpha_o3 + cca.beta_o3 + cca.gamma_o3) ;
    avg.RPL.beta_o3 = avg.beta_o3 ./ sum(avg.Regional_o3,2) ;
    cca.RPL.gamma_o3 = cca.gamma_o3 ./ (cca.delta_o3 + cca.theta_o3 + cca.alpha_o3 + cca.beta_o3 + cca.gamma_o3) ;
    avg.RPL.gamma_o3 = avg.gamma_o3 ./ sum(avg.Regional_o3,2) ;     

    cca.RPL.Regional_o3 = [cca.RPL.delta_o3 cca.RPL.theta_o3 cca.RPL.alpha_o3 cca.RPL.beta_o3 cca.RPL.gamma_o3]; 
    avg.RPL.Regional_o3 = [avg.RPL.delta_o3 avg.RPL.theta_o3 avg.RPL.alpha_o3 avg.RPL.beta_o3 avg.RPL.gamma_o3];
        
    cca.z.Regional_o3 = [zscore(cca.delta_o3) zscore(cca.theta_o3) zscore(cca.alpha_o3) zscore(cca.beta_o3) zscore(cca.gamma_o3)];
    avg.z.Regional_o3 = [zscore(avg.delta_o3) zscore(avg.theta_o3) zscore(avg.alpha_o3) zscore(avg.beta_o3) zscore(avg.gamma_o3)];                   
    
    
    %% Statistical Analysis
%         lowIdx = find(kss==min(kss)):1:find(kss==min(kss))+200;
%         lowIdx = lowIdx';
%         highIdx = find(kss==max(kss))-200:1:find(kss==max(kss));
%         highIdx = highIdx';
    
    %         최소+1, 최대-1
    lowIdx = min(kss)<kss & min(kss)+1>kss;
    highIdx = max(kss)>kss & max(kss)-1<kss;
    
    % PSD T-TEST 분석
    [h, p, ci, stats] = ttest2(avg.delta(lowIdx), avg.delta(highIdx));
    p_delta(s) = p;
    [h, p, ci, stats] = ttest2(avg.theta(lowIdx), avg.theta(highIdx));
    p_theta(s) = p;
    [h, p, ci, stats] = ttest2(avg.alpha(lowIdx), avg.alpha(highIdx));
    p_alpha(s) = p;
    [h, p, ci, stats] = ttest2(avg.beta(lowIdx), avg.beta(highIdx));
    p_beta(s) = p;
    [h, p, ci, stats] = ttest2(avg.gamma(lowIdx), avg.gamma(highIdx));
    p_gamma(s) = p;
    
    p_analysis(s,1:5) = [p_delta(s), p_theta(s), p_alpha(s), p_beta(s), p_gamma(s)];
    p_analysis_min(s) = find(p_analysis(s,1:5)==min(p_analysis(s,1:5)));
    
    Result.ttest.total(s,:) = p_analysis(s,1:5);
    
    % RPL T-TEST 분석
    [h, p, ci, stats] = ttest2(avg.RPL.delta(lowIdx), avg.RPL.delta(highIdx));
    p_delta(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.theta(lowIdx), avg.RPL.theta(highIdx));
    p_theta(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.alpha(lowIdx), avg.RPL.alpha(highIdx));
    p_alpha(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.beta(lowIdx), avg.RPL.beta(highIdx));
    p_beta(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.gamma(lowIdx), avg.RPL.gamma(highIdx));
    p_gamma(s) = p;
    
    p_analysis_RPL(s,1:5) = [p_delta(s), p_theta(s), p_alpha(s), p_beta(s), p_gamma(s)];
    p_analysis_RPL_min(s) = find(p_analysis_RPL(s,1:5)==min(p_analysis_RPL(s,1:5)));
    
    Result.ttest.total_RPL(s,:) = p_analysis_RPL(s,1:5);    
    
    
    % z-score T-TEST 분석
    [h, p, ci, stats] = ttest2(avg.z.PSD(lowIdx,1), avg.z.PSD(highIdx,1));
    p_delta(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.PSD(lowIdx,2), avg.z.PSD(lowIdx,2));
    p_theta(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.PSD(lowIdx,3), avg.z.PSD(lowIdx,3));
    p_alpha(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.PSD(lowIdx,4), avg.z.PSD(lowIdx,4));
    p_beta(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.PSD(lowIdx,5), avg.z.PSD(lowIdx,5));
    p_gamma(s) = p;
    
    p_analysis_z(s,1:5) = [p_delta(s), p_theta(s), p_alpha(s), p_beta(s), p_gamma(s)];
    p_analysis_z_min(s) = find(p_analysis_z(s,1:5)==min(p_analysis_z(s,1:5)));
    
    Result.ttest.total_z(s,:) = p_analysis_z(s,1:5);    
    
    
    
    % Regional PSD T-TEST 분석
    [h, p, ci, stats] = ttest2(avg.delta_f(lowIdx), avg.delta_f(highIdx));
    p_delta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.theta_f(lowIdx), avg.theta_f(highIdx));
    p_theta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.alpha_f(lowIdx), avg.alpha_f(highIdx));
    p_alpha_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.beta_f(lowIdx), avg.beta_f(highIdx));
    p_beta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.gamma_f(lowIdx), avg.gamma_f(highIdx));
    p_gamma_f(s) = p;
    
    p_analysis_f(s,1:5) = [p_delta_f(s), p_theta_f(s), p_alpha_f(s), p_beta_f(s), p_gamma_f(s)];
    p_analysis_f_min(s) = find(p_analysis_f(s,1:5)==min(p_analysis_f(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.delta_c(lowIdx), avg.delta_c(highIdx));
    p_delta_c(s) = p;
    [h, p, ci, stats] = ttest2(avg.theta_c(lowIdx), avg.theta_c(highIdx));
    p_theta_c(s) = p;
    [h, p, ci, stats] = ttest2(avg.alpha_c(lowIdx), avg.alpha_c(highIdx));
    p_alpha_c(s) = p;
    [h, p, ci, stats] = ttest2(avg.beta_c(lowIdx), avg.beta_c(highIdx));
    p_beta_c(s) = p;
    [h, p, ci, stats] = ttest2(avg.gamma_c(lowIdx), avg.gamma_c(highIdx));
    p_gamma_c(s) = p;
    
    p_analysis_c(s,1:5) = [p_delta_c(s), p_theta_c(s), p_alpha_c(s), p_beta_c(s), p_gamma_c(s)];
    p_analysis_c_min(s) = find(p_analysis_c(s,1:5)==min(p_analysis_c(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.delta_p(lowIdx), avg.delta_p(highIdx));
    p_delta_p(s) = p;
    [h, p, ci, stats] = ttest2(avg.theta_p(lowIdx), avg.theta_p(highIdx));
    p_theta_p(s) = p;
    [h, p, ci, stats] = ttest2(avg.alpha_p(lowIdx), avg.alpha_p(highIdx));
    p_alpha_p(s) = p;
    [h, p, ci, stats] = ttest2(avg.beta_p(lowIdx), avg.beta_p(highIdx));
    p_beta_p(s) = p;
    [h, p, ci, stats] = ttest2(avg.gamma_p(lowIdx), avg.gamma_p(highIdx));
    p_gamma_p(s) = p;
    
    p_analysis_p(s,1:5) = [p_delta_p(s), p_theta_p(s), p_alpha_p(s), p_beta_p(s), p_gamma_p(s)];
    p_analysis_p_min(s) = find(p_analysis_p(s,1:5)==min(p_analysis_p(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.delta_t(lowIdx), avg.delta_t(highIdx));
    p_delta_t(s) = p;
    [h, p, ci, stats] = ttest2(avg.theta_t(lowIdx), avg.theta_t(highIdx));
    p_theta_t(s) = p;
    [h, p, ci, stats] = ttest2(avg.alpha_t(lowIdx), avg.alpha_t(highIdx));
    p_alpha_t(s) = p;
    [h, p, ci, stats] = ttest2(avg.beta_t(lowIdx), avg.beta_t(highIdx));
    p_beta_t(s) = p;
    [h, p, ci, stats] = ttest2(avg.gamma_t(lowIdx), avg.gamma_t(highIdx));
    p_gamma_t(s) = p;
    
    p_analysis_t(s,1:5) = [p_delta_t(s), p_theta_t(s), p_alpha_t(s), p_beta_t(s), p_gamma_t(s)];
    p_analysis_t_min(s) = find(p_analysis_t(s,1:5)==min(p_analysis_t(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.delta_o(lowIdx), avg.delta_o(highIdx));
    p_delta_o(s) = p;
    [h, p, ci, stats] = ttest2(avg.theta_o(lowIdx), avg.theta_o(highIdx));
    p_theta_o(s) = p;
    [h, p, ci, stats] = ttest2(avg.alpha_o(lowIdx), avg.alpha_o(highIdx));
    p_alpha_o(s) = p;
    [h, p, ci, stats] = ttest2(avg.beta_o(lowIdx), avg.beta_o(highIdx));
    p_beta_o(s) = p;
    [h, p, ci, stats] = ttest2(avg.gamma_o(lowIdx), avg.gamma_o(highIdx));
    p_gamma_o(s) = p;
    
    p_analysis_o(s,1:5) = [p_delta_o(s), p_theta_o(s), p_alpha_o(s), p_beta_o(s), p_gamma_o(s)];
    p_analysis_o_min(s) = find(p_analysis_o(s,1:5)==min(p_analysis_o(s,1:5)));
    
    Result.ttest.regional(s,:) = [p_analysis_f(s,:) p_analysis_c(s,:) p_analysis_p(s,:) p_analysis_t(s,:) p_analysis_o(s,:)];

    
    
    [h, p, ci, stats] = ttest2(avg.RPL.delta_f(lowIdx), avg.RPL.delta_f(highIdx));
    p_delta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.theta_f(lowIdx), avg.RPL.theta_f(highIdx));
    p_theta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.alpha_f(lowIdx), avg.RPL.alpha_f(highIdx));
    p_alpha_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.beta_f(lowIdx), avg.RPL.beta_f(highIdx));
    p_beta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.gamma_f(lowIdx), avg.RPL.gamma_f(highIdx));
    p_gamma_f(s) = p;
    
    p_analysis_RPL_f(s,1:5) = [p_delta_f(s), p_theta_f(s), p_alpha_f(s), p_beta_f(s), p_gamma_f(s)];
    p_analysis_RPL_f_min(s) = find(p_analysis_RPL_f(s,1:5)==min(p_analysis_RPL_f(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.RPL.delta_c(lowIdx), avg.RPL.delta_c(highIdx));
    p_delta_c(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.theta_c(lowIdx), avg.RPL.theta_c(highIdx));
    p_theta_c(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.alpha_c(lowIdx), avg.RPL.alpha_c(highIdx));
    p_alpha_c(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.beta_c(lowIdx), avg.RPL.beta_c(highIdx));
    p_beta_c(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.gamma_c(lowIdx), avg.RPL.gamma_c(highIdx));
    p_gamma_c(s) = p;
    
    p_analysis_RPL_c(s,1:5) = [p_delta_c(s), p_theta_c(s), p_alpha_c(s), p_beta_c(s), p_gamma_c(s)];
    p_analysis_RPL_c_min(s) = find(p_analysis_RPL_c(s,1:5)==min(p_analysis_RPL_c(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.RPL.delta_p(lowIdx), avg.RPL.delta_p(highIdx));
    p_delta_p(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.theta_p(lowIdx), avg.RPL.theta_p(highIdx));
    p_theta_p(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.alpha_p(lowIdx), avg.RPL.alpha_p(highIdx));
    p_alpha_p(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.beta_p(lowIdx), avg.RPL.beta_p(highIdx));
    p_beta_p(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.gamma_p(lowIdx), avg.RPL.gamma_p(highIdx));
    p_gamma_p(s) = p;
    
    p_analysis_RPL_p(s,1:5) = [p_delta_p(s), p_theta_p(s), p_alpha_p(s), p_beta_p(s), p_gamma_p(s)];
    p_analysis_RPL_p_min(s) = find(p_analysis_RPL_p(s,1:5)==min(p_analysis_RPL_p(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.RPL.delta_t(lowIdx), avg.RPL.delta_t(highIdx));
    p_delta_t(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.theta_t(lowIdx), avg.RPL.theta_t(highIdx));
    p_theta_t(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.alpha_t(lowIdx), avg.RPL.alpha_t(highIdx));
    p_alpha_t(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.beta_t(lowIdx), avg.RPL.beta_t(highIdx));
    p_beta_t(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.gamma_t(lowIdx), avg.RPL.gamma_t(highIdx));
    p_gamma_t(s) = p;
    
    p_analysis_RPL_t(s,1:5) = [p_delta_t(s), p_theta_t(s), p_alpha_t(s), p_beta_t(s), p_gamma_t(s)];
    p_analysis_RPL_t_min(s) = find(p_analysis_RPL_t(s,1:5)==min(p_analysis_RPL_t(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.RPL.delta_o(lowIdx), avg.RPL.delta_o(highIdx));
    p_delta_o(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.theta_o(lowIdx), avg.RPL.theta_o(highIdx));
    p_theta_o(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.alpha_o(lowIdx), avg.RPL.alpha_o(highIdx));
    p_alpha_o(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.beta_o(lowIdx), avg.RPL.beta_o(highIdx));
    p_beta_o(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.gamma_o(lowIdx), avg.RPL.gamma_o(highIdx));
    p_gamma_o(s) = p;
    
    p_analysis_RPL_o(s,1:5) = [p_delta_o(s), p_theta_o(s), p_alpha_o(s), p_beta_o(s), p_gamma_o(s)];
    p_analysis_RPL_o_min(s) = find(p_analysis_RPL_o(s,1:5)==min(p_analysis_RPL_o(s,1:5)));
    
    Result.ttest.regional_RPL(s,:) = [p_analysis_RPL_f(s,:) p_analysis_RPL_c(s,:) p_analysis_RPL_p(s,:) p_analysis_RPL_t(s,:) p_analysis_RPL_o(s,:)];

    
    
    [h, p, ci, stats] = ttest2(avg.z.Regional_f(lowIdx,1), avg.z.Regional_f(highIdx,1));
    p_delta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_f(lowIdx,2), avg.z.Regional_f(highIdx,2));
    p_theta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_f(lowIdx,3), avg.z.Regional_f(highIdx,3));
    p_alpha_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_f(lowIdx,4), avg.z.Regional_f(highIdx,4));
    p_beta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_f(lowIdx,5), avg.z.Regional_f(highIdx,5));
    p_gamma_f(s) = p;
    
    p_analysis_z_f(s,1:5) = [p_delta_f(s), p_theta_f(s), p_alpha_f(s), p_beta_f(s), p_gamma_f(s)];
    p_analysis_z_f_min(s) = find(p_analysis_z_f(s,1:5)==min(p_analysis_z_f(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.z.Regional_c(lowIdx,1), avg.z.Regional_c(highIdx,1));
    p_delta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_c(lowIdx,2), avg.z.Regional_c(highIdx,2));
    p_theta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_c(lowIdx,3), avg.z.Regional_c(highIdx,3));
    p_alpha_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_c(lowIdx,4), avg.z.Regional_c(highIdx,4));
    p_beta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_c(lowIdx,5), avg.z.Regional_c(highIdx,5));
    p_gamma_f(s) = p;
    
    p_analysis_z_c(s,1:5) = [p_delta_c(s), p_theta_c(s), p_alpha_c(s), p_beta_c(s), p_gamma_c(s)];
    p_analysis_z_c_min(s) = find(p_analysis_z_c(s,1:5)==min(p_analysis_z_c(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.z.Regional_p(lowIdx,1), avg.z.Regional_p(highIdx,1));
    p_delta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_p(lowIdx,2), avg.z.Regional_p(highIdx,2));
    p_theta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_p(lowIdx,3), avg.z.Regional_p(highIdx,3));
    p_alpha_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_p(lowIdx,4), avg.z.Regional_p(highIdx,4));
    p_beta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_p(lowIdx,5), avg.z.Regional_p(highIdx,5));
    p_gamma_f(s) = p;
    
    p_analysis_z_p(s,1:5) = [p_delta_p(s), p_theta_p(s), p_alpha_p(s), p_beta_p(s), p_gamma_p(s)];
    p_analysis_z_p_min(s) = find(p_analysis_z_p(s,1:5)==min(p_analysis_z_p(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.z.Regional_t(lowIdx,1), avg.z.Regional_t(highIdx,1));
    p_delta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_t(lowIdx,2), avg.z.Regional_t(highIdx,2));
    p_theta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_t(lowIdx,3), avg.z.Regional_t(highIdx,3));
    p_alpha_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_t(lowIdx,4), avg.z.Regional_t(highIdx,4));
    p_beta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_t(lowIdx,5), avg.z.Regional_t(highIdx,5));
    p_gamma_f(s) = p;
    
    p_analysis_z_t(s,1:5) = [p_delta_t(s), p_theta_t(s), p_alpha_t(s), p_beta_t(s), p_gamma_t(s)];
    p_analysis_z_t_min(s) = find(p_analysis_z_t(s,1:5)==min(p_analysis_z_t(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.z.Regional_o(lowIdx,1), avg.z.Regional_o(highIdx,1));
    p_delta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_o(lowIdx,2), avg.z.Regional_o(highIdx,2));
    p_theta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_o(lowIdx,3), avg.z.Regional_o(highIdx,3));
    p_alpha_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_o(lowIdx,4), avg.z.Regional_o(highIdx,4));
    p_beta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_o(lowIdx,5), avg.z.Regional_o(highIdx,5));
    p_gamma_f(s) = p;
    
    p_analysis_z_o(s,1:5) = [p_delta_o(s), p_theta_o(s), p_alpha_o(s), p_beta_o(s), p_gamma_o(s)];
    p_analysis_z_o_min(s) = find(p_analysis_z_o(s,1:5)==min(p_analysis_z_o(s,1:5)));
    
    Result.ttest.regional_z(s,:) = [p_analysis_f(s,:) p_analysis_c(s,:) p_analysis_p(s,:) p_analysis_t(s,:) p_analysis_o(s,:)];

    
    
    
    
    
    % Regional Regional T-TEST 분석
    [h, p, ci, stats] = ttest2(avg.delta_f1(lowIdx), avg.delta_f1(highIdx));
    p_delta_f1(s) = p;
    [h, p, ci, stats] = ttest2(avg.theta_f1(lowIdx), avg.theta_f1(highIdx));
    p_theta_f1(s) = p;
    [h, p, ci, stats] = ttest2(avg.alpha_f1(lowIdx), avg.alpha_f1(highIdx));
    p_alpha_f1(s) = p;
    [h, p, ci, stats] = ttest2(avg.beta_f1(lowIdx), avg.beta_f1(highIdx));
    p_beta_f1(s) = p;
    [h, p, ci, stats] = ttest2(avg.gamma_f1(lowIdx), avg.gamma_f1(highIdx));
    p_gamma_f1(s) = p;
    
    p_analysis_f1(s,1:5) = [p_delta_f1(s), p_theta_f1(s), p_alpha_f1(s), p_beta_f1(s), p_gamma_f1(s)];
    p_analysis_f1_min(s) = find(p_analysis_f1(s,1:5)==min(p_analysis_f1(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.delta_f2(lowIdx), avg.delta_f2(highIdx));
    p_delta_f2(s) = p;
    [h, p, ci, stats] = ttest2(avg.theta_f2(lowIdx), avg.theta_f2(highIdx));
    p_theta_f2(s) = p;
    [h, p, ci, stats] = ttest2(avg.alpha_f2(lowIdx), avg.alpha_f2(highIdx));
    p_alpha_f2(s) = p;
    [h, p, ci, stats] = ttest2(avg.beta_f2(lowIdx), avg.beta_f2(highIdx));
    p_beta_f2(s) = p;
    [h, p, ci, stats] = ttest2(avg.gamma_f2(lowIdx), avg.gamma_f2(highIdx));
    p_gamma_f2(s) = p;
    
    p_analysis_f2(s,1:5) = [p_delta_f2(s), p_theta_f2(s), p_alpha_f2(s), p_beta_f2(s), p_gamma_f2(s)];
    p_analysis_f2_min(s) = find(p_analysis_f2(s,1:5)==min(p_analysis_f2(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.delta_f3(lowIdx), avg.delta_f3(highIdx));
    p_delta_f3(s) = p;
    [h, p, ci, stats] = ttest2(avg.theta_f3(lowIdx), avg.theta_f3(highIdx));
    p_theta_f3(s) = p;
    [h, p, ci, stats] = ttest2(avg.alpha_f3(lowIdx), avg.alpha_f3(highIdx));
    p_alpha_f3(s) = p;
    [h, p, ci, stats] = ttest2(avg.beta_f3(lowIdx), avg.beta_f3(highIdx));
    p_beta_f3(s) = p;
    [h, p, ci, stats] = ttest2(avg.gamma_f3(lowIdx), avg.gamma_f3(highIdx));
    p_gamma_f3(s) = p;
    
    p_analysis_f3(s,1:5) = [p_delta_f3(s), p_theta_f3(s), p_alpha_f3(s), p_beta_f3(s), p_gamma_f3(s)];
    p_analysis_f3_min(s) = find(p_analysis_f3(s,1:5)==min(p_analysis_f3(s,1:5)));
    
    
    
    [h, p, ci, stats] = ttest2(avg.delta_c1(lowIdx), avg.delta_c1(highIdx));
    p_delta_c1(s) = p;
    [h, p, ci, stats] = ttest2(avg.theta_c1(lowIdx), avg.theta_c1(highIdx));
    p_theta_c1(s) = p;
    [h, p, ci, stats] = ttest2(avg.alpha_c1(lowIdx), avg.alpha_c1(highIdx));
    p_alpha_c1(s) = p;
    [h, p, ci, stats] = ttest2(avg.beta_c1(lowIdx), avg.beta_c1(highIdx));
    p_beta_c1(s) = p;
    [h, p, ci, stats] = ttest2(avg.gamma_c1(lowIdx), avg.gamma_c1(highIdx));
    p_gamma_c1(s) = p;
    
    p_analysis_c1(s,1:5) = [p_delta_c1(s), p_theta_c1(s), p_alpha_c1(s), p_beta_c1(s), p_gamma_c1(s)];
    p_analysis_c1_min(s) = find(p_analysis_c1(s,1:5)==min(p_analysis_c1(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.delta_c2(lowIdx), avg.delta_c2(highIdx));
    p_delta_c2(s) = p;
    [h, p, ci, stats] = ttest2(avg.theta_c2(lowIdx), avg.theta_c2(highIdx));
    p_theta_c2(s) = p;
    [h, p, ci, stats] = ttest2(avg.alpha_c2(lowIdx), avg.alpha_c2(highIdx));
    p_alpha_c2(s) = p;
    [h, p, ci, stats] = ttest2(avg.beta_c2(lowIdx), avg.beta_c2(highIdx));
    p_beta_c2(s) = p;
    [h, p, ci, stats] = ttest2(avg.gamma_c2(lowIdx), avg.gamma_c2(highIdx));
    p_gamma_c2(s) = p;
    
    p_analysis_c2(s,1:5) = [p_delta_c2(s), p_theta_c2(s), p_alpha_c2(s), p_beta_c2(s), p_gamma_c2(s)];
    p_analysis_c2_min(s) = find(p_analysis_c2(s,1:5)==min(p_analysis_c2(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.delta_c3(lowIdx), avg.delta_c3(highIdx));
    p_delta_c3(s) = p;
    [h, p, ci, stats] = ttest2(avg.theta_c3(lowIdx), avg.theta_c3(highIdx));
    p_theta_c3(s) = p;
    [h, p, ci, stats] = ttest2(avg.alpha_c3(lowIdx), avg.alpha_c3(highIdx));
    p_alpha_c3(s) = p;
    [h, p, ci, stats] = ttest2(avg.beta_c3(lowIdx), avg.beta_c3(highIdx));
    p_beta_c3(s) = p;
    [h, p, ci, stats] = ttest2(avg.gamma_c3(lowIdx), avg.gamma_c3(highIdx));
    p_gamma_c3(s) = p;
    
    p_analysis_c3(s,1:5) = [p_delta_c3(s), p_theta_c3(s), p_alpha_c3(s), p_beta_c3(s), p_gamma_c3(s)];
    p_analysis_c3_min(s) = find(p_analysis_c3(s,1:5)==min(p_analysis_c3(s,1:5)));
    
    

    
    
    [h, p, ci, stats] = ttest2(avg.delta_p1(lowIdx), avg.delta_p1(highIdx));
    p_delta_p1(s) = p;
    [h, p, ci, stats] = ttest2(avg.theta_p1(lowIdx), avg.theta_p1(highIdx));
    p_theta_p1(s) = p;
    [h, p, ci, stats] = ttest2(avg.alpha_p1(lowIdx), avg.alpha_p1(highIdx));
    p_alpha_p1(s) = p;
    [h, p, ci, stats] = ttest2(avg.beta_p1(lowIdx), avg.beta_p1(highIdx));
    p_beta_p1(s) = p;
    [h, p, ci, stats] = ttest2(avg.gamma_p1(lowIdx), avg.gamma_p1(highIdx));
    p_gamma_p1(s) = p;
    
    p_analysis_p1(s,1:5) = [p_delta_p1(s), p_theta_p1(s), p_alpha_p1(s), p_beta_p1(s), p_gamma_p1(s)];
    p_analysis_p1_min(s) = find(p_analysis_p1(s,1:5)==min(p_analysis_p1(s,1:5)));
    
    
    
    [h, p, ci, stats] = ttest2(avg.delta_p2(lowIdx), avg.delta_p2(highIdx));
    p_delta_p2(s) = p;
    [h, p, ci, stats] = ttest2(avg.theta_p2(lowIdx), avg.theta_p2(highIdx));
    p_theta_p2(s) = p;
    [h, p, ci, stats] = ttest2(avg.alpha_p2(lowIdx), avg.alpha_p2(highIdx));
    p_alpha_p2(s) = p;
    [h, p, ci, stats] = ttest2(avg.beta_p2(lowIdx), avg.beta_p2(highIdx));
    p_beta_p2(s) = p;
    [h, p, ci, stats] = ttest2(avg.gamma_p2(lowIdx), avg.gamma_p2(highIdx));
    p_gamma_p2(s) = p;
    
    p_analysis_p2(s,1:5) = [p_delta_p2(s), p_theta_p2(s), p_alpha_p2(s), p_beta_p2(s), p_gamma_p2(s)];
    p_analysis_p2_min(s) = find(p_analysis_p2(s,1:5)==min(p_analysis_p2(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.delta_p3(lowIdx), avg.delta_p3(highIdx));
    p_delta_p3(s) = p;
    [h, p, ci, stats] = ttest2(avg.theta_p3(lowIdx), avg.theta_p3(highIdx));
    p_theta_p3(s) = p;
    [h, p, ci, stats] = ttest2(avg.alpha_p3(lowIdx), avg.alpha_p3(highIdx));
    p_alpha_p3(s) = p;
    [h, p, ci, stats] = ttest2(avg.beta_p3(lowIdx), avg.beta_p3(highIdx));
    p_beta_p3(s) = p;
    [h, p, ci, stats] = ttest2(avg.gamma_p3(lowIdx), avg.gamma_p3(highIdx));
    p_gamma_p3(s) = p;
    
    p_analysis_p3(s,1:5) = [p_delta_p3(s), p_theta_p3(s), p_alpha_p3(s), p_beta_p3(s), p_gamma_p3(s)];
    p_analysis_p3_min(s) = find(p_analysis_p3(s,1:5)==min(p_analysis_p3(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.delta_t1(lowIdx), avg.delta_t1(highIdx));
    p_delta_t1(s) = p;
    [h, p, ci, stats] = ttest2(avg.theta_t1(lowIdx), avg.theta_t1(highIdx));
    p_theta_t1(s) = p;
    [h, p, ci, stats] = ttest2(avg.alpha_t1(lowIdx), avg.alpha_t1(highIdx));
    p_alpha_t1(s) = p;
    [h, p, ci, stats] = ttest2(avg.beta_t1(lowIdx), avg.beta_t1(highIdx));
    p_beta_t1(s) = p;
    [h, p, ci, stats] = ttest2(avg.gamma_t1(lowIdx), avg.gamma_t1(highIdx));
    p_gamma_t1(s) = p;
    
    p_analysis_t1(s,1:5) = [p_delta_t1(s), p_theta_t1(s), p_alpha_t1(s), p_beta_t1(s), p_gamma_t1(s)];
    p_analysis_t1_min(s) = find(p_analysis_t1(s,1:5)==min(p_analysis_t1(s,1:5)));
    
    [h, p, ci, stats] = ttest2(avg.delta_t2(lowIdx), avg.delta_t2(highIdx));
    p_delta_t2(s) = p;
    [h, p, ci, stats] = ttest2(avg.theta_t2(lowIdx), avg.theta_t2(highIdx));
    p_theta_t2(s) = p;
    [h, p, ci, stats] = ttest2(avg.alpha_t2(lowIdx), avg.alpha_t2(highIdx));
    p_alpha_t2(s) = p;
    [h, p, ci, stats] = ttest2(avg.beta_t2(lowIdx), avg.beta_t2(highIdx));
    p_beta_t2(s) = p;
    [h, p, ci, stats] = ttest2(avg.gamma_t2(lowIdx), avg.gamma_t2(highIdx));
    p_gamma_t2(s) = p;
    
    p_analysis_t2(s,1:5) = [p_delta_t2(s), p_theta_t2(s), p_alpha_t2(s), p_beta_t2(s), p_gamma_t2(s)];
    p_analysis_t2_min(s) = find(p_analysis_t2(s,1:5)==min(p_analysis_t2(s,1:5)));
    
    
    
    [h, p, ci, stats] = ttest2(avg.delta_o1(lowIdx), avg.delta_o1(highIdx));
    p_delta_o1(s) = p;
    [h, p, ci, stats] = ttest2(avg.theta_o1(lowIdx), avg.theta_o1(highIdx));
    p_theta_o1(s) = p;
    [h, p, ci, stats] = ttest2(avg.alpha_o1(lowIdx), avg.alpha_o1(highIdx));
    p_alpha_o1(s) = p;
    [h, p, ci, stats] = ttest2(avg.beta_o1(lowIdx), avg.beta_o1(highIdx));
    p_beta_o1(s) = p;
    [h, p, ci, stats] = ttest2(avg.gamma_o1(lowIdx), avg.gamma_o1(highIdx));
    p_gamma_o1(s) = p;
    
    p_analysis_o1(s,1:5) = [p_delta_o1(s), p_theta_o1(s), p_alpha_o1(s), p_beta_o1(s), p_gamma_o1(s)];
    p_analysis_o1_min(s) = find(p_analysis_o1(s,1:5)==min(p_analysis_o1(s,1:5)));
    
    [h, p, ci, stats] = ttest2(avg.delta_o2(lowIdx), avg.delta_o2(highIdx));
    p_delta_o2(s) = p;
    [h, p, ci, stats] = ttest2(avg.theta_o2(lowIdx), avg.theta_o2(highIdx));
    p_theta_o2(s) = p;
    [h, p, ci, stats] = ttest2(avg.alpha_o2(lowIdx), avg.alpha_o2(highIdx));
    p_alpha_o2(s) = p;
    [h, p, ci, stats] = ttest2(avg.beta_o2(lowIdx), avg.beta_o2(highIdx));
    p_beta_o2(s) = p;
    [h, p, ci, stats] = ttest2(avg.gamma_o2(lowIdx), avg.gamma_o2(highIdx));
    p_gamma_o2(s) = p;
    
    p_analysis_o2(s,1:5) = [p_delta_o2(s), p_theta_o2(s), p_alpha_o2(s), p_beta_o2(s), p_gamma_o2(s)];
    p_analysis_o2_min(s) = find(p_analysis_o2(s,1:5)==min(p_analysis_o2(s,1:5)));
    
    [h, p, ci, stats] = ttest2(avg.delta_o3(lowIdx), avg.delta_o3(highIdx));
    p_delta_o3(s) = p;
    [h, p, ci, stats] = ttest2(avg.theta_o3(lowIdx), avg.theta_o3(highIdx));
    p_theta_o3(s) = p;
    [h, p, ci, stats] = ttest2(avg.alpha_o3(lowIdx), avg.alpha_o3(highIdx));
    p_alpha_o3(s) = p;
    [h, p, ci, stats] = ttest2(avg.beta_o3(lowIdx), avg.beta_o3(highIdx));
    p_beta_o3(s) = p;
    [h, p, ci, stats] = ttest2(avg.gamma_o3(lowIdx), avg.gamma_o3(highIdx));
    p_gamma_o3(s) = p;
    
    p_analysis_o3(s,1:5) = [p_delta_o3(s), p_theta_o3(s), p_alpha_o3(s), p_beta_o3(s), p_gamma_o3(s)];
    p_analysis_o3_min(s) = find(p_analysis_o3(s,1:5)==min(p_analysis_o3(s,1:5)));
    
    Result.ttest.vertical(s,:) = [p_analysis_f1(s,:) p_analysis_f2(s,:) p_analysis_f3(s,:) p_analysis_c1(s,:) p_analysis_c2(s,:) p_analysis_c3(s,:) p_analysis_p1(s,:) p_analysis_p2(s,:) p_analysis_p3(s,:) p_analysis_t1(s,:) p_analysis_t2(s,:) p_analysis_o1(s,:) p_analysis_o2(s,:) p_analysis_o3(s,:)];
    
    
    
        % Regional Vertical T-TEST 분석
    [h, p, ci, stats] = ttest2(avg.RPL.delta_f1(lowIdx), avg.RPL.delta_f1(highIdx));
    p_delta_f1(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.theta_f1(lowIdx), avg.RPL.theta_f1(highIdx));
    p_theta_f1(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.alpha_f1(lowIdx), avg.RPL.alpha_f1(highIdx));
    p_alpha_f1(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.beta_f1(lowIdx), avg.RPL.beta_f1(highIdx));
    p_beta_f1(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.gamma_f1(lowIdx), avg.RPL.gamma_f1(highIdx));
    p_gamma_f1(s) = p;
    
    p_analysis_f1_rpl(s,1:5) = [p_delta_f1(s), p_theta_f1(s), p_alpha_f1(s), p_beta_f1(s), p_gamma_f1(s)];
    p_analysis_f1_rpl_min(s) = find(p_analysis_f1_rpl(s,1:5)==min(p_analysis_f1_rpl(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.RPL.delta_f2(lowIdx), avg.RPL.delta_f2(highIdx));
    p_delta_f2(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.theta_f2(lowIdx), avg.RPL.theta_f2(highIdx));
    p_theta_f2(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.alpha_f2(lowIdx), avg.RPL.alpha_f2(highIdx));
    p_alpha_f2(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.beta_f2(lowIdx), avg.RPL.beta_f2(highIdx));
    p_beta_f2(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.gamma_f2(lowIdx), avg.RPL.gamma_f2(highIdx));
    p_gamma_f2(s) = p;
    
    p_analysis_f2_rpl(s,1:5) = [p_delta_f2(s), p_theta_f2(s), p_alpha_f2(s), p_beta_f2(s), p_gamma_f2(s)];
    p_analysis_f2_rpl_min(s) = find(p_analysis_f2_rpl(s,1:5)==min(p_analysis_f2_rpl(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.RPL.delta_f3(lowIdx), avg.RPL.delta_f3(highIdx));
    p_delta_f3(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.theta_f3(lowIdx), avg.RPL.theta_f3(highIdx));
    p_theta_f3(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.alpha_f3(lowIdx), avg.RPL.alpha_f3(highIdx));
    p_alpha_f3(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.beta_f3(lowIdx), avg.RPL.beta_f3(highIdx));
    p_beta_f3(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.gamma_f3(lowIdx), avg.RPL.gamma_f3(highIdx));
    p_gamma_f3(s) = p;
    
    p_analysis_f3_rpl(s,1:5) = [p_delta_f3(s), p_theta_f3(s), p_alpha_f3(s), p_beta_f3(s), p_gamma_f3(s)];
    p_analysis_f3_rpl_min(s) = find(p_analysis_f3_rpl(s,1:5)==min(p_analysis_f3_rpl(s,1:5)));
    
    
    
    [h, p, ci, stats] = ttest2(avg.RPL.delta_c1(lowIdx), avg.RPL.delta_c1(highIdx));
    p_delta_c1(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.theta_c1(lowIdx), avg.RPL.theta_c1(highIdx));
    p_theta_c1(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.alpha_c1(lowIdx), avg.RPL.alpha_c1(highIdx));
    p_alpha_c1(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.beta_c1(lowIdx), avg.RPL.beta_c1(highIdx));
    p_beta_c1(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.gamma_c1(lowIdx), avg.RPL.gamma_c1(highIdx));
    p_gamma_c1(s) = p;
    
    p_analysis_c1_rpl(s,1:5) = [p_delta_c1(s), p_theta_c1(s), p_alpha_c1(s), p_beta_c1(s), p_gamma_c1(s)];
    p_analysis_c1_rpl_min(s) = find(p_analysis_c1_rpl(s,1:5)==min(p_analysis_c1_rpl(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.RPL.delta_c2(lowIdx), avg.RPL.delta_c2(highIdx));
    p_delta_c2(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.theta_c2(lowIdx), avg.RPL.theta_c2(highIdx));
    p_theta_c2(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.alpha_c2(lowIdx), avg.RPL.alpha_c2(highIdx));
    p_alpha_c2(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.beta_c2(lowIdx), avg.RPL.beta_c2(highIdx));
    p_beta_c2(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.gamma_c2(lowIdx), avg.RPL.gamma_c2(highIdx));
    p_gamma_c2(s) = p;
    
    p_analysis_c2_rpl(s,1:5) = [p_delta_c2(s), p_theta_c2(s), p_alpha_c2(s), p_beta_c2(s), p_gamma_c2(s)];
    p_analysis_c2_rpl_min(s) = find(p_analysis_c2_rpl(s,1:5)==min(p_analysis_c2_rpl(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.RPL.delta_c3(lowIdx), avg.RPL.delta_c3(highIdx));
    p_delta_c3(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.theta_c3(lowIdx), avg.RPL.theta_c3(highIdx));
    p_theta_c3(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.alpha_c3(lowIdx), avg.RPL.alpha_c3(highIdx));
    p_alpha_c3(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.beta_c3(lowIdx), avg.RPL.beta_c3(highIdx));
    p_beta_c3(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.gamma_c3(lowIdx), avg.RPL.gamma_c3(highIdx));
    p_gamma_c3(s) = p;
    
    p_analysis_c3_rpl(s,1:5) = [p_delta_c3(s), p_theta_c3(s), p_alpha_c3(s), p_beta_c3(s), p_gamma_c3(s)];
    p_analysis_c3_rpl_min(s) = find(p_analysis_c3_rpl(s,1:5)==min(p_analysis_c3_rpl(s,1:5)));
    
    

    
    
    [h, p, ci, stats] = ttest2(avg.RPL.delta_p1(lowIdx), avg.RPL.delta_p1(highIdx));
    p_delta_p1(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.theta_p1(lowIdx), avg.RPL.theta_p1(highIdx));
    p_theta_p1(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.alpha_p1(lowIdx), avg.RPL.alpha_p1(highIdx));
    p_alpha_p1(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.beta_p1(lowIdx), avg.RPL.beta_p1(highIdx));
    p_beta_p1(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.gamma_p1(lowIdx), avg.RPL.gamma_p1(highIdx));
    p_gamma_p1(s) = p;
    
    p_analysis_p1_rpl(s,1:5) = [p_delta_p1(s), p_theta_p1(s), p_alpha_p1(s), p_beta_p1(s), p_gamma_p1(s)];
    p_analysis_p1_rpl_min(s) = find(p_analysis_p1_rpl(s,1:5)==min(p_analysis_p1_rpl(s,1:5)));
    
    
    
    [h, p, ci, stats] = ttest2(avg.RPL.delta_p2(lowIdx), avg.RPL.delta_p2(highIdx));
    p_delta_p2(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.theta_p2(lowIdx), avg.RPL.theta_p2(highIdx));
    p_theta_p2(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.alpha_p2(lowIdx), avg.RPL.alpha_p2(highIdx));
    p_alpha_p2(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.beta_p2(lowIdx), avg.RPL.beta_p2(highIdx));
    p_beta_p2(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.gamma_p2(lowIdx), avg.RPL.gamma_p2(highIdx));
    p_gamma_p2(s) = p;
    
    p_analysis_p2_rpl(s,1:5) = [p_delta_p2(s), p_theta_p2(s), p_alpha_p2(s), p_beta_p2(s), p_gamma_p2(s)];
    p_analysis_p2_rpl_min(s) = find(p_analysis_p2_rpl(s,1:5)==min(p_analysis_p2_rpl(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.RPL.delta_p3(lowIdx), avg.RPL.delta_p3(highIdx));
    p_delta_p3(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.theta_p3(lowIdx), avg.RPL.theta_p3(highIdx));
    p_theta_p3(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.alpha_p3(lowIdx), avg.RPL.alpha_p3(highIdx));
    p_alpha_p3(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.beta_p3(lowIdx), avg.RPL.beta_p3(highIdx));
    p_beta_p3(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.gamma_p3(lowIdx), avg.RPL.gamma_p3(highIdx));
    p_gamma_p3(s) = p;
    
    p_analysis_p3_rpl(s,1:5) = [p_delta_p3(s), p_theta_p3(s), p_alpha_p3(s), p_beta_p3(s), p_gamma_p3(s)];
    p_analysis_p3_rpl_min(s) = find(p_analysis_p3_rpl(s,1:5)==min(p_analysis_p3_rpl(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.RPL.delta_t1(lowIdx), avg.RPL.delta_t1(highIdx));
    p_delta_t1(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.theta_t1(lowIdx), avg.RPL.theta_t1(highIdx));
    p_theta_t1(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.alpha_t1(lowIdx), avg.RPL.alpha_t1(highIdx));
    p_alpha_t1(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.beta_t1(lowIdx), avg.RPL.beta_t1(highIdx));
    p_beta_t1(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.gamma_t1(lowIdx), avg.RPL.gamma_t1(highIdx));
    p_gamma_t1(s) = p;
    
    p_analysis_t1_rpl(s,1:5) = [p_delta_t1(s), p_theta_t1(s), p_alpha_t1(s), p_beta_t1(s), p_gamma_t1(s)];
    p_analysis_t1_rpl_min(s) = find(p_analysis_t1_rpl(s,1:5)==min(p_analysis_t1_rpl(s,1:5)));
    
    [h, p, ci, stats] = ttest2(avg.RPL.delta_t2(lowIdx), avg.RPL.delta_t2(highIdx));
    p_delta_t2(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.theta_t2(lowIdx), avg.RPL.theta_t2(highIdx));
    p_theta_t2(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.alpha_t2(lowIdx), avg.RPL.alpha_t2(highIdx));
    p_alpha_t2(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.beta_t2(lowIdx), avg.RPL.beta_t2(highIdx));
    p_beta_t2(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.gamma_t2(lowIdx), avg.RPL.gamma_t2(highIdx));
    p_gamma_t2(s) = p;
    
    p_analysis_t2_rpl(s,1:5) = [p_delta_t2(s), p_theta_t2(s), p_alpha_t2(s), p_beta_t2(s), p_gamma_t2(s)];
    p_analysis_t2_rpl_min(s) = find(p_analysis_t2_rpl(s,1:5)==min(p_analysis_t2_rpl(s,1:5)));
    
    
    
    [h, p, ci, stats] = ttest2(avg.RPL.delta_o1(lowIdx), avg.RPL.delta_o1(highIdx));
    p_delta_o1(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.theta_o1(lowIdx), avg.RPL.theta_o1(highIdx));
    p_theta_o1(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.alpha_o1(lowIdx), avg.RPL.alpha_o1(highIdx));
    p_alpha_o1(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.beta_o1(lowIdx), avg.RPL.beta_o1(highIdx));
    p_beta_o1(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.gamma_o1(lowIdx), avg.RPL.gamma_o1(highIdx));
    p_gamma_o1(s) = p;
    
    p_analysis_o1_rpl(s,1:5) = [p_delta_o1(s), p_theta_o1(s), p_alpha_o1(s), p_beta_o1(s), p_gamma_o1(s)];
    p_analysis_o1_rpl_min(s) = find(p_analysis_o1_rpl(s,1:5)==min(p_analysis_o1_rpl(s,1:5)));
    
    [h, p, ci, stats] = ttest2(avg.RPL.delta_o2(lowIdx), avg.RPL.delta_o2(highIdx));
    p_delta_o2(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.theta_o2(lowIdx), avg.RPL.theta_o2(highIdx));
    p_theta_o2(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.alpha_o2(lowIdx), avg.RPL.alpha_o2(highIdx));
    p_alpha_o2(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.beta_o2(lowIdx), avg.RPL.beta_o2(highIdx));
    p_beta_o2(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.gamma_o2(lowIdx), avg.RPL.gamma_o2(highIdx));
    p_gamma_o2(s) = p;
    
    p_analysis_o2_rpl(s,1:5) = [p_delta_o2(s), p_theta_o2(s), p_alpha_o2(s), p_beta_o2(s), p_gamma_o2(s)];
    p_analysis_o2_rpl_min(s) = find(p_analysis_o2_rpl(s,1:5)==min(p_analysis_o2_rpl(s,1:5)));
    
    [h, p, ci, stats] = ttest2(avg.RPL.delta_o3(lowIdx), avg.RPL.delta_o3(highIdx));
    p_delta_o3(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.theta_o3(lowIdx), avg.RPL.theta_o3(highIdx));
    p_theta_o3(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.alpha_o3(lowIdx), avg.RPL.alpha_o3(highIdx));
    p_alpha_o3(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.beta_o3(lowIdx), avg.RPL.beta_o3(highIdx));
    p_beta_o3(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.gamma_o3(lowIdx), avg.RPL.gamma_o3(highIdx));
    p_gamma_o3(s) = p;
    
    p_analysis_o3_rpl(s,1:5) = [p_delta_o3(s), p_theta_o3(s), p_alpha_o3(s), p_beta_o3(s), p_gamma_o3(s)];
    p_analysis_o3_rpl_min(s) = find(p_analysis_o3_rpl(s,1:5)==min(p_analysis_o3_rpl(s,1:5)));
    
    Result.ttest.vertical_rpl(s,:) = [p_analysis_f1(s,:) p_analysis_f2(s,:) p_analysis_f3(s,:) p_analysis_c1(s,:) p_analysis_c2(s,:) p_analysis_c3(s,:) p_analysis_p1(s,:) p_analysis_p2(s,:) p_analysis_p3(s,:) p_analysis_t1(s,:) p_analysis_t2(s,:) p_analysis_o1(s,:) p_analysis_o2(s,:) p_analysis_o3(s,:)];
    
    
    
    
    
    
    
            % Regional Vertical T-TEST 분석
    [h, p, ci, stats] = ttest2(avg.z.Regional_f1(lowIdx,1), avg.z.Regional_f1(highIdx,1));
    p_delta_f1(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_f1(lowIdx,2), avg.z.Regional_f1(highIdx,2));
    p_theta_f1(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_f1(lowIdx,3), avg.z.Regional_f1(highIdx,3));
    p_alpha_f1(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_f1(lowIdx,4), avg.z.Regional_f1(highIdx,4));
    p_beta_f1(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_f1(lowIdx,5), avg.z.Regional_f1(highIdx,5));
    p_gamma_f1(s) = p;
    
    p_analysis_f1_z(s,1:5) = [p_delta_f1(s), p_theta_f1(s), p_alpha_f1(s), p_beta_f1(s), p_gamma_f1(s)];
    p_analysis_f1_z_min(s) = find(p_analysis_f1_z(s,1:5)==min(p_analysis_f1_z(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.z.Regional_f2(lowIdx,1), avg.z.Regional_f2(highIdx,1));
    p_delta_f2(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_f2(lowIdx,2), avg.z.Regional_f2(highIdx,2));
    p_theta_f2(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_f2(lowIdx,3), avg.z.Regional_f2(highIdx,3));
    p_alpha_f2(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_f2(lowIdx,4), avg.z.Regional_f2(highIdx,4));
    p_beta_f2(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_f2(lowIdx,5), avg.z.Regional_f2(highIdx,5));
    p_gamma_f2(s) = p;
    
    p_analysis_f2_z(s,1:5) = [p_delta_f2(s), p_theta_f2(s), p_alpha_f2(s), p_beta_f2(s), p_gamma_f2(s)];
    p_analysis_f2_z_min(s) = find(p_analysis_f2_z(s,1:5)==min(p_analysis_f2_z(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.z.Regional_f3(lowIdx,1), avg.z.Regional_f3(highIdx,1));
    p_delta_f3(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_f3(lowIdx,2), avg.z.Regional_f3(highIdx,2));
    p_theta_f3(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_f3(lowIdx,3), avg.z.Regional_f3(highIdx,3));
    p_alpha_f3(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_f3(lowIdx,4), avg.z.Regional_f3(highIdx,4));
    p_beta_f3(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_f3(lowIdx,5), avg.z.Regional_f3(highIdx,5));
    p_gamma_f3(s) = p;
    
    p_analysis_f3_z(s,1:5) = [p_delta_f3(s), p_theta_f3(s), p_alpha_f3(s), p_beta_f3(s), p_gamma_f3(s)];
    p_analysis_f3_z_min(s) = find(p_analysis_f3_z(s,1:5)==min(p_analysis_f3_z(s,1:5)));
    
    
    
    [h, p, ci, stats] = ttest2(avg.z.Regional_c1(lowIdx,1), avg.z.Regional_c1(highIdx,1));
    p_delta_c1(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_c1(lowIdx,2), avg.z.Regional_c1(highIdx,2));
    p_theta_c1(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_c1(lowIdx,3), avg.z.Regional_c1(highIdx,3));
    p_alpha_c1(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_c1(lowIdx,4), avg.z.Regional_c1(highIdx,4));
    p_beta_c1(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_c1(lowIdx,5), avg.z.Regional_c1(highIdx,5));
    p_gamma_c1(s) = p;
    
    p_analysis_c1_z(s,1:5) = [p_delta_c1(s), p_theta_c1(s), p_alpha_c1(s), p_beta_c1(s), p_gamma_c1(s)];
    p_analysis_c1_z_min(s) = find(p_analysis_c1_z(s,1:5)==min(p_analysis_c1_z(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.z.Regional_c2(lowIdx,1), avg.z.Regional_c2(highIdx,1));
    p_delta_c2(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_c2(lowIdx,2), avg.z.Regional_c2(highIdx,2));
    p_theta_c2(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_c2(lowIdx,3), avg.z.Regional_c2(highIdx,3));
    p_alpha_c2(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_c2(lowIdx,4), avg.z.Regional_c2(highIdx,4));
    p_beta_c2(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_c2(lowIdx,5), avg.z.Regional_c2(highIdx,5));
    p_gamma_c2(s) = p;
    
    p_analysis_c2_z(s,1:5) = [p_delta_c2(s), p_theta_c2(s), p_alpha_c2(s), p_beta_c2(s), p_gamma_c2(s)];
    p_analysis_c2_z_min(s) = find(p_analysis_c2_z(s,1:5)==min(p_analysis_c2_z(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.z.Regional_c3(lowIdx,1), avg.z.Regional_c3(highIdx,1));
    p_delta_c3(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_c3(lowIdx,2), avg.z.Regional_c3(highIdx,2));
    p_theta_c3(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_c3(lowIdx,3), avg.z.Regional_c3(highIdx,3));
    p_alpha_c3(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_c3(lowIdx,4), avg.z.Regional_c3(highIdx,4));
    p_beta_c3(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_c3(lowIdx,5), avg.z.Regional_c3(highIdx,5));
    p_gamma_c3(s) = p;
    
    p_analysis_c3_z(s,1:5) = [p_delta_c3(s), p_theta_c3(s), p_alpha_c3(s), p_beta_c3(s), p_gamma_c3(s)];
    p_analysis_c3_z_min(s) = find(p_analysis_c3_z(s,1:5)==min(p_analysis_c3_z(s,1:5)));
    
    

    
    
    [h, p, ci, stats] = ttest2(avg.z.Regional_p1(lowIdx,1), avg.z.Regional_p1(highIdx,1));
    p_delta_p1(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_p1(lowIdx,2), avg.z.Regional_p1(highIdx,2));
    p_theta_p1(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_p1(lowIdx,3), avg.z.Regional_p1(highIdx,3));
    p_alpha_p1(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_p1(lowIdx,4), avg.z.Regional_p1(highIdx,4));
    p_beta_p1(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_p1(lowIdx,5), avg.z.Regional_p1(highIdx,5));
    p_gamma_p1(s) = p;
    
    p_analysis_p1_z(s,1:5) = [p_delta_p1(s), p_theta_p1(s), p_alpha_p1(s), p_beta_p1(s), p_gamma_p1(s)];
    p_analysis_p1_z_min(s) = find(p_analysis_p1_z(s,1:5)==min(p_analysis_p1_z(s,1:5)));
    
    
    
    [h, p, ci, stats] = ttest2(avg.z.Regional_p2(lowIdx,1), avg.z.Regional_p2(highIdx,1));
    p_delta_p2(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_p2(lowIdx,2), avg.z.Regional_p2(highIdx,2));
    p_theta_p2(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_p2(lowIdx,3), avg.z.Regional_p2(highIdx,3));
    p_alpha_p2(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_p2(lowIdx,4), avg.z.Regional_p2(highIdx,4));
    p_beta_p2(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_p2(lowIdx,5), avg.z.Regional_p2(highIdx,5));
    p_gamma_p2(s) = p;
    
    p_analysis_p2_z(s,1:5) = [p_delta_p2(s), p_theta_p2(s), p_alpha_p2(s), p_beta_p2(s), p_gamma_p2(s)];
    p_analysis_p2_z_min(s) = find(p_analysis_p2_z(s,1:5)==min(p_analysis_p2_z(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.z.Regional_p3(lowIdx,1), avg.z.Regional_p3(highIdx,1));
    p_delta_p3(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_p3(lowIdx,2), avg.z.Regional_p3(highIdx,2));
    p_theta_p3(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_p3(lowIdx,3), avg.z.Regional_p3(highIdx,3));
    p_alpha_p3(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_p3(lowIdx,4), avg.z.Regional_p3(highIdx,4));
    p_beta_p3(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_p3(lowIdx,5), avg.z.Regional_p3(highIdx,5));
    p_gamma_p3(s) = p;
    
    p_analysis_p3_z(s,1:5) = [p_delta_p3(s), p_theta_p3(s), p_alpha_p3(s), p_beta_p3(s), p_gamma_p3(s)];
    p_analysis_p3_z_min(s) = find(p_analysis_p3_z(s,1:5)==min(p_analysis_p3_z(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.z.Regional_t1(lowIdx,1), avg.z.Regional_t1(highIdx,1));
    p_delta_t1(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_t1(lowIdx,2), avg.z.Regional_t1(highIdx,2));
    p_theta_t1(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_t1(lowIdx,3), avg.z.Regional_t1(highIdx,3));
    p_alpha_t1(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_t1(lowIdx,4), avg.z.Regional_t1(highIdx,4));
    p_beta_t1(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_t1(lowIdx,5), avg.z.Regional_t1(highIdx,5));
    p_gamma_t1(s) = p;
    
    p_analysis_t1_z(s,1:5) = [p_delta_t1(s), p_theta_t1(s), p_alpha_t1(s), p_beta_t1(s), p_gamma_t1(s)];
    p_analysis_t1_z_min(s) = find(p_analysis_t1_z(s,1:5)==min(p_analysis_t1_z(s,1:5)));
    
    [h, p, ci, stats] = ttest2(avg.z.Regional_t2(lowIdx,1), avg.z.Regional_t2(highIdx,1));
    p_delta_t2(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_t2(lowIdx,2), avg.z.Regional_t2(highIdx,2));
    p_theta_t2(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_t2(lowIdx,3), avg.z.Regional_t2(highIdx,3));
    p_alpha_t2(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_t2(lowIdx,4), avg.z.Regional_t2(highIdx,4));
    p_beta_t2(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_t2(lowIdx,5), avg.z.Regional_t2(highIdx,5));
    p_gamma_t2(s) = p;
    
    p_analysis_t2_z(s,1:5) = [p_delta_t2(s), p_theta_t2(s), p_alpha_t2(s), p_beta_t2(s), p_gamma_t2(s)];
    p_analysis_t2_z_min(s) = find(p_analysis_t2_z(s,1:5)==min(p_analysis_t2_z(s,1:5)));
    
    
    
    [h, p, ci, stats] = ttest2(avg.z.Regional_o1(lowIdx,1), avg.z.Regional_o1(highIdx,1));
    p_delta_o1(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_o1(lowIdx,2), avg.z.Regional_o1(highIdx,2));
    p_theta_o1(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_o1(lowIdx,3), avg.z.Regional_o1(highIdx,3));
    p_alpha_o1(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_o1(lowIdx,4), avg.z.Regional_o1(highIdx,4));
    p_beta_o1(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_o1(lowIdx,5), avg.z.Regional_o1(highIdx,5));
    p_gamma_o1(s) = p;
    
    p_analysis_o1_z(s,1:5) = [p_delta_o1(s), p_theta_o1(s), p_alpha_o1(s), p_beta_o1(s), p_gamma_o1(s)];
    p_analysis_o1_z_min(s) = find(p_analysis_o1_z(s,1:5)==min(p_analysis_o1_z(s,1:5)));
    
    [h, p, ci, stats] = ttest2(avg.z.Regional_o2(lowIdx,1), avg.z.Regional_o2(highIdx,1));
    p_delta_o2(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_o2(lowIdx,2), avg.z.Regional_o2(highIdx,2));
    p_theta_o2(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_o2(lowIdx,3), avg.z.Regional_o2(highIdx,3));
    p_alpha_o2(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_o2(lowIdx,4), avg.z.Regional_o2(highIdx,4));
    p_beta_o2(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_o2(lowIdx,5), avg.z.Regional_o2(highIdx,5));
    p_gamma_o2(s) = p;
    
    p_analysis_o2_z(s,1:5) = [p_delta_o2(s), p_theta_o2(s), p_alpha_o2(s), p_beta_o2(s), p_gamma_o2(s)];
    p_analysis_o2_z_min(s) = find(p_analysis_o2_z(s,1:5)==min(p_analysis_o2_z(s,1:5)));
    
    [h, p, ci, stats] = ttest2(avg.z.Regional_o3(lowIdx,1), avg.z.Regional_o3(highIdx,1));
    p_delta_o3(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_o3(lowIdx,2), avg.z.Regional_o3(highIdx,2));
    p_theta_o3(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_o3(lowIdx,3), avg.z.Regional_o3(highIdx,3));
    p_alpha_o3(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_o3(lowIdx,4), avg.z.Regional_o3(highIdx,4));
    p_beta_o3(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_o3(lowIdx,5), avg.z.Regional_o3(highIdx,5));
    p_gamma_o3(s) = p;
    
    p_analysis_o3_z(s,1:5) = [p_delta_o3(s), p_theta_o3(s), p_alpha_o3(s), p_beta_o3(s), p_gamma_o3(s)];
    p_analysis_o3_z_min(s) = find(p_analysis_o3_z(s,1:5)==min(p_analysis_o3_z(s,1:5)));
    
    Result.ttest.vertical_z(s,:) = [p_analysis_f1(s,:) p_analysis_f2(s,:) p_analysis_f3(s,:) p_analysis_c1(s,:) p_analysis_c2(s,:) p_analysis_c3(s,:) p_analysis_p1(s,:) p_analysis_p2(s,:) p_analysis_p3(s,:) p_analysis_t1(s,:) p_analysis_t2(s,:) p_analysis_o1(s,:) p_analysis_o2(s,:) p_analysis_o3(s,:)];
    
    
    
    
    
    
    
    
    
    
    %% Feature Selection
    feature_ALL = avg.PSD;
    feature_PSD = avg.PSD(:,find(p_analysis(s,:)<0.05));
    feature_Regional = [avg.Regional_f(:,find(p_analysis_f(s,:)<0.05)),avg.Regional_c(:,find(p_analysis_c(s,:)<0.05)),avg.Regional_p(:,find(p_analysis_p(s,:)<0.05)),avg.Regional_t(:,find(p_analysis_t(s,:)<0.05)),avg.Regional_o(:,find(p_analysis_o(s,:)<0.05))];
    feature_Vertical = [avg.Regional_f1(:,find(p_analysis_f1(s,:)<0.05)),avg.Regional_f2(:,find(p_analysis_f2(s,:)<0.05)),avg.Regional_f3(:,find(p_analysis_f3(s,:)<0.05)),avg.Regional_c1(:,find(p_analysis_c1(s,:)<0.05)),avg.Regional_c2(:,find(p_analysis_c2(s,:)<0.05)),avg.Regional_c3(:,find(p_analysis_c3(s,:)<0.05)),avg.Regional_p1(:,find(p_analysis_p1(s,:)<0.05)),avg.Regional_p2(:,find(p_analysis_p2(s,:)<0.05)),avg.Regional_p3(:,find(p_analysis_p3(s,:)<0.05)),avg.Regional_t1(:,find(p_analysis_t1(s,:)<0.05)),avg.Regional_t2(:,find(p_analysis_t2(s,:)<0.05)),avg.Regional_o1(:,find(p_analysis_o1(s,:)<0.05)),avg.Regional_o2(:,find(p_analysis_o2(s,:)<0.05)),avg.Regional_o3(:,find(p_analysis_o3(s,:)<0.05))];
    
    feature_ALL_RPL = avg.RPL.PSD;
    feature_PSD_RPL = avg.RPL.PSD(:,find(p_analysis_RPL(s,:)<0.05));
    feature_Regional_RPL = [avg.RPL.Regional_f(:,find(p_analysis_RPL_f(s,:)<0.05)),avg.RPL.Regional_c(:,find(p_analysis_RPL_c(s,:)<0.05)),avg.RPL.Regional_p(:,find(p_analysis_RPL_p(s,:)<0.05)),avg.RPL.Regional_t(:,find(p_analysis_RPL_t(s,:)<0.05)),avg.RPL.Regional_o(:,find(p_analysis_RPL_o(s,:)<0.05))];
    feature_Vertical_RPL = [avg.RPL.Regional_f1(:,find(p_analysis_f1_rpl(s,:)<0.05)),avg.RPL.Regional_f2(:,find(p_analysis_f2_rpl(s,:)<0.05)),avg.RPL.Regional_f3(:,find(p_analysis_f3_rpl(s,:)<0.05)),avg.RPL.Regional_c1(:,find(p_analysis_c1_rpl(s,:)<0.05)),avg.RPL.Regional_c2(:,find(p_analysis_c2_rpl(s,:)<0.05)),avg.RPL.Regional_c3(:,find(p_analysis_c3_rpl(s,:)<0.05)),avg.RPL.Regional_p1(:,find(p_analysis_p1_rpl(s,:)<0.05)),avg.RPL.Regional_p2(:,find(p_analysis_p2_rpl(s,:)<0.05)),avg.RPL.Regional_p3(:,find(p_analysis_p3_rpl(s,:)<0.05)),avg.RPL.Regional_t1(:,find(p_analysis_t1_rpl(s,:)<0.05)),avg.RPL.Regional_t2(:,find(p_analysis_t2_rpl(s,:)<0.05)),avg.RPL.Regional_o1(:,find(p_analysis_o1_rpl(s,:)<0.05)),avg.RPL.Regional_o2(:,find(p_analysis_o2_rpl(s,:)<0.05)),avg.RPL.Regional_o3(:,find(p_analysis_o3_rpl(s,:)<0.05))];

    feature_ALL_z = avg.z.PSD;
    feature_PSD_z = avg.z.PSD(:,find(p_analysis_z(s,:)<0.05));
    feature_Regional_z = [avg.z.Regional_f(:,find(p_analysis_z_f(s,:)<0.05)),avg.z.Regional_c(:,find(p_analysis_z_c(s,:)<0.05)),avg.z.Regional_p(:,find(p_analysis_z_p(s,:)<0.05)),avg.z.Regional_t(:,find(p_analysis_z_t(s,:)<0.05)),avg.z.Regional_o(:,find(p_analysis_z_o(s,:)<0.05))];
    feature_Vertical_z = [avg.z.Regional_f1(:,find(p_analysis_f1_z(s,:)<0.05)),avg.z.Regional_f2(:,find(p_analysis_f2_z(s,:)<0.05)),avg.z.Regional_f3(:,find(p_analysis_f3_z(s,:)<0.05)),avg.z.Regional_c1(:,find(p_analysis_c1_z(s,:)<0.05)),avg.z.Regional_c2(:,find(p_analysis_c2_z(s,:)<0.05)),avg.z.Regional_c3(:,find(p_analysis_c3_z(s,:)<0.05)),avg.z.Regional_p1(:,find(p_analysis_p1_z(s,:)<0.05)),avg.z.Regional_p2(:,find(p_analysis_p2_z(s,:)<0.05)),avg.z.Regional_p3(:,find(p_analysis_p3_z(s,:)<0.05)),avg.z.Regional_t1(:,find(p_analysis_t1_z(s,:)<0.05)),avg.z.Regional_t2(:,find(p_analysis_t2_z(s,:)<0.05)),avg.z.Regional_o1(:,find(p_analysis_o1_z(s,:)<0.05)),avg.z.Regional_o2(:,find(p_analysis_o2_z(s,:)<0.05)),avg.z.Regional_o3(:,find(p_analysis_o3_z(s,:)<0.05))];
    
    %% Classification
    [Accuracy_ALL AUC_ALL] = LDA_4CV(feature_ALL, highIdx, lowIdx, s);
    [Accuracy_PSD AUC_PSD] = LDA_4CV(feature_PSD, highIdx, lowIdx, s);
    [Accuracy_Regional AUC_Regional] = LDA_4CV(feature_Regional, highIdx, lowIdx, s);
    [Accuracy_Vertical AUC_Vertical] = LDA_4CV(feature_Vertical, highIdx, lowIdx, s);
    
    [Accuracy_ALL_RPL AUC_ALL_RPL] = LDA_4CV(feature_ALL_RPL, highIdx, lowIdx, s);
    [Accuracy_PSD_RPL AUC_PSD_RPL] = LDA_4CV(feature_PSD_RPL, highIdx, lowIdx, s);
    [Accuracy_Regional_RPL AUC_Regional_RPL] = LDA_4CV(feature_Regional_RPL, highIdx, lowIdx, s);
    [Accuracy_Vertical_RPL AUC_Vertical_RPL] = LDA_4CV(feature_Vertical_RPL, highIdx, lowIdx, s);
    
    [Accuracy_ALL_z AUC_ALL_z] = LDA_4CV(feature_ALL_z, highIdx, lowIdx, s);
    [Accuracy_PSD_z AUC_PSD_z] = LDA_4CV(feature_PSD_z, highIdx, lowIdx, s);
    [Accuracy_Regional_z AUC_Regional_z] = LDA_4CV(feature_Regional_z, highIdx, lowIdx, s);
    [Accuracy_Vertical_z AUC_Vertical_z] = LDA_4CV(feature_Vertical_z, highIdx, lowIdx, s);
    
    Result.Fatigue_Accuracy(s,:) = [Accuracy_ALL Accuracy_PSD Accuracy_Regional Accuracy_Vertical];
    Result.Fatigue_AUC(s,:) = [AUC_ALL AUC_PSD AUC_Regional AUC_Vertical];
    
    Result.Fatigue_Accuracy_RPL(s,:) = [Accuracy_ALL_RPL Accuracy_PSD_RPL Accuracy_Regional_RPL Accuracy_Vertical_RPL];
    Result.Fatigue_AUC_RPL(s,:) = [AUC_ALL_RPL AUC_PSD_RPL AUC_Regional_RPL AUC_Vertical_RPL];
    
    Result.Fatigue_Accuracy_z(s,:) = [Accuracy_ALL_z Accuracy_PSD_z Accuracy_Regional_z Accuracy_Vertical_z];
    Result.Fatigue_AUC_z(s,:) = [AUC_ALL_z AUC_PSD_z AUC_Regional_z AUC_Vertical_z];
    
%     %% Regression
%     [Prediction_ALL, RMSE_ALL, Error_Rate_ALL] = MLR_4CV(feature_ALL, kss, s);
%     [Prediction_PSD, RMSE_PSD, Error_Rate_PSD] = MLR_4CV(feature_PSD, kss, s);
%     [Prediction_Regional, RMSE_Regional, Error_Rate_Regional] = MLR_4CV(feature_Regional, kss, s);
%     [Prediction_Vertical, RMSE_Vertical, Error_Rate_Vertical] = MLR_4CV(feature_Vertical, kss, s);
%     
%     [Prediction_ALL_RPL, RMSE_ALL_RPL, Error_Rate_ALL_RPL] = MLR_4CV(feature_ALL_RPL, kss, s);
%     [Prediction_PSD_RPL, RMSE_PSD_RPL, Error_Rate_PSD_RPL] = MLR_4CV(feature_PSD_RPL, kss, s);
%     [Prediction_Regional_RPL, RMSE_Regional_RPL, Error_Rate_Regional_RPL] = MLR_4CV(feature_Regional_RPL, kss, s);
%     [Prediction_Vertical_RPL, RMSE_Vertical_RPL, Error_Rate_Vertical_RPL] = MLR_4CV(feature_Vertical_RPL, kss, s);
%     
%     [Prediction_ALL_z, RMSE_ALL_z, Error_Rate_ALL_z] = MLR_4CV(feature_ALL_z, kss, s);
%     [Prediction_PSD_z, RMSE_PSD_z, Error_Rate_PSD_z] = MLR_4CV(feature_PSD_z, kss, s);
%     [Prediction_Regional_z, RMSE_Regional_z, Error_Rate_Regional_z] = MLR_4CV(feature_Regional_z, kss, s);
%     [Prediction_Vertical_z, RMSE_Vertical_z, Error_Rate_Vertical_z] = MLR_4CV(feature_Vertical_z, kss, s);
%  
%     Result.Fatigue_Prediction(s,:) = [Prediction_ALL Prediction_PSD Prediction_Regional Prediction_Vertical];
%     Result.Fatigue_RMSE(s,:) = [RMSE_ALL RMSE_PSD RMSE_Regional RMSE_Vertical];   
%     Result.Fatigue_Error(s,:) = [Error_Rate_ALL Error_Rate_PSD Error_Rate_Regional Error_Rate_Vertical];   
%     
%     Result.Fatigue_Prediction_RPL(s,:) = [Prediction_ALL_RPL Prediction_PSD_RPL Prediction_Regional_RPL Prediction_Vertical_RPL];
%     Result.Fatigue_RMSE_RPL(s,:) = [RMSE_ALL_RPL RMSE_PSD_RPL RMSE_Regional_RPL RMSE_Vertical_RPL];   
%     Result.Fatigue_Error_RPL(s,:) = [Error_Rate_ALL_RPL Error_Rate_PSD_RPL Error_Rate_Regional_RPL Error_Rate_Vertical_RPL];   
%     
%     Result.Fatigue_Prediction_z(s,:) = [Prediction_ALL_z Prediction_PSD_z Prediction_Regional_z Prediction_Vertical_z];
%     Result.Fatigue_RMSE_z(s,:) = [RMSE_ALL_z RMSE_PSD_z RMSE_Regional_z RMSE_Vertical_z];   
%     Result.Fatigue_Error_z(s,:) = [Error_Rate_ALL_z Error_Rate_PSD_z Error_Rate_Regional_z Error_Rate_Vertical_z];   

    %% Save
    Result.Fatigue_Accuracy_1(s,:) = Result.Fatigue_Accuracy(s,:);
    Result.Fatigue_AUC_1(s,:) = Result.Fatigue_AUC(s,:);  
%     Result.Fatigue_Prediction_1(s,:) = Result.Fatigue_Prediction(s,:);
%     Result.Fatigue_RMSE_1(s,:) = Result.Fatigue_RMSE(s,:);
%     Result.Fatigue_Error_1(s,:) = Result.Fatigue_Error(s,:);
    
    Result.Fatigue_Accuracy_1_RPL(s,:) = Result.Fatigue_Accuracy_RPL(s,:);
    Result.Fatigue_AUC_1_RPL(s,:) = Result.Fatigue_AUC_RPL(s,:);  
%     Result.Fatigue_Prediction_1_RPL(s,:) = Result.Fatigue_Prediction_RPL(s,:);
%     Result.Fatigue_RMSE_1_RPL(s,:) = Result.Fatigue_RMSE_RPL(s,:);
%     Result.Fatigue_Error_1_RPL(s,:) = Result.Fatigue_Error_RPL(s,:);
    
    Result.Fatigue_Accuracy_1_z(s,:) = Result.Fatigue_Accuracy_z(s,:);
    Result.Fatigue_AUC_1_z(s,:) = Result.Fatigue_AUC_z(s,:);  
%     Result.Fatigue_Prediction_1_z(s,:) = Result.Fatigue_Prediction_z(s,:);
%     Result.Fatigue_RMSE_1_z(s,:) = Result.Fatigue_RMSE_z(s,:);
%     Result.Fatigue_Error_1_z(s,:) = Result.Fatigue_Error_z(s,:);

    
    
    
    
        %% Segmentation of bio-signals with 10 seconds length of epoch
    epoch = segmentationFatigue_10(cnt, mrk, interKSS);
    kssIdx = find(strcmp(epoch.mClab, 'KSS'));
    kss = squeeze(mean(epoch.misc(:, kssIdx, :), 1));
    
    %% Regional Channel Selection
    channel_f=epoch.x(:,[1:5,31:36],:);
    channel_c=epoch.x(:,[6:9,11:13,16:19,39:40,43:46,48:50],:);
    channel_p=epoch.x(:,[21:25,52:55],:);
    channel_t=epoch.x(:,[10,14:15,20,37:38,41:42,47,51],:);
    channel_o=epoch.x(:,[26:30,56:60],:);
    
    channel_f1=epoch.x(:,[1,3,31,33],:);
    channel_f2=epoch.x(:,[2,5,32,36],:);
    channel_f3=epoch.x(:,[4,34:35],:);
    channel_c1=epoch.x(:,[6:7,11,16,17,39,43,44,48],:);
    channel_c2=epoch.x(:,[8:9,13,18,19,40,45,46,50],:);
    channel_c3=epoch.x(:,[12,49],:);
    channel_p1=epoch.x(:,[21:22,52,53],:);
    channel_p2=epoch.x(:,[24:25,54,55],:);
    channel_p3=epoch.x(:,[23],:);
    channel_t1=epoch.x(:,[10,15,37,38,47],:);
    channel_t2=epoch.x(:,[14,20,41,42,51],:);
    channel_o1=epoch.x(:,[26,27,56,57],:);
    channel_o2=epoch.x(:,[29,30,59,60],:);
    channel_o3=epoch.x(:,[28,58],:);
    
    %% Feature Extraction - Power spectrum analysis for EEG signals
    
    % PSD Feature Extraction
    [cca.delta, avg.delta] = Power_spectrum(epoch.x, [0.5 4], cnt.fs);
    [cca.theta, avg.theta] = Power_spectrum(epoch.x, [4 8], cnt.fs);
    [cca.alpha, avg.alpha] = Power_spectrum(epoch.x, [8 13], cnt.fs);
    [cca.beta, avg.beta] = Power_spectrum(epoch.x, [13 30], cnt.fs);
    [cca.gamma, avg.gamma] = Power_spectrum(epoch.x, [30 40], cnt.fs);
    avg.delta = avg.delta'; avg.theta=avg.theta'; avg.alpha=avg.alpha'; avg.beta = avg.beta'; avg.gamma = avg.gamma';
    
    cca.PSD = [cca.delta cca.theta cca.alpha cca.beta cca.gamma];
    avg.PSD = [avg.delta avg.theta avg.alpha avg.beta avg.gamma];

    % RPL Feature Extraction
    cca.RPL.delta = cca.delta ./ (cca.delta + cca.theta + cca.alpha + cca.beta + cca.gamma) ;
    avg.RPL.delta = avg.delta ./ sum(avg.PSD,2) ;
    cca.RPL.theta = cca.theta ./ (cca.delta + cca.theta + cca.alpha + cca.beta + cca.gamma) ;
    avg.RPL.theta = avg.theta ./ sum(avg.PSD,2) ;
    cca.RPL.alpha = cca.alpha ./ (cca.delta + cca.theta + cca.alpha + cca.beta + cca.gamma) ;
    avg.RPL.alpha = avg.alpha ./ sum(avg.PSD,2) ;
    cca.RPL.beta = cca.beta ./ (cca.delta + cca.theta + cca.alpha + cca.beta + cca.gamma) ;
    avg.RPL.beta = avg.beta ./ sum(avg.PSD,2) ;
    cca.RPL.gamma = cca.gamma ./ (cca.delta + cca.theta + cca.alpha + cca.beta + cca.gamma) ;
    avg.RPL.gamma = avg.gamma ./ sum(avg.PSD,2) ;
    
    cca.RPL.PSD = [cca.RPL.delta, cca.RPL.theta, cca.RPL.alpha, cca.RPL.beta, cca.RPL.gamma];
    avg.RPL.PSD = [avg.RPL.delta, avg.RPL.theta, avg.RPL.alpha, avg.RPL.beta, avg.RPL.gamma];
    
    % Z-SCORE Feature Extraction
    cca.z.PSD = [zscore(cca.delta) zscore(cca.theta) zscore(cca.alpha) zscore(cca.beta) zscore(cca.gamma)];
    avg.z.PSD = [zscore(avg.delta) zscore(avg.theta) zscore(avg.alpha) zscore(avg.beta) zscore(avg.gamma)];
    
    
    % Regional Feature Extraction
    [cca.delta_f,avg.delta_f] = Power_spectrum(channel_f, [0.5 4], cnt.fs);
    [cca.theta_f,avg.theta_f] = Power_spectrum(channel_f, [4 8], cnt.fs);
    [cca.alpha_f,avg.alpha_f] = Power_spectrum(channel_f, [8 13], cnt.fs);
    [cca.beta_f,avg.beta_f] = Power_spectrum(channel_f, [13 30], cnt.fs);
    [cca.gamma_f,avg.gamma_f] = Power_spectrum(channel_f, [30 40], cnt.fs);
    avg.delta_f = avg.delta_f'; avg.theta_f=avg.theta_f'; avg.alpha_f=avg.alpha_f'; avg.beta_f = avg.beta_f'; avg.gamma_f = avg.gamma_f';
    
    cca.Regional_f = [cca.delta_f cca.theta_f cca.alpha_f cca.beta_f cca.gamma_f];
    avg.Regional_f = [avg.delta_f avg.theta_f avg.alpha_f avg.beta_f avg.gamma_f];
    
    cca.RPL.delta_f = cca.delta_f ./ (cca.delta_f + cca.theta_f + cca.alpha_f + cca.beta_f + cca.gamma_f) ;
    avg.RPL.delta_f = avg.delta_f ./ sum(avg.Regional_f,2) ;
    cca.RPL.theta_f = cca.theta_f ./ (cca.delta_f + cca.theta_f + cca.alpha_f + cca.beta_f + cca.gamma_f) ;
    avg.RPL.theta_f = avg.theta_f ./ sum(avg.Regional_f,2) ;
    cca.RPL.alpha_f = cca.alpha_f ./ (cca.delta_f + cca.theta_f + cca.alpha_f + cca.beta_f + cca.gamma_f) ;
    avg.RPL.alpha_f = avg.alpha_f ./ sum(avg.Regional_f,2) ;
    cca.RPL.beta_f = cca.beta_f ./ (cca.delta_f + cca.theta_f + cca.alpha_f + cca.beta_f + cca.gamma_f) ;
    avg.RPL.beta_f = avg.beta_f ./ sum(avg.Regional_f,2) ;
    cca.RPL.gamma_f = cca.gamma_f ./ (cca.delta_f + cca.theta_f + cca.alpha_f + cca.beta_f + cca.gamma_f) ;
    avg.RPL.gamma_f = avg.gamma_f ./ sum(avg.Regional_f,2) ;    
    
    cca.RPL.Regional_f = [cca.RPL.delta_f cca.RPL.theta_f cca.RPL.alpha_f cca.RPL.beta_f cca.RPL.gamma_f]; 
    avg.RPL.Regional_f = [avg.RPL.delta_f avg.RPL.theta_f avg.RPL.alpha_f avg.RPL.beta_f avg.RPL.gamma_f];
    
    cca.z.Regional_f = [zscore(cca.delta_f) zscore(cca.theta_f) zscore(cca.alpha_f) zscore(cca.beta_f) zscore(cca.gamma_f)];
    avg.z.Regional_f = [zscore(avg.delta_f) zscore(avg.theta_f) zscore(avg.alpha_f) zscore(avg.beta_f) zscore(avg.gamma_f)];   
    
    
    
    [cca.delta_c,avg.delta_c] = Power_spectrum(channel_c, [0.5 4], cnt.fs);
    [cca.theta_c,avg.theta_c] = Power_spectrum(channel_c, [4 8], cnt.fs);
    [cca.alpha_c,avg.alpha_c] = Power_spectrum(channel_c, [8 13], cnt.fs);
    [cca.beta_c,avg.beta_c] = Power_spectrum(channel_c, [13 30], cnt.fs);
    [cca.gamma_c,avg.gamma_c] = Power_spectrum(channel_c, [30 40], cnt.fs);
    avg.delta_c = avg.delta_c'; avg.theta_c=avg.theta_c'; avg.alpha_c=avg.alpha_c'; avg.beta_c = avg.beta_c'; avg.gamma_c = avg.gamma_c';
    
    cca.Regional_c = [cca.delta_c cca.theta_c cca.alpha_c cca.beta_c cca.gamma_c];
    avg.Regional_c = [avg.delta_c avg.theta_c avg.alpha_c avg.beta_c avg.gamma_c];
    
    cca.RPL.delta_c = cca.delta_c ./ (cca.delta_c + cca.theta_c + cca.alpha_c + cca.beta_c + cca.gamma_c) ;
    avg.RPL.delta_c = avg.delta_c ./ sum(avg.Regional_c,2) ;
    cca.RPL.theta_c = cca.theta_c ./ (cca.delta_c + cca.theta_c + cca.alpha_c + cca.beta_c + cca.gamma_c) ;
    avg.RPL.theta_c = avg.theta_c ./ sum(avg.Regional_c,2) ;
    cca.RPL.alpha_c = cca.alpha_c ./ (cca.delta_c + cca.theta_c + cca.alpha_c + cca.beta_c + cca.gamma_c) ;
    avg.RPL.alpha_c = avg.alpha_c ./ sum(avg.Regional_c,2) ;
    cca.RPL.beta_c = cca.beta_c ./ (cca.delta_c + cca.theta_c + cca.alpha_c + cca.beta_c + cca.gamma_c) ;
    avg.RPL.beta_c = avg.beta_c ./ sum(avg.Regional_c,2) ;
    cca.RPL.gamma_c = cca.gamma_c ./ (cca.delta_c + cca.theta_c + cca.alpha_c + cca.beta_c + cca.gamma_c) ;
    avg.RPL.gamma_c = avg.gamma_c ./ sum(avg.Regional_c,2) ;     
    
    cca.RPL.Regional_c = [cca.RPL.delta_c cca.RPL.theta_c cca.RPL.alpha_c cca.RPL.beta_c cca.RPL.gamma_c]; 
    avg.RPL.Regional_c = [avg.RPL.delta_c avg.RPL.theta_c avg.RPL.alpha_c avg.RPL.beta_c avg.RPL.gamma_c];
    
    cca.z.Regional_c = [zscore(cca.delta_c) zscore(cca.theta_c) zscore(cca.alpha_c) zscore(cca.beta_c) zscore(cca.gamma_c)];
    avg.z.Regional_c = [zscore(avg.delta_c) zscore(avg.theta_c) zscore(avg.alpha_c) zscore(avg.beta_c) zscore(avg.gamma_c)];   
    
    
    [cca.delta_p,avg.delta_p] = Power_spectrum(channel_p, [0.5 4], cnt.fs);
    [cca.theta_p,avg.theta_p] = Power_spectrum(channel_p, [4 8], cnt.fs);
    [cca.alpha_p,avg.alpha_p] = Power_spectrum(channel_p, [8 13], cnt.fs);
    [cca.beta_p,avg.beta_p] = Power_spectrum(channel_p, [13 30], cnt.fs);
    [cca.gamma_p,avg.gamma_p] = Power_spectrum(channel_p, [30 40], cnt.fs);
    avg.delta_p = avg.delta_p'; avg.theta_p=avg.theta_p'; avg.alpha_p=avg.alpha_p'; avg.beta_p = avg.beta_p'; avg.gamma_p = avg.gamma_p';
    
    cca.Regional_p = [cca.delta_p cca.theta_p cca.alpha_p cca.beta_p cca.gamma_p];
    avg.Regional_p = [avg.delta_p avg.theta_p avg.alpha_p avg.beta_p avg.gamma_p];

    cca.RPL.delta_p = cca.delta_p ./ (cca.delta_p + cca.theta_p + cca.alpha_p + cca.beta_p + cca.gamma_p) ;
    avg.RPL.delta_p = avg.delta_p ./ sum(avg.Regional_p,2) ;
    cca.RPL.theta_p = cca.theta_p ./ (cca.delta_p + cca.theta_p + cca.alpha_p + cca.beta_p + cca.gamma_p) ;
    avg.RPL.theta_p = avg.theta_p ./ sum(avg.Regional_p,2) ;
    cca.RPL.alpha_p = cca.alpha_p ./ (cca.delta_p + cca.theta_p + cca.alpha_p + cca.beta_p + cca.gamma_p) ;
    avg.RPL.alpha_p = avg.alpha_p ./ sum(avg.Regional_p,2) ;
    cca.RPL.beta_p = cca.beta_p ./ (cca.delta_p + cca.theta_p + cca.alpha_p + cca.beta_p + cca.gamma_p) ;
    avg.RPL.beta_p = avg.beta_p ./ sum(avg.Regional_p,2) ;
    cca.RPL.gamma_p = cca.gamma_p ./ (cca.delta_p + cca.theta_p + cca.alpha_p + cca.beta_p + cca.gamma_p) ;
    avg.RPL.gamma_p = avg.gamma_p ./ sum(avg.Regional_p,2) ;     
    
    cca.RPL.Regional_p = [cca.RPL.delta_p cca.RPL.theta_p cca.RPL.alpha_p cca.RPL.beta_p cca.RPL.gamma_p]; 
    avg.RPL.Regional_p = [avg.RPL.delta_p avg.RPL.theta_p avg.RPL.alpha_p avg.RPL.beta_p avg.RPL.gamma_p];
    
    cca.z.Regional_p = [zscore(cca.delta_p) zscore(cca.theta_p) zscore(cca.alpha_p) zscore(cca.beta_p) zscore(cca.gamma_p)];
    avg.z.Regional_p = [zscore(avg.delta_p) zscore(avg.theta_p) zscore(avg.alpha_p) zscore(avg.beta_p) zscore(avg.gamma_p)];   
    
    
    [cca.delta_t,avg.delta_t] = Power_spectrum(channel_t, [0.5 4], cnt.fs);
    [cca.theta_t,avg.theta_t] = Power_spectrum(channel_t, [4 8], cnt.fs);
    [cca.alpha_t,avg.alpha_t] = Power_spectrum(channel_t, [8 13], cnt.fs);
    [cca.beta_t,avg.beta_t] = Power_spectrum(channel_t, [13 30], cnt.fs);
    [cca.gamma_t,avg.gamma_t] = Power_spectrum(channel_t, [30 40], cnt.fs);
    avg.delta_t = avg.delta_t'; avg.theta_t=avg.theta_t'; avg.alpha_t=avg.alpha_t'; avg.beta_t = avg.beta_t'; avg.gamma_t = avg.gamma_t';
    
    cca.Regional_t = [cca.delta_t cca.theta_t cca.alpha_t cca.beta_t cca.gamma_t];
    avg.Regional_t = [avg.delta_t avg.theta_t avg.alpha_t avg.beta_t avg.gamma_t];

    cca.RPL.delta_t = cca.delta_t ./ (cca.delta_t + cca.theta_t + cca.alpha_t + cca.beta_t + cca.gamma_t) ;
    avg.RPL.delta_t = avg.delta_t ./ sum(avg.Regional_t,2) ;
    cca.RPL.theta_t = cca.theta_t ./ (cca.delta_t + cca.theta_t + cca.alpha_t + cca.beta_t + cca.gamma_t) ;
    avg.RPL.theta_t = avg.theta_t ./ sum(avg.Regional_t,2) ;
    cca.RPL.alpha_t = cca.alpha_t ./ (cca.delta_t + cca.theta_t + cca.alpha_t + cca.beta_t + cca.gamma_t) ;
    avg.RPL.alpha_t = avg.alpha_t ./ sum(avg.Regional_t,2) ;
    cca.RPL.beta_t = cca.beta_t ./ (cca.delta_t + cca.theta_t + cca.alpha_t + cca.beta_t + cca.gamma_t) ;
    avg.RPL.beta_t = avg.beta_t ./ sum(avg.Regional_t,2) ;
    cca.RPL.gamma_t = cca.gamma_t ./ (cca.delta_t + cca.theta_t + cca.alpha_t + cca.beta_t + cca.gamma_t) ;
    avg.RPL.gamma_t = avg.gamma_t ./ sum(avg.Regional_t,2) ;     
    
    cca.RPL.Regional_t = [cca.RPL.delta_t cca.RPL.theta_t cca.RPL.alpha_t cca.RPL.beta_t cca.RPL.gamma_t]; 
    avg.RPL.Regional_t = [avg.RPL.delta_t avg.RPL.theta_t avg.RPL.alpha_t avg.RPL.beta_t avg.RPL.gamma_t];
    
    cca.z.Regional_t = [zscore(cca.delta_t) zscore(cca.theta_t) zscore(cca.alpha_t) zscore(cca.beta_t) zscore(cca.gamma_t)];
    avg.z.Regional_t = [zscore(avg.delta_t) zscore(avg.theta_t) zscore(avg.alpha_t) zscore(avg.beta_t) zscore(avg.gamma_t)];   
    
    
    [cca.delta_o,avg.delta_o] = Power_spectrum(channel_o, [0.5 4], cnt.fs);
    [cca.theta_o,avg.theta_o] = Power_spectrum(channel_o, [4 8], cnt.fs);
    [cca.alpha_o,avg.alpha_o] = Power_spectrum(channel_o, [8 13], cnt.fs);
    [cca.beta_o,avg.beta_o] = Power_spectrum(channel_o, [13 30], cnt.fs);
    [cca.gamma_o,avg.gamma_o] = Power_spectrum(channel_o, [30 40], cnt.fs);
    avg.delta_o = avg.delta_o'; avg.theta_o=avg.theta_o'; avg.alpha_o=avg.alpha_o'; avg.beta_o = avg.beta_o'; avg.gamma_o = avg.gamma_o';
    
    cca.Regional_o = [cca.delta_o cca.theta_o cca.alpha_o cca.beta_o cca.gamma_o];
    avg.Regional_o = [avg.delta_o avg.theta_o avg.alpha_o avg.beta_o avg.gamma_o];
    
    cca.RPL.delta_o = cca.delta_o ./ (cca.delta_o + cca.theta_o + cca.alpha_o + cca.beta_o + cca.gamma_o) ;
    avg.RPL.delta_o = avg.delta_o ./ sum(avg.Regional_o,2) ;
    cca.RPL.theta_o = cca.theta_o ./ (cca.delta_o + cca.theta_o + cca.alpha_o + cca.beta_o + cca.gamma_o) ;
    avg.RPL.theta_o = avg.theta_o ./ sum(avg.Regional_o,2) ;
    cca.RPL.alpha_o = cca.alpha_o ./ (cca.delta_o + cca.theta_o + cca.alpha_o + cca.beta_o + cca.gamma_o) ;
    avg.RPL.alpha_o = avg.alpha_o ./ sum(avg.Regional_o,2) ;
    cca.RPL.beta_o = cca.beta_o ./ (cca.delta_o + cca.theta_o + cca.alpha_o + cca.beta_o + cca.gamma_o) ;
    avg.RPL.beta_o = avg.beta_o ./ sum(avg.Regional_o,2) ;
    cca.RPL.gamma_o = cca.gamma_o ./ (cca.delta_o + cca.theta_o + cca.alpha_o + cca.beta_o + cca.gamma_o) ;
    avg.RPL.gamma_o = avg.gamma_o ./ sum(avg.Regional_o,2) ;     
    
    cca.RPL.Regional_o = [cca.RPL.delta_o cca.RPL.theta_o cca.RPL.alpha_o cca.RPL.beta_o cca.RPL.gamma_o]; 
    avg.RPL.Regional_o = [avg.RPL.delta_o avg.RPL.theta_o avg.RPL.alpha_o avg.RPL.beta_o avg.RPL.gamma_o];
    
    cca.z.Regional_o = [zscore(cca.delta_o) zscore(cca.theta_o) zscore(cca.alpha_o) zscore(cca.beta_o) zscore(cca.gamma_o)];
    avg.z.Regional_o = [zscore(avg.delta_o) zscore(avg.theta_o) zscore(avg.alpha_o) zscore(avg.beta_o) zscore(avg.gamma_o)];   
    
    
    
    
    % Regional Vertical Feature Extraction
    [cca.delta_f1,avg.delta_f1] = Power_spectrum(channel_f1, [0.5 4], cnt.fs);
    [cca.theta_f1,avg.theta_f1] = Power_spectrum(channel_f1, [4 8], cnt.fs);
    [cca.alpha_f1,avg.alpha_f1] = Power_spectrum(channel_f1, [8 13], cnt.fs);
    [cca.beta_f1,avg.beta_f1] = Power_spectrum(channel_f1, [13 30], cnt.fs);
    [cca.gamma_f1,avg.gamma_f1] = Power_spectrum(channel_f1, [30 40], cnt.fs);
    avg.delta_f1 = avg.delta_f1'; avg.theta_f1=avg.theta_f1'; avg.alpha_f1=avg.alpha_f1'; avg.beta_f1 = avg.beta_f1'; avg.gamma_f1 = avg.gamma_f1';
    
    cca.Regional_f1 = [cca.delta_f1 cca.theta_f1 cca.alpha_f1 cca.beta_f1 cca.gamma_f1];
    avg.Regional_f1 = [avg.delta_f1 avg.theta_f1 avg.alpha_f1 avg.beta_f1 avg.gamma_f1];
    
    cca.RPL.delta_f1 = cca.delta_f1 ./ (cca.delta_f1 + cca.theta_f1 + cca.alpha_f1 + cca.beta_f1 + cca.gamma_f1) ;
    avg.RPL.delta_f1 = avg.delta_f1 ./ sum(avg.Regional_f1,2) ;
    cca.RPL.theta_f1 = cca.theta_f1 ./ (cca.delta_f1 + cca.theta_f1 + cca.alpha_f1 + cca.beta_f1 + cca.gamma_f1) ;
    avg.RPL.theta_f1 = avg.theta_f1 ./ sum(avg.Regional_f1,2) ;
    cca.RPL.alpha_f1 = cca.alpha_f1 ./ (cca.delta_f1 + cca.theta_f1 + cca.alpha_f1 + cca.beta_f1 + cca.gamma_f1) ;
    avg.RPL.alpha_f1 = avg.alpha_f1 ./ sum(avg.Regional_f1,2) ;
    cca.RPL.beta_f1 = cca.beta_f1 ./ (cca.delta_f1 + cca.theta_f1 + cca.alpha_f1 + cca.beta_f1 + cca.gamma_f1) ;
    avg.RPL.beta_f1 = avg.beta_f1 ./ sum(avg.Regional_f1,2) ;
    cca.RPL.gamma_f1 = cca.gamma_f1 ./ (cca.delta_f1 + cca.theta_f1 + cca.alpha_f1 + cca.beta_f1 + cca.gamma_f1) ;
    avg.RPL.gamma_f1 = avg.gamma_f1 ./ sum(avg.Regional_f1,2) ;    
    
    cca.RPL.Regional_f1 = [cca.RPL.delta_f1 cca.RPL.theta_f1 cca.RPL.alpha_f1 cca.RPL.beta_f1 cca.RPL.gamma_f1]; 
    avg.RPL.Regional_f1 = [avg.RPL.delta_f1 avg.RPL.theta_f1 avg.RPL.alpha_f1 avg.RPL.beta_f1 avg.RPL.gamma_f1];
    
    cca.z.Regional_f1 = [zscore(cca.delta_f1) zscore(cca.theta_f1) zscore(cca.alpha_f1) zscore(cca.beta_f1) zscore(cca.gamma_f1)];
    avg.z.Regional_f1 = [zscore(avg.delta_f1) zscore(avg.theta_f1) zscore(avg.alpha_f1) zscore(avg.beta_f1) zscore(avg.gamma_f1)];   
    
    
    [cca.delta_f2,avg.delta_f2] = Power_spectrum(channel_f2, [0.5 4], cnt.fs);
    [cca.theta_f2,avg.theta_f2] = Power_spectrum(channel_f2, [4 8], cnt.fs);
    [cca.alpha_f2,avg.alpha_f2] = Power_spectrum(channel_f2, [8 13], cnt.fs);
    [cca.beta_f2,avg.beta_f2] = Power_spectrum(channel_f2, [13 30], cnt.fs);
    [cca.gamma_f2,avg.gamma_f2] = Power_spectrum(channel_f2, [30 40], cnt.fs);
    avg.delta_f2 = avg.delta_f2'; avg.theta_f2=avg.theta_f2'; avg.alpha_f2=avg.alpha_f2'; avg.beta_f2 = avg.beta_f2'; avg.gamma_f2 = avg.gamma_f2';
    
    cca.Regional_f2 = [cca.delta_f2 cca.theta_f2 cca.alpha_f2 cca.beta_f2 cca.gamma_f2];
    avg.Regional_f2 = [avg.delta_f2 avg.theta_f2 avg.alpha_f2 avg.beta_f2 avg.gamma_f2];
    
    cca.RPL.delta_f2 = cca.delta_f2 ./ (cca.delta_f2 + cca.theta_f2 + cca.alpha_f2 + cca.beta_f2 + cca.gamma_f2) ;
    avg.RPL.delta_f2 = avg.delta_f2 ./ sum(avg.Regional_f2,2) ;
    cca.RPL.theta_f2 = cca.theta_f2 ./ (cca.delta_f2 + cca.theta_f2 + cca.alpha_f2 + cca.beta_f2 + cca.gamma_f2) ;
    avg.RPL.theta_f2 = avg.theta_f2 ./ sum(avg.Regional_f2,2) ;
    cca.RPL.alpha_f2 = cca.alpha_f2 ./ (cca.delta_f2 + cca.theta_f2 + cca.alpha_f2 + cca.beta_f2 + cca.gamma_f2) ;
    avg.RPL.alpha_f2 = avg.alpha_f2 ./ sum(avg.Regional_f2,2) ;
    cca.RPL.beta_f2 = cca.beta_f2 ./ (cca.delta_f2 + cca.theta_f2 + cca.alpha_f2 + cca.beta_f2 + cca.gamma_f2) ;
    avg.RPL.beta_f2 = avg.beta_f2 ./ sum(avg.Regional_f2,2) ;
    cca.RPL.gamma_f2 = cca.gamma_f2 ./ (cca.delta_f2 + cca.theta_f2 + cca.alpha_f2 + cca.beta_f2 + cca.gamma_f2) ;
    avg.RPL.gamma_f2 = avg.gamma_f2 ./ sum(avg.Regional_f2,2) ;    
    
    cca.RPL.Regional_f2 = [cca.RPL.delta_f2 cca.RPL.theta_f2 cca.RPL.alpha_f2 cca.RPL.beta_f2 cca.RPL.gamma_f2]; 
    avg.RPL.Regional_f2 = [avg.RPL.delta_f2 avg.RPL.theta_f2 avg.RPL.alpha_f2 avg.RPL.beta_f2 avg.RPL.gamma_f2];
    
    cca.z.Regional_f2 = [zscore(cca.delta_f2) zscore(cca.theta_f2) zscore(cca.alpha_f2) zscore(cca.beta_f2) zscore(cca.gamma_f2)];
    avg.z.Regional_f2 = [zscore(avg.delta_f2) zscore(avg.theta_f2) zscore(avg.alpha_f2) zscore(avg.beta_f2) zscore(avg.gamma_f2)];       
    
    
    [cca.delta_f3,avg.delta_f3] = Power_spectrum(channel_f3, [0.5 4], cnt.fs);
    [cca.theta_f3,avg.theta_f3] = Power_spectrum(channel_f3, [4 8], cnt.fs);
    [cca.alpha_f3,avg.alpha_f3] = Power_spectrum(channel_f3, [8 13], cnt.fs);
    [cca.beta_f3,avg.beta_f3] = Power_spectrum(channel_f3, [13 30], cnt.fs);
    [cca.gamma_f3,avg.gamma_f3] = Power_spectrum(channel_f3, [30 40], cnt.fs);
    avg.delta_f3 = avg.delta_f3'; avg.theta_f3=avg.theta_f3'; avg.alpha_f3=avg.alpha_f3'; avg.beta_f3 = avg.beta_f3'; avg.gamma_f3 = avg.gamma_f3';
    
    cca.Regional_f3 = [cca.delta_f3 cca.theta_f3 cca.alpha_f3 cca.beta_f3 cca.gamma_f3];
    avg.Regional_f3 = [avg.delta_f3 avg.theta_f3 avg.alpha_f3 avg.beta_f3 avg.gamma_f3];
    
    cca.RPL.delta_f3 = cca.delta_f3 ./ (cca.delta_f3 + cca.theta_f3 + cca.alpha_f3 + cca.beta_f3 + cca.gamma_f3) ;
    avg.RPL.delta_f3 = avg.delta_f3 ./ sum(avg.Regional_f3,2) ;
    cca.RPL.theta_f3 = cca.theta_f3 ./ (cca.delta_f3 + cca.theta_f3 + cca.alpha_f3 + cca.beta_f3 + cca.gamma_f3) ;
    avg.RPL.theta_f3 = avg.theta_f3 ./ sum(avg.Regional_f3,2) ;
    cca.RPL.alpha_f3 = cca.alpha_f3 ./ (cca.delta_f3 + cca.theta_f3 + cca.alpha_f3 + cca.beta_f3 + cca.gamma_f3) ;
    avg.RPL.alpha_f3 = avg.alpha_f3 ./ sum(avg.Regional_f3,2) ;
    cca.RPL.beta_f3 = cca.beta_f3 ./ (cca.delta_f3 + cca.theta_f3 + cca.alpha_f3 + cca.beta_f3 + cca.gamma_f3) ;
    avg.RPL.beta_f3 = avg.beta_f3 ./ sum(avg.Regional_f3,2) ;
    cca.RPL.gamma_f3 = cca.gamma_f3 ./ (cca.delta_f3 + cca.theta_f3 + cca.alpha_f3 + cca.beta_f3 + cca.gamma_f3) ;
    avg.RPL.gamma_f3 = avg.gamma_f3 ./ sum(avg.Regional_f3,2) ;    
    
    cca.RPL.Regional_f3 = [cca.RPL.delta_f3 cca.RPL.theta_f3 cca.RPL.alpha_f3 cca.RPL.beta_f3 cca.RPL.gamma_f3]; 
    avg.RPL.Regional_f3 = [avg.RPL.delta_f3 avg.RPL.theta_f3 avg.RPL.alpha_f3 avg.RPL.beta_f3 avg.RPL.gamma_f3];
    
    cca.z.Regional_f3 = [zscore(cca.delta_f3) zscore(cca.theta_f3) zscore(cca.alpha_f3) zscore(cca.beta_f3) zscore(cca.gamma_f3)];
    avg.z.Regional_f3 = [zscore(avg.delta_f3) zscore(avg.theta_f3) zscore(avg.alpha_f3) zscore(avg.beta_f3) zscore(avg.gamma_f3)];           
    
    
    [cca.delta_c1,avg.delta_c1] = Power_spectrum(channel_c1, [0.5 4], cnt.fs);
    [cca.theta_c1,avg.theta_c1] = Power_spectrum(channel_c1, [4 8], cnt.fs);
    [cca.alpha_c1,avg.alpha_c1] = Power_spectrum(channel_c1, [8 13], cnt.fs);
    [cca.beta_c1,avg.beta_c1] = Power_spectrum(channel_c1, [13 30], cnt.fs);
    [cca.gamma_c1,avg.gamma_c1] = Power_spectrum(channel_c1, [30 40], cnt.fs);
    avg.delta_c1 = avg.delta_c1'; avg.theta_c1=avg.theta_c1'; avg.alpha_c1=avg.alpha_c1'; avg.beta_c1 = avg.beta_c1'; avg.gamma_c1 = avg.gamma_c1';
    
    cca.Regional_c1 = [cca.delta_c1 cca.theta_c1 cca.alpha_c1 cca.beta_c1 cca.gamma_c1];
    avg.Regional_c1 = [avg.delta_c1 avg.theta_c1 avg.alpha_c1 avg.beta_c1 avg.gamma_c1];
    
    cca.RPL.delta_c1 = cca.delta_c1 ./ (cca.delta_c1 + cca.theta_c1 + cca.alpha_c1 + cca.beta_c1 + cca.gamma_c1) ;
    avg.RPL.delta_c1 = avg.delta_c1 ./ sum(avg.Regional_c1,2) ;
    cca.RPL.theta_c1 = cca.theta_c1 ./ (cca.delta_c1 + cca.theta_c1 + cca.alpha_c1 + cca.beta_c1 + cca.gamma_c1) ;
    avg.RPL.theta_c1 = avg.theta_c1 ./ sum(avg.Regional_c1,2) ;
    cca.RPL.alpha_c1 = cca.alpha_c1 ./ (cca.delta_c1 + cca.theta_c1 + cca.alpha_c1 + cca.beta_c1 + cca.gamma_c1) ;
    avg.RPL.alpha_c1 = avg.alpha_c1 ./ sum(avg.Regional_c1,2) ;
    cca.RPL.beta_c1 = cca.beta_c1 ./ (cca.delta_c1 + cca.theta_c1 + cca.alpha_c1 + cca.beta_c1 + cca.gamma_c1) ;
    avg.RPL.beta_c1 = avg.beta_c1 ./ sum(avg.Regional_c1,2) ;
    cca.RPL.gamma_c1 = cca.gamma_c1 ./ (cca.delta_c1 + cca.theta_c1 + cca.alpha_c1 + cca.beta_c1 + cca.gamma_c1) ;
    avg.RPL.gamma_c1 = avg.gamma_c1 ./ sum(avg.Regional_c1,2) ;     
    
    cca.RPL.Regional_c1 = [cca.RPL.delta_c1 cca.RPL.theta_c1 cca.RPL.alpha_c1 cca.RPL.beta_c1 cca.RPL.gamma_c1]; 
    avg.RPL.Regional_c1 = [avg.RPL.delta_c1 avg.RPL.theta_c1 avg.RPL.alpha_c1 avg.RPL.beta_c1 avg.RPL.gamma_c1];
    
    cca.z.Regional_c1 = [zscore(cca.delta_c1) zscore(cca.theta_c1) zscore(cca.alpha_c1) zscore(cca.beta_c1) zscore(cca.gamma_c1)];
    avg.z.Regional_c1 = [zscore(avg.delta_c1) zscore(avg.theta_c1) zscore(avg.alpha_c1) zscore(avg.beta_c1) zscore(avg.gamma_c1)];               
    
    [cca.delta_c2,avg.delta_c2] = Power_spectrum(channel_c2, [0.5 4], cnt.fs);
    [cca.theta_c2,avg.theta_c2] = Power_spectrum(channel_c2, [4 8], cnt.fs);
    [cca.alpha_c2,avg.alpha_c2] = Power_spectrum(channel_c2, [8 13], cnt.fs);
    [cca.beta_c2,avg.beta_c2] = Power_spectrum(channel_c2, [13 30], cnt.fs);
    [cca.gamma_c2,avg.gamma_c2] = Power_spectrum(channel_c2, [30 40], cnt.fs);
    avg.delta_c2 = avg.delta_c2'; avg.theta_c2=avg.theta_c2'; avg.alpha_c2=avg.alpha_c2'; avg.beta_c2 = avg.beta_c2'; avg.gamma_c2 = avg.gamma_c2';
    
    cca.Regional_c2 = [cca.delta_c2 cca.theta_c2 cca.alpha_c2 cca.beta_c2 cca.gamma_c2];
    avg.Regional_c2 = [avg.delta_c2 avg.theta_c2 avg.alpha_c2 avg.beta_c2 avg.gamma_c2];
    
    cca.RPL.delta_c2 = cca.delta_c2 ./ (cca.delta_c2 + cca.theta_c2 + cca.alpha_c2 + cca.beta_c2 + cca.gamma_c2) ;
    avg.RPL.delta_c2 = avg.delta_c2 ./ sum(avg.Regional_c2,2) ;
    cca.RPL.theta_c2 = cca.theta_c2 ./ (cca.delta_c2 + cca.theta_c2 + cca.alpha_c2 + cca.beta_c2 + cca.gamma_c2) ;
    avg.RPL.theta_c2 = avg.theta_c2 ./ sum(avg.Regional_c2,2) ;
    cca.RPL.alpha_c2 = cca.alpha_c2 ./ (cca.delta_c2 + cca.theta_c2 + cca.alpha_c2 + cca.beta_c2 + cca.gamma_c2) ;
    avg.RPL.alpha_c2 = avg.alpha_c2 ./ sum(avg.Regional_c2,2) ;
    cca.RPL.beta_c2 = cca.beta_c2 ./ (cca.delta_c2 + cca.theta_c2 + cca.alpha_c2 + cca.beta_c2 + cca.gamma_c2) ;
    avg.RPL.beta_c2 = avg.beta_c2 ./ sum(avg.Regional_c2,2) ;
    cca.RPL.gamma_c2 = cca.gamma_c2 ./ (cca.delta_c2 + cca.theta_c2 + cca.alpha_c2 + cca.beta_c2 + cca.gamma_c2) ;
    avg.RPL.gamma_c2 = avg.gamma_c2 ./ sum(avg.Regional_c2,2) ;     
    
    cca.RPL.Regional_c2 = [cca.RPL.delta_c2 cca.RPL.theta_c2 cca.RPL.alpha_c2 cca.RPL.beta_c2 cca.RPL.gamma_c2]; 
    avg.RPL.Regional_c2 = [avg.RPL.delta_c2 avg.RPL.theta_c2 avg.RPL.alpha_c2 avg.RPL.beta_c2 avg.RPL.gamma_c2];
    
    cca.z.Regional_c2 = [zscore(cca.delta_c2) zscore(cca.theta_c2) zscore(cca.alpha_c2) zscore(cca.beta_c2) zscore(cca.gamma_c2)];
    avg.z.Regional_c2 = [zscore(avg.delta_c2) zscore(avg.theta_c2) zscore(avg.alpha_c2) zscore(avg.beta_c2) zscore(avg.gamma_c2)];                   
    
    [cca.delta_c3,avg.delta_c3] = Power_spectrum(channel_c3, [0.5 4], cnt.fs);
    [cca.theta_c3,avg.theta_c3] = Power_spectrum(channel_c3, [4 8], cnt.fs);
    [cca.alpha_c3,avg.alpha_c3] = Power_spectrum(channel_c3, [8 13], cnt.fs);
    [cca.beta_c3,avg.beta_c3] = Power_spectrum(channel_c3, [13 30], cnt.fs);
    [cca.gamma_c3,avg.gamma_c3] = Power_spectrum(channel_c3, [30 40], cnt.fs);
    avg.delta_c3 = avg.delta_c3'; avg.theta_c3=avg.theta_c3'; avg.alpha_c3=avg.alpha_c3'; avg.beta_c3 = avg.beta_c3'; avg.gamma_c3 = avg.gamma_c3';
    
    cca.Regional_c3 = [cca.delta_c3 cca.theta_c3 cca.alpha_c3 cca.beta_c3 cca.gamma_c3];
    avg.Regional_c3 = [avg.delta_c3 avg.theta_c3 avg.alpha_c3 avg.beta_c3 avg.gamma_c3];
    
    cca.RPL.delta_c3 = cca.delta_c3 ./ (cca.delta_c3 + cca.theta_c3 + cca.alpha_c3 + cca.beta_c3 + cca.gamma_c3) ;
    avg.RPL.delta_c3 = avg.delta_c3 ./ sum(avg.Regional_c3,2) ;
    cca.RPL.theta_c3 = cca.theta_c3 ./ (cca.delta_c3 + cca.theta_c3 + cca.alpha_c3 + cca.beta_c3 + cca.gamma_c3) ;
    avg.RPL.theta_c3 = avg.theta_c3 ./ sum(avg.Regional_c3,2) ;
    cca.RPL.alpha_c3 = cca.alpha_c3 ./ (cca.delta_c3 + cca.theta_c3 + cca.alpha_c3 + cca.beta_c3 + cca.gamma_c3) ;
    avg.RPL.alpha_c3 = avg.alpha_c3 ./ sum(avg.Regional_c3,2) ;
    cca.RPL.beta_c3 = cca.beta_c3 ./ (cca.delta_c3 + cca.theta_c3 + cca.alpha_c3 + cca.beta_c3 + cca.gamma_c3) ;
    avg.RPL.beta_c3 = avg.beta_c3 ./ sum(avg.Regional_c3,2) ;
    cca.RPL.gamma_c3 = cca.gamma_c3 ./ (cca.delta_c3 + cca.theta_c3 + cca.alpha_c3 + cca.beta_c3 + cca.gamma_c3) ;
    avg.RPL.gamma_c3 = avg.gamma_c3 ./ sum(avg.Regional_c3,2) ;     
    
    cca.RPL.Regional_c3 = [cca.RPL.delta_c3 cca.RPL.theta_c3 cca.RPL.alpha_c3 cca.RPL.beta_c3 cca.RPL.gamma_c3]; 
    avg.RPL.Regional_c3 = [avg.RPL.delta_c3 avg.RPL.theta_c3 avg.RPL.alpha_c3 avg.RPL.beta_c3 avg.RPL.gamma_c3];
    
    cca.z.Regional_c3 = [zscore(cca.delta_c3) zscore(cca.theta_c3) zscore(cca.alpha_c3) zscore(cca.beta_c3) zscore(cca.gamma_c3)];
    avg.z.Regional_c3 = [zscore(avg.delta_c3) zscore(avg.theta_c3) zscore(avg.alpha_c3) zscore(avg.beta_c3) zscore(avg.gamma_c3)];                   
    
    [cca.delta_p1,avg.delta_p1] = Power_spectrum(channel_p1, [0.5 4], cnt.fs);
    [cca.theta_p1,avg.theta_p1] = Power_spectrum(channel_p1, [4 8], cnt.fs);
    [cca.alpha_p1,avg.alpha_p1] = Power_spectrum(channel_p1, [8 13], cnt.fs);
    [cca.beta_p1,avg.beta_p1] = Power_spectrum(channel_p1, [13 30], cnt.fs);
    [cca.gamma_p1,avg.gamma_p1] = Power_spectrum(channel_p1, [30 40], cnt.fs);
    avg.delta_p1 = avg.delta_p1'; avg.theta_p1=avg.theta_p1'; avg.alpha_p1=avg.alpha_p1'; avg.beta_p1 = avg.beta_p1'; avg.gamma_p1 = avg.gamma_p1';
    
    cca.Regional_p1 = [cca.delta_p1 cca.theta_p1 cca.alpha_p1 cca.beta_p1 cca.gamma_p1];
    avg.Regional_p1 = [avg.delta_p1 avg.theta_p1 avg.alpha_p1 avg.beta_p1 avg.gamma_p1];
    
    cca.RPL.delta_p1 = cca.delta_p1 ./ (cca.delta_p1 + cca.theta_p1 + cca.alpha_p1 + cca.beta_p1 + cca.gamma_p1) ;
    avg.RPL.delta_p1 = avg.delta_p1 ./ sum(avg.Regional_p1,2) ;
    cca.RPL.theta_p1 = cca.theta_p1 ./ (cca.delta_p1 + cca.theta_p1 + cca.alpha_p1 + cca.beta_p1 + cca.gamma_p1) ;
    avg.RPL.theta_p1 = avg.theta_p1 ./ sum(avg.Regional_p1,2) ;
    cca.RPL.alpha_p1 = cca.alpha_p1 ./ (cca.delta_p1 + cca.theta_p1 + cca.alpha_p1 + cca.beta_p1 + cca.gamma_p1) ;
    avg.RPL.alpha_p1 = avg.alpha_p1 ./ sum(avg.Regional_p1,2) ;
    cca.RPL.beta_p1 = cca.beta_p1 ./ (cca.delta_p1 + cca.theta_p1 + cca.alpha_p1 + cca.beta_p1 + cca.gamma_p1) ;
    avg.RPL.beta_p1 = avg.beta_p1 ./ sum(avg.Regional_p1,2) ;
    cca.RPL.gamma_p1 = cca.gamma_p1 ./ (cca.delta_p1 + cca.theta_p1 + cca.alpha_p1 + cca.beta_p1 + cca.gamma_p1) ;
    avg.RPL.gamma_p1 = avg.gamma_p1 ./ sum(avg.Regional_p1,2) ;     
    
    cca.RPL.Regional_p1 = [cca.RPL.delta_p1 cca.RPL.theta_p1 cca.RPL.alpha_p1 cca.RPL.beta_p1 cca.RPL.gamma_p1]; 
    avg.RPL.Regional_p1 = [avg.RPL.delta_p1 avg.RPL.theta_p1 avg.RPL.alpha_p1 avg.RPL.beta_p1 avg.RPL.gamma_p1];
    
    cca.z.Regional_p1 = [zscore(cca.delta_p1) zscore(cca.theta_p1) zscore(cca.alpha_p1) zscore(cca.beta_p1) zscore(cca.gamma_p1)];
    avg.z.Regional_p1 = [zscore(avg.delta_p1) zscore(avg.theta_p1) zscore(avg.alpha_p1) zscore(avg.beta_p1) zscore(avg.gamma_p1)];               
    
    [cca.delta_p2,avg.delta_p2] = Power_spectrum(channel_p2, [0.5 4], cnt.fs);
    [cca.theta_p2,avg.theta_p2] = Power_spectrum(channel_p2, [4 8], cnt.fs);
    [cca.alpha_p2,avg.alpha_p2] = Power_spectrum(channel_p2, [8 13], cnt.fs);
    [cca.beta_p2,avg.beta_p2] = Power_spectrum(channel_p2, [13 30], cnt.fs);
    [cca.gamma_p2,avg.gamma_p2] = Power_spectrum(channel_p2, [30 40], cnt.fs);
    avg.delta_p2 = avg.delta_p2'; avg.theta_p2=avg.theta_p2'; avg.alpha_p2=avg.alpha_p2'; avg.beta_p2 = avg.beta_p2'; avg.gamma_p2 = avg.gamma_p2';
    
    cca.Regional_p2 = [cca.delta_p2 cca.theta_p2 cca.alpha_p2 cca.beta_p2 cca.gamma_p2];
    avg.Regional_p2 = [avg.delta_p2 avg.theta_p2 avg.alpha_p2 avg.beta_p2 avg.gamma_p2];
    
    cca.RPL.delta_p2 = cca.delta_p2 ./ (cca.delta_p2 + cca.theta_p2 + cca.alpha_p2 + cca.beta_p2 + cca.gamma_p2) ;
    avg.RPL.delta_p2 = avg.delta_p2 ./ sum(avg.Regional_p2,2) ;
    cca.RPL.theta_p2 = cca.theta_p2 ./ (cca.delta_p2 + cca.theta_p2 + cca.alpha_p2 + cca.beta_p2 + cca.gamma_p2) ;
    avg.RPL.theta_p2 = avg.theta_p2 ./ sum(avg.Regional_p2,2) ;
    cca.RPL.alpha_p2 = cca.alpha_p2 ./ (cca.delta_p2 + cca.theta_p2 + cca.alpha_p2 + cca.beta_p2 + cca.gamma_p2) ;
    avg.RPL.alpha_p2 = avg.alpha_p2 ./ sum(avg.Regional_p2,2) ;
    cca.RPL.beta_p2 = cca.beta_p2 ./ (cca.delta_p2 + cca.theta_p2 + cca.alpha_p2 + cca.beta_p2 + cca.gamma_p2) ;
    avg.RPL.beta_p2 = avg.beta_p2 ./ sum(avg.Regional_p2,2) ;
    cca.RPL.gamma_p2 = cca.gamma_p2 ./ (cca.delta_p2 + cca.theta_p2 + cca.alpha_p2 + cca.beta_p2 + cca.gamma_p2) ;
    avg.RPL.gamma_p2 = avg.gamma_p2 ./ sum(avg.Regional_p2,2) ;     
    
    cca.RPL.Regional_p2 = [cca.RPL.delta_p2 cca.RPL.theta_p2 cca.RPL.alpha_p2 cca.RPL.beta_p2 cca.RPL.gamma_p2]; 
    avg.RPL.Regional_p2 = [avg.RPL.delta_p2 avg.RPL.theta_p2 avg.RPL.alpha_p2 avg.RPL.beta_p2 avg.RPL.gamma_p2];
    
    cca.z.Regional_p2 = [zscore(cca.delta_p2) zscore(cca.theta_p2) zscore(cca.alpha_p2) zscore(cca.beta_p2) zscore(cca.gamma_p2)];
    avg.z.Regional_p2 = [zscore(avg.delta_p2) zscore(avg.theta_p2) zscore(avg.alpha_p2) zscore(avg.beta_p2) zscore(avg.gamma_p2)];               
    
    
    [cca.delta_p3,avg.delta_p3] = Power_spectrum(channel_p3, [0.5 4], cnt.fs);
    [cca.theta_p3,avg.theta_p3] = Power_spectrum(channel_p3, [4 8], cnt.fs);
    [cca.alpha_p3,avg.alpha_p3] = Power_spectrum(channel_p3, [8 13], cnt.fs);
    [cca.beta_p3,avg.beta_p3] = Power_spectrum(channel_p3, [13 30], cnt.fs);
    [cca.gamma_p3,avg.gamma_p3] = Power_spectrum(channel_p3, [30 40], cnt.fs);
    avg.delta_p3 = avg.delta_p3'; avg.theta_p3=avg.theta_p3'; avg.alpha_p3=avg.alpha_p3'; avg.beta_p3 = avg.beta_p3'; avg.gamma_p3 = avg.gamma_p3';
    
    cca.Regional_p3 = [cca.delta_p3 cca.theta_p3 cca.alpha_p3 cca.beta_p3 cca.gamma_p3];
    avg.Regional_p3 = [avg.delta_p3 avg.theta_p3 avg.alpha_p3 avg.beta_p3 avg.gamma_p3];
    
    cca.RPL.delta_p3 = cca.delta_p3 ./ (cca.delta_p3 + cca.theta_p3 + cca.alpha_p3 + cca.beta_p3 + cca.gamma_p3) ;
    avg.RPL.delta_p3 = avg.delta_p3 ./ sum(avg.Regional_p3,2) ;
    cca.RPL.theta_p3 = cca.theta_p3 ./ (cca.delta_p3 + cca.theta_p3 + cca.alpha_p3 + cca.beta_p3 + cca.gamma_p3) ;
    avg.RPL.theta_p3 = avg.theta_p3 ./ sum(avg.Regional_p3,2) ;
    cca.RPL.alpha_p3 = cca.alpha_p3 ./ (cca.delta_p3 + cca.theta_p3 + cca.alpha_p3 + cca.beta_p3 + cca.gamma_p3) ;
    avg.RPL.alpha_p3 = avg.alpha_p3 ./ sum(avg.Regional_p3,2) ;
    cca.RPL.beta_p3 = cca.beta_p3 ./ (cca.delta_p3 + cca.theta_p3 + cca.alpha_p3 + cca.beta_p3 + cca.gamma_p3) ;
    avg.RPL.beta_p3 = avg.beta_p3 ./ sum(avg.Regional_p3,2) ;
    cca.RPL.gamma_p3 = cca.gamma_p3 ./ (cca.delta_p3 + cca.theta_p3 + cca.alpha_p3 + cca.beta_p3 + cca.gamma_p3) ;
    avg.RPL.gamma_p3 = avg.gamma_p3 ./ sum(avg.Regional_p3,2) ;     
    
    cca.RPL.Regional_p3 = [cca.RPL.delta_p3 cca.RPL.theta_p3 cca.RPL.alpha_p3 cca.RPL.beta_p3 cca.RPL.gamma_p3]; 
    avg.RPL.Regional_p3 = [avg.RPL.delta_p3 avg.RPL.theta_p3 avg.RPL.alpha_p3 avg.RPL.beta_p3 avg.RPL.gamma_p3];
    
    cca.z.Regional_p3 = [zscore(cca.delta_p3) zscore(cca.theta_p3) zscore(cca.alpha_p3) zscore(cca.beta_p3) zscore(cca.gamma_p3)];
    avg.z.Regional_p3 = [zscore(avg.delta_p3) zscore(avg.theta_p3) zscore(avg.alpha_p3) zscore(avg.beta_p3) zscore(avg.gamma_p3)];               
    
    
    [cca.delta_t1,avg.delta_t1] = Power_spectrum(channel_t1, [0.5 4], cnt.fs);
    [cca.theta_t1,avg.theta_t1] = Power_spectrum(channel_t1, [4 8], cnt.fs);
    [cca.alpha_t1,avg.alpha_t1] = Power_spectrum(channel_t1, [8 13], cnt.fs);
    [cca.beta_t1,avg.beta_t1] = Power_spectrum(channel_t1, [13 30], cnt.fs);
    [cca.gamma_t1,avg.gamma_t1] = Power_spectrum(channel_t1, [30 40], cnt.fs);
    avg.delta_t1 = avg.delta_t1'; avg.theta_t1=avg.theta_t1'; avg.alpha_t1=avg.alpha_t1'; avg.beta_t1 = avg.beta_t1'; avg.gamma_t1 = avg.gamma_t1';
    
    cca.Regional_t1 = [cca.delta_t1 cca.theta_t1 cca.alpha_t1 cca.beta_t1 cca.gamma_t1];
    avg.Regional_t1 = [avg.delta_t1 avg.theta_t1 avg.alpha_t1 avg.beta_t1 avg.gamma_t1];
    
    cca.RPL.delta_t1 = cca.delta_t1 ./ (cca.delta_t1 + cca.theta_t1 + cca.alpha_t1 + cca.beta_t1 + cca.gamma_t1) ;
    avg.RPL.delta_t1 = avg.delta_t1 ./ sum(avg.Regional_t1,2) ;
    cca.RPL.theta_t1 = cca.theta_t1 ./ (cca.delta_t1 + cca.theta_t1 + cca.alpha_t1 + cca.beta_t1 + cca.gamma_t1) ;
    avg.RPL.theta_t1 = avg.theta_t1 ./ sum(avg.Regional_t1,2) ;
    cca.RPL.alpha_t1 = cca.alpha_t1 ./ (cca.delta_t1 + cca.theta_t1 + cca.alpha_t1 + cca.beta_t1 + cca.gamma_t1) ;
    avg.RPL.alpha_t1 = avg.alpha_t1 ./ sum(avg.Regional_t1,2) ;
    cca.RPL.beta_t1 = cca.beta_t1 ./ (cca.delta_t1 + cca.theta_t1 + cca.alpha_t1 + cca.beta_t1 + cca.gamma_t1) ;
    avg.RPL.beta_t1 = avg.beta_t1 ./ sum(avg.Regional_t1,2) ;
    cca.RPL.gamma_t1 = cca.gamma_t1 ./ (cca.delta_t1 + cca.theta_t1 + cca.alpha_t1 + cca.beta_t1 + cca.gamma_t1) ;
    avg.RPL.gamma_t1 = avg.gamma_t1 ./ sum(avg.Regional_t1,2) ;     

    cca.RPL.Regional_t1 = [cca.RPL.delta_t1 cca.RPL.theta_t1 cca.RPL.alpha_t1 cca.RPL.beta_t1 cca.RPL.gamma_t1]; 
    avg.RPL.Regional_t1 = [avg.RPL.delta_t1 avg.RPL.theta_t1 avg.RPL.alpha_t1 avg.RPL.beta_t1 avg.RPL.gamma_t1];
    
    cca.z.Regional_t1 = [zscore(cca.delta_t1) zscore(cca.theta_t1) zscore(cca.alpha_t1) zscore(cca.beta_t1) zscore(cca.gamma_t1)];
    avg.z.Regional_t1 = [zscore(avg.delta_t1) zscore(avg.theta_t1) zscore(avg.alpha_t1) zscore(avg.beta_t1) zscore(avg.gamma_t1)];                   
    
    
    [cca.delta_t2,avg.delta_t2] = Power_spectrum(channel_t2, [0.5 4], cnt.fs);
    [cca.theta_t2,avg.theta_t2] = Power_spectrum(channel_t2, [4 8], cnt.fs);
    [cca.alpha_t2,avg.alpha_t2] = Power_spectrum(channel_t2, [8 13], cnt.fs);
    [cca.beta_t2,avg.beta_t2] = Power_spectrum(channel_t2, [13 30], cnt.fs);
    [cca.gamma_t2,avg.gamma_t2] = Power_spectrum(channel_t2, [30 40], cnt.fs);
    avg.delta_t2 = avg.delta_t2'; avg.theta_t2=avg.theta_t2'; avg.alpha_t2=avg.alpha_t2'; avg.beta_t2 = avg.beta_t2'; avg.gamma_t2 = avg.gamma_t2';
    
    cca.Regional_t2 = [cca.delta_t2 cca.theta_t2 cca.alpha_t2 cca.beta_t2 cca.gamma_t2];
    avg.Regional_t2 = [avg.delta_t2 avg.theta_t2 avg.alpha_t2 avg.beta_t2 avg.gamma_t2];
    
    cca.RPL.delta_t2 = cca.delta_t2 ./ (cca.delta_t2 + cca.theta_t2 + cca.alpha_t2 + cca.beta_t2 + cca.gamma_t2) ;
    avg.RPL.delta_t2 = avg.delta_t2 ./ sum(avg.Regional_t2,2) ;
    cca.RPL.theta_t2 = cca.theta_t2 ./ (cca.delta_t2 + cca.theta_t2 + cca.alpha_t2 + cca.beta_t2 + cca.gamma_t2) ;
    avg.RPL.theta_t2 = avg.theta_t2 ./ sum(avg.Regional_t2,2) ;
    cca.RPL.alpha_t2 = cca.alpha_t2 ./ (cca.delta_t2 + cca.theta_t2 + cca.alpha_t2 + cca.beta_t2 + cca.gamma_t2) ;
    avg.RPL.alpha_t2 = avg.alpha_t2 ./ sum(avg.Regional_t2,2) ;
    cca.RPL.beta_t2 = cca.beta_t2 ./ (cca.delta_t2 + cca.theta_t2 + cca.alpha_t2 + cca.beta_t2 + cca.gamma_t2) ;
    avg.RPL.beta_t2 = avg.beta_t2 ./ sum(avg.Regional_t2,2) ;
    cca.RPL.gamma_t2 = cca.gamma_t2 ./ (cca.delta_t2 + cca.theta_t2 + cca.alpha_t2 + cca.beta_t2 + cca.gamma_t2) ;
    avg.RPL.gamma_t2 = avg.gamma_t2 ./ sum(avg.Regional_t2,2) ;     
    
    cca.RPL.Regional_t2 = [cca.RPL.delta_t2 cca.RPL.theta_t2 cca.RPL.alpha_t2 cca.RPL.beta_t2 cca.RPL.gamma_t2]; 
    avg.RPL.Regional_t2 = [avg.RPL.delta_t2 avg.RPL.theta_t2 avg.RPL.alpha_t2 avg.RPL.beta_t2 avg.RPL.gamma_t2];
    
    cca.z.Regional_t2 = [zscore(cca.delta_t2) zscore(cca.theta_t2) zscore(cca.alpha_t2) zscore(cca.beta_t2) zscore(cca.gamma_t2)];
    avg.z.Regional_t2 = [zscore(avg.delta_t2) zscore(avg.theta_t2) zscore(avg.alpha_t2) zscore(avg.beta_t2) zscore(avg.gamma_t2)];                   
    
    
    
    
    
    [cca.delta_o1,avg.delta_o1] = Power_spectrum(channel_o1, [0.5 4], cnt.fs);
    [cca.theta_o1,avg.theta_o1] = Power_spectrum(channel_o1, [4 8], cnt.fs);
    [cca.alpha_o1,avg.alpha_o1] = Power_spectrum(channel_o1, [8 13], cnt.fs);
    [cca.beta_o1,avg.beta_o1] = Power_spectrum(channel_o1, [13 30], cnt.fs);
    [cca.gamma_o1,avg.gamma_o1] = Power_spectrum(channel_o1, [30 40], cnt.fs);
    avg.delta_o1 = avg.delta_o1'; avg.theta_o1=avg.theta_o1'; avg.alpha_o1=avg.alpha_o1'; avg.beta_o1 = avg.beta_o1'; avg.gamma_o1 = avg.gamma_o1';
    
    cca.Regional_o1 = [cca.delta_o1 cca.theta_o1 cca.alpha_o1 cca.beta_o1 cca.gamma_o1];
    avg.Regional_o1 = [avg.delta_o1 avg.theta_o1 avg.alpha_o1 avg.beta_o1 avg.gamma_o1];

    cca.RPL.delta_o1 = cca.delta_o1 ./ (cca.delta_o1 + cca.theta_o1 + cca.alpha_o1 + cca.beta_o1 + cca.gamma_o1) ;
    avg.RPL.delta_o1 = avg.delta_o1 ./ sum(avg.Regional_o1,2) ;
    cca.RPL.theta_o1 = cca.theta_o1 ./ (cca.delta_o1 + cca.theta_o1 + cca.alpha_o1 + cca.beta_o1 + cca.gamma_o1) ;
    avg.RPL.theta_o1 = avg.theta_o1 ./ sum(avg.Regional_o1,2) ;
    cca.RPL.alpha_o1 = cca.alpha_o1 ./ (cca.delta_o1 + cca.theta_o1 + cca.alpha_o1 + cca.beta_o1 + cca.gamma_o1) ;
    avg.RPL.alpha_o1 = avg.alpha_o1 ./ sum(avg.Regional_o1,2) ;
    cca.RPL.beta_o1 = cca.beta_o1 ./ (cca.delta_o1 + cca.theta_o1 + cca.alpha_o1 + cca.beta_o1 + cca.gamma_o1) ;
    avg.RPL.beta_o1 = avg.beta_o1 ./ sum(avg.Regional_o1,2) ;
    cca.RPL.gamma_o1 = cca.gamma_o1 ./ (cca.delta_o1 + cca.theta_o1 + cca.alpha_o1 + cca.beta_o1 + cca.gamma_o1) ;
    avg.RPL.gamma_o1 = avg.gamma_o1 ./ sum(avg.Regional_o1,2) ;     

    cca.RPL.Regional_o1 = [cca.RPL.delta_o1 cca.RPL.theta_o1 cca.RPL.alpha_o1 cca.RPL.beta_o1 cca.RPL.gamma_o1]; 
    avg.RPL.Regional_o1 = [avg.RPL.delta_o1 avg.RPL.theta_o1 avg.RPL.alpha_o1 avg.RPL.beta_o1 avg.RPL.gamma_o1];
    
    cca.z.Regional_o1 = [zscore(cca.delta_o1) zscore(cca.theta_o1) zscore(cca.alpha_o1) zscore(cca.beta_o1) zscore(cca.gamma_o1)];
    avg.z.Regional_o1 = [zscore(avg.delta_o1) zscore(avg.theta_o1) zscore(avg.alpha_o1) zscore(avg.beta_o1) zscore(avg.gamma_o1)];                   

    
    
    
    [cca.delta_o2,avg.delta_o2] = Power_spectrum(channel_o2, [0.5 4], cnt.fs);
    [cca.theta_o2,avg.theta_o2] = Power_spectrum(channel_o2, [4 8], cnt.fs);
    [cca.alpha_o2,avg.alpha_o2] = Power_spectrum(channel_o2, [8 13], cnt.fs);
    [cca.beta_o2,avg.beta_o2] = Power_spectrum(channel_o2, [13 30], cnt.fs);
    [cca.gamma_o2,avg.gamma_o2] = Power_spectrum(channel_o2, [30 40], cnt.fs);
    avg.delta_o2 = avg.delta_o2'; avg.theta_o2=avg.theta_o2'; avg.alpha_o2=avg.alpha_o2'; avg.beta_o2 = avg.beta_o2'; avg.gamma_o2 = avg.gamma_o2';
    
    cca.Regional_o2 = [cca.delta_o2 cca.theta_o2 cca.alpha_o2 cca.beta_o2 cca.gamma_o2];
    avg.Regional_o2 = [avg.delta_o2 avg.theta_o2 avg.alpha_o2 avg.beta_o2 avg.gamma_o2];
    
    cca.RPL.delta_o2 = cca.delta_o2 ./ (cca.delta_o2 + cca.theta_o2 + cca.alpha_o2 + cca.beta_o2 + cca.gamma_o2) ;
    avg.RPL.delta_o2 = avg.delta_o2 ./ sum(avg.Regional_o2,2) ;
    cca.RPL.theta_o2 = cca.theta_o2 ./ (cca.delta_o2 + cca.theta_o2 + cca.alpha_o2 + cca.beta_o2 + cca.gamma_o2) ;
    avg.RPL.theta_o2 = avg.theta_o2 ./ sum(avg.Regional_o2,2) ;
    cca.RPL.alpha_o2 = cca.alpha_o2 ./ (cca.delta_o2 + cca.theta_o2 + cca.alpha_o2 + cca.beta_o2 + cca.gamma_o2) ;
    avg.RPL.alpha_o2 = avg.alpha_o2 ./ sum(avg.Regional_o2,2) ;
    cca.RPL.beta_o2 = cca.beta_o2 ./ (cca.delta_o2 + cca.theta_o2 + cca.alpha_o2 + cca.beta_o2 + cca.gamma_o2) ;
    avg.RPL.beta_o2 = avg.beta_o2 ./ sum(avg.Regional_o2,2) ;
    cca.RPL.gamma_o2 = cca.gamma_o2 ./ (cca.delta_o2 + cca.theta_o2 + cca.alpha_o2 + cca.beta_o2 + cca.gamma_o2) ;
    avg.RPL.gamma_o2 = avg.gamma_o2 ./ sum(avg.Regional_o2,2) ;     

    cca.RPL.Regional_o2 = [cca.RPL.delta_o2 cca.RPL.theta_o2 cca.RPL.alpha_o2 cca.RPL.beta_o2 cca.RPL.gamma_o2]; 
    avg.RPL.Regional_o2 = [avg.RPL.delta_o2 avg.RPL.theta_o2 avg.RPL.alpha_o2 avg.RPL.beta_o2 avg.RPL.gamma_o2];
        
    cca.z.Regional_o2 = [zscore(cca.delta_o2) zscore(cca.theta_o2) zscore(cca.alpha_o2) zscore(cca.beta_o2) zscore(cca.gamma_o2)];
    avg.z.Regional_o2 = [zscore(avg.delta_o2) zscore(avg.theta_o2) zscore(avg.alpha_o2) zscore(avg.beta_o2) zscore(avg.gamma_o2)];                   
    
    
    
    [cca.delta_o3,avg.delta_o3] = Power_spectrum(channel_o3, [0.5 4], cnt.fs);
    [cca.theta_o3,avg.theta_o3] = Power_spectrum(channel_o3, [4 8], cnt.fs);
    [cca.alpha_o3,avg.alpha_o3] = Power_spectrum(channel_o3, [8 13], cnt.fs);
    [cca.beta_o3,avg.beta_o3] = Power_spectrum(channel_o3, [13 30], cnt.fs);
    [cca.gamma_o3,avg.gamma_o3] = Power_spectrum(channel_o3, [30 40], cnt.fs);
    avg.delta_o3 = avg.delta_o3'; avg.theta_o3=avg.theta_o3'; avg.alpha_o3=avg.alpha_o3'; avg.beta_o3 = avg.beta_o3'; avg.gamma_o3 = avg.gamma_o3';
    
    cca.Regional_o3 = [cca.delta_o3 cca.theta_o3 cca.alpha_o3 cca.beta_o3 cca.gamma_o3];
    avg.Regional_o3 = [avg.delta_o3 avg.theta_o3 avg.alpha_o3 avg.beta_o3 avg.gamma_o3];

    cca.RPL.delta_o3 = cca.delta_o3 ./ (cca.delta_o3 + cca.theta_o3 + cca.alpha_o3 + cca.beta_o3 + cca.gamma_o3) ;
    avg.RPL.delta_o3 = avg.delta_o3 ./ sum(avg.Regional_o3,2) ;
    cca.RPL.theta_o3 = cca.theta_o3 ./ (cca.delta_o3 + cca.theta_o3 + cca.alpha_o3 + cca.beta_o3 + cca.gamma_o3) ;
    avg.RPL.theta_o3 = avg.theta_o3 ./ sum(avg.Regional_o3,2) ;
    cca.RPL.alpha_o3 = cca.alpha_o3 ./ (cca.delta_o3 + cca.theta_o3 + cca.alpha_o3 + cca.beta_o3 + cca.gamma_o3) ;
    avg.RPL.alpha_o3 = avg.alpha_o3 ./ sum(avg.Regional_o3,2) ;
    cca.RPL.beta_o3 = cca.beta_o3 ./ (cca.delta_o3 + cca.theta_o3 + cca.alpha_o3 + cca.beta_o3 + cca.gamma_o3) ;
    avg.RPL.beta_o3 = avg.beta_o3 ./ sum(avg.Regional_o3,2) ;
    cca.RPL.gamma_o3 = cca.gamma_o3 ./ (cca.delta_o3 + cca.theta_o3 + cca.alpha_o3 + cca.beta_o3 + cca.gamma_o3) ;
    avg.RPL.gamma_o3 = avg.gamma_o3 ./ sum(avg.Regional_o3,2) ;     

    cca.RPL.Regional_o3 = [cca.RPL.delta_o3 cca.RPL.theta_o3 cca.RPL.alpha_o3 cca.RPL.beta_o3 cca.RPL.gamma_o3]; 
    avg.RPL.Regional_o3 = [avg.RPL.delta_o3 avg.RPL.theta_o3 avg.RPL.alpha_o3 avg.RPL.beta_o3 avg.RPL.gamma_o3];
        
    cca.z.Regional_o3 = [zscore(cca.delta_o3) zscore(cca.theta_o3) zscore(cca.alpha_o3) zscore(cca.beta_o3) zscore(cca.gamma_o3)];
    avg.z.Regional_o3 = [zscore(avg.delta_o3) zscore(avg.theta_o3) zscore(avg.alpha_o3) zscore(avg.beta_o3) zscore(avg.gamma_o3)];                   
    
    
    %% Statistical Analysis
%         lowIdx = find(kss==min(kss)):1:find(kss==min(kss))+200;
%         lowIdx = lowIdx';
%         highIdx = find(kss==max(kss))-200:1:find(kss==max(kss));
%         highIdx = highIdx';
    
    %         최소+1, 최대-1
    lowIdx = min(kss)<kss & min(kss)+1>kss;
    highIdx = max(kss)>kss & max(kss)-1<kss;
    
    % PSD T-TEST 분석
    [h, p, ci, stats] = ttest2(avg.delta(lowIdx), avg.delta(highIdx));
    p_delta(s) = p;
    [h, p, ci, stats] = ttest2(avg.theta(lowIdx), avg.theta(highIdx));
    p_theta(s) = p;
    [h, p, ci, stats] = ttest2(avg.alpha(lowIdx), avg.alpha(highIdx));
    p_alpha(s) = p;
    [h, p, ci, stats] = ttest2(avg.beta(lowIdx), avg.beta(highIdx));
    p_beta(s) = p;
    [h, p, ci, stats] = ttest2(avg.gamma(lowIdx), avg.gamma(highIdx));
    p_gamma(s) = p;
    
    p_analysis(s,1:5) = [p_delta(s), p_theta(s), p_alpha(s), p_beta(s), p_gamma(s)];
    p_analysis_min(s) = find(p_analysis(s,1:5)==min(p_analysis(s,1:5)));
    
    Result.ttest.total(s,:) = p_analysis(s,1:5);
    
    % RPL T-TEST 분석
    [h, p, ci, stats] = ttest2(avg.RPL.delta(lowIdx), avg.RPL.delta(highIdx));
    p_delta(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.theta(lowIdx), avg.RPL.theta(highIdx));
    p_theta(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.alpha(lowIdx), avg.RPL.alpha(highIdx));
    p_alpha(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.beta(lowIdx), avg.RPL.beta(highIdx));
    p_beta(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.gamma(lowIdx), avg.RPL.gamma(highIdx));
    p_gamma(s) = p;
    
    p_analysis_RPL(s,1:5) = [p_delta(s), p_theta(s), p_alpha(s), p_beta(s), p_gamma(s)];
    p_analysis_RPL_min(s) = find(p_analysis_RPL(s,1:5)==min(p_analysis_RPL(s,1:5)));
    
    Result.ttest.total_RPL(s,:) = p_analysis_RPL(s,1:5);    
    
    
    % z-score T-TEST 분석
    [h, p, ci, stats] = ttest2(avg.z.PSD(lowIdx,1), avg.z.PSD(highIdx,1));
    p_delta(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.PSD(lowIdx,2), avg.z.PSD(lowIdx,2));
    p_theta(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.PSD(lowIdx,3), avg.z.PSD(lowIdx,3));
    p_alpha(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.PSD(lowIdx,4), avg.z.PSD(lowIdx,4));
    p_beta(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.PSD(lowIdx,5), avg.z.PSD(lowIdx,5));
    p_gamma(s) = p;
    
    p_analysis_z(s,1:5) = [p_delta(s), p_theta(s), p_alpha(s), p_beta(s), p_gamma(s)];
    p_analysis_z_min(s) = find(p_analysis_z(s,1:5)==min(p_analysis_z(s,1:5)));
    
    Result.ttest.total_z(s,:) = p_analysis_z(s,1:5);    
    
    
    
    % Regional PSD T-TEST 분석
    [h, p, ci, stats] = ttest2(avg.delta_f(lowIdx), avg.delta_f(highIdx));
    p_delta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.theta_f(lowIdx), avg.theta_f(highIdx));
    p_theta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.alpha_f(lowIdx), avg.alpha_f(highIdx));
    p_alpha_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.beta_f(lowIdx), avg.beta_f(highIdx));
    p_beta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.gamma_f(lowIdx), avg.gamma_f(highIdx));
    p_gamma_f(s) = p;
    
    p_analysis_f(s,1:5) = [p_delta_f(s), p_theta_f(s), p_alpha_f(s), p_beta_f(s), p_gamma_f(s)];
    p_analysis_f_min(s) = find(p_analysis_f(s,1:5)==min(p_analysis_f(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.delta_c(lowIdx), avg.delta_c(highIdx));
    p_delta_c(s) = p;
    [h, p, ci, stats] = ttest2(avg.theta_c(lowIdx), avg.theta_c(highIdx));
    p_theta_c(s) = p;
    [h, p, ci, stats] = ttest2(avg.alpha_c(lowIdx), avg.alpha_c(highIdx));
    p_alpha_c(s) = p;
    [h, p, ci, stats] = ttest2(avg.beta_c(lowIdx), avg.beta_c(highIdx));
    p_beta_c(s) = p;
    [h, p, ci, stats] = ttest2(avg.gamma_c(lowIdx), avg.gamma_c(highIdx));
    p_gamma_c(s) = p;
    
    p_analysis_c(s,1:5) = [p_delta_c(s), p_theta_c(s), p_alpha_c(s), p_beta_c(s), p_gamma_c(s)];
    p_analysis_c_min(s) = find(p_analysis_c(s,1:5)==min(p_analysis_c(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.delta_p(lowIdx), avg.delta_p(highIdx));
    p_delta_p(s) = p;
    [h, p, ci, stats] = ttest2(avg.theta_p(lowIdx), avg.theta_p(highIdx));
    p_theta_p(s) = p;
    [h, p, ci, stats] = ttest2(avg.alpha_p(lowIdx), avg.alpha_p(highIdx));
    p_alpha_p(s) = p;
    [h, p, ci, stats] = ttest2(avg.beta_p(lowIdx), avg.beta_p(highIdx));
    p_beta_p(s) = p;
    [h, p, ci, stats] = ttest2(avg.gamma_p(lowIdx), avg.gamma_p(highIdx));
    p_gamma_p(s) = p;
    
    p_analysis_p(s,1:5) = [p_delta_p(s), p_theta_p(s), p_alpha_p(s), p_beta_p(s), p_gamma_p(s)];
    p_analysis_p_min(s) = find(p_analysis_p(s,1:5)==min(p_analysis_p(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.delta_t(lowIdx), avg.delta_t(highIdx));
    p_delta_t(s) = p;
    [h, p, ci, stats] = ttest2(avg.theta_t(lowIdx), avg.theta_t(highIdx));
    p_theta_t(s) = p;
    [h, p, ci, stats] = ttest2(avg.alpha_t(lowIdx), avg.alpha_t(highIdx));
    p_alpha_t(s) = p;
    [h, p, ci, stats] = ttest2(avg.beta_t(lowIdx), avg.beta_t(highIdx));
    p_beta_t(s) = p;
    [h, p, ci, stats] = ttest2(avg.gamma_t(lowIdx), avg.gamma_t(highIdx));
    p_gamma_t(s) = p;
    
    p_analysis_t(s,1:5) = [p_delta_t(s), p_theta_t(s), p_alpha_t(s), p_beta_t(s), p_gamma_t(s)];
    p_analysis_t_min(s) = find(p_analysis_t(s,1:5)==min(p_analysis_t(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.delta_o(lowIdx), avg.delta_o(highIdx));
    p_delta_o(s) = p;
    [h, p, ci, stats] = ttest2(avg.theta_o(lowIdx), avg.theta_o(highIdx));
    p_theta_o(s) = p;
    [h, p, ci, stats] = ttest2(avg.alpha_o(lowIdx), avg.alpha_o(highIdx));
    p_alpha_o(s) = p;
    [h, p, ci, stats] = ttest2(avg.beta_o(lowIdx), avg.beta_o(highIdx));
    p_beta_o(s) = p;
    [h, p, ci, stats] = ttest2(avg.gamma_o(lowIdx), avg.gamma_o(highIdx));
    p_gamma_o(s) = p;
    
    p_analysis_o(s,1:5) = [p_delta_o(s), p_theta_o(s), p_alpha_o(s), p_beta_o(s), p_gamma_o(s)];
    p_analysis_o_min(s) = find(p_analysis_o(s,1:5)==min(p_analysis_o(s,1:5)));
    
    Result.ttest.regional(s,:) = [p_analysis_f(s,:) p_analysis_c(s,:) p_analysis_p(s,:) p_analysis_t(s,:) p_analysis_o(s,:)];

    
    
    [h, p, ci, stats] = ttest2(avg.RPL.delta_f(lowIdx), avg.RPL.delta_f(highIdx));
    p_delta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.theta_f(lowIdx), avg.RPL.theta_f(highIdx));
    p_theta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.alpha_f(lowIdx), avg.RPL.alpha_f(highIdx));
    p_alpha_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.beta_f(lowIdx), avg.RPL.beta_f(highIdx));
    p_beta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.gamma_f(lowIdx), avg.RPL.gamma_f(highIdx));
    p_gamma_f(s) = p;
    
    p_analysis_RPL_f(s,1:5) = [p_delta_f(s), p_theta_f(s), p_alpha_f(s), p_beta_f(s), p_gamma_f(s)];
    p_analysis_RPL_f_min(s) = find(p_analysis_RPL_f(s,1:5)==min(p_analysis_RPL_f(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.RPL.delta_c(lowIdx), avg.RPL.delta_c(highIdx));
    p_delta_c(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.theta_c(lowIdx), avg.RPL.theta_c(highIdx));
    p_theta_c(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.alpha_c(lowIdx), avg.RPL.alpha_c(highIdx));
    p_alpha_c(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.beta_c(lowIdx), avg.RPL.beta_c(highIdx));
    p_beta_c(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.gamma_c(lowIdx), avg.RPL.gamma_c(highIdx));
    p_gamma_c(s) = p;
    
    p_analysis_RPL_c(s,1:5) = [p_delta_c(s), p_theta_c(s), p_alpha_c(s), p_beta_c(s), p_gamma_c(s)];
    p_analysis_RPL_c_min(s) = find(p_analysis_RPL_c(s,1:5)==min(p_analysis_RPL_c(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.RPL.delta_p(lowIdx), avg.RPL.delta_p(highIdx));
    p_delta_p(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.theta_p(lowIdx), avg.RPL.theta_p(highIdx));
    p_theta_p(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.alpha_p(lowIdx), avg.RPL.alpha_p(highIdx));
    p_alpha_p(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.beta_p(lowIdx), avg.RPL.beta_p(highIdx));
    p_beta_p(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.gamma_p(lowIdx), avg.RPL.gamma_p(highIdx));
    p_gamma_p(s) = p;
    
    p_analysis_RPL_p(s,1:5) = [p_delta_p(s), p_theta_p(s), p_alpha_p(s), p_beta_p(s), p_gamma_p(s)];
    p_analysis_RPL_p_min(s) = find(p_analysis_RPL_p(s,1:5)==min(p_analysis_RPL_p(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.RPL.delta_t(lowIdx), avg.RPL.delta_t(highIdx));
    p_delta_t(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.theta_t(lowIdx), avg.RPL.theta_t(highIdx));
    p_theta_t(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.alpha_t(lowIdx), avg.RPL.alpha_t(highIdx));
    p_alpha_t(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.beta_t(lowIdx), avg.RPL.beta_t(highIdx));
    p_beta_t(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.gamma_t(lowIdx), avg.RPL.gamma_t(highIdx));
    p_gamma_t(s) = p;
    
    p_analysis_RPL_t(s,1:5) = [p_delta_t(s), p_theta_t(s), p_alpha_t(s), p_beta_t(s), p_gamma_t(s)];
    p_analysis_RPL_t_min(s) = find(p_analysis_RPL_t(s,1:5)==min(p_analysis_RPL_t(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.RPL.delta_o(lowIdx), avg.RPL.delta_o(highIdx));
    p_delta_o(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.theta_o(lowIdx), avg.RPL.theta_o(highIdx));
    p_theta_o(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.alpha_o(lowIdx), avg.RPL.alpha_o(highIdx));
    p_alpha_o(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.beta_o(lowIdx), avg.RPL.beta_o(highIdx));
    p_beta_o(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.gamma_o(lowIdx), avg.RPL.gamma_o(highIdx));
    p_gamma_o(s) = p;
    
    p_analysis_RPL_o(s,1:5) = [p_delta_o(s), p_theta_o(s), p_alpha_o(s), p_beta_o(s), p_gamma_o(s)];
    p_analysis_RPL_o_min(s) = find(p_analysis_RPL_o(s,1:5)==min(p_analysis_RPL_o(s,1:5)));
    
    Result.ttest.regional_RPL(s,:) = [p_analysis_RPL_f(s,:) p_analysis_RPL_c(s,:) p_analysis_RPL_p(s,:) p_analysis_RPL_t(s,:) p_analysis_RPL_o(s,:)];

    
    
    [h, p, ci, stats] = ttest2(avg.z.Regional_f(lowIdx,1), avg.z.Regional_f(highIdx,1));
    p_delta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_f(lowIdx,2), avg.z.Regional_f(highIdx,2));
    p_theta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_f(lowIdx,3), avg.z.Regional_f(highIdx,3));
    p_alpha_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_f(lowIdx,4), avg.z.Regional_f(highIdx,4));
    p_beta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_f(lowIdx,5), avg.z.Regional_f(highIdx,5));
    p_gamma_f(s) = p;
    
    p_analysis_z_f(s,1:5) = [p_delta_f(s), p_theta_f(s), p_alpha_f(s), p_beta_f(s), p_gamma_f(s)];
    p_analysis_z_f_min(s) = find(p_analysis_z_f(s,1:5)==min(p_analysis_z_f(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.z.Regional_c(lowIdx,1), avg.z.Regional_c(highIdx,1));
    p_delta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_c(lowIdx,2), avg.z.Regional_c(highIdx,2));
    p_theta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_c(lowIdx,3), avg.z.Regional_c(highIdx,3));
    p_alpha_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_c(lowIdx,4), avg.z.Regional_c(highIdx,4));
    p_beta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_c(lowIdx,5), avg.z.Regional_c(highIdx,5));
    p_gamma_f(s) = p;
    
    p_analysis_z_c(s,1:5) = [p_delta_c(s), p_theta_c(s), p_alpha_c(s), p_beta_c(s), p_gamma_c(s)];
    p_analysis_z_c_min(s) = find(p_analysis_z_c(s,1:5)==min(p_analysis_z_c(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.z.Regional_p(lowIdx,1), avg.z.Regional_p(highIdx,1));
    p_delta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_p(lowIdx,2), avg.z.Regional_p(highIdx,2));
    p_theta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_p(lowIdx,3), avg.z.Regional_p(highIdx,3));
    p_alpha_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_p(lowIdx,4), avg.z.Regional_p(highIdx,4));
    p_beta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_p(lowIdx,5), avg.z.Regional_p(highIdx,5));
    p_gamma_f(s) = p;
    
    p_analysis_z_p(s,1:5) = [p_delta_p(s), p_theta_p(s), p_alpha_p(s), p_beta_p(s), p_gamma_p(s)];
    p_analysis_z_p_min(s) = find(p_analysis_z_p(s,1:5)==min(p_analysis_z_p(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.z.Regional_t(lowIdx,1), avg.z.Regional_t(highIdx,1));
    p_delta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_t(lowIdx,2), avg.z.Regional_t(highIdx,2));
    p_theta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_t(lowIdx,3), avg.z.Regional_t(highIdx,3));
    p_alpha_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_t(lowIdx,4), avg.z.Regional_t(highIdx,4));
    p_beta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_t(lowIdx,5), avg.z.Regional_t(highIdx,5));
    p_gamma_f(s) = p;
    
    p_analysis_z_t(s,1:5) = [p_delta_t(s), p_theta_t(s), p_alpha_t(s), p_beta_t(s), p_gamma_t(s)];
    p_analysis_z_t_min(s) = find(p_analysis_z_t(s,1:5)==min(p_analysis_z_t(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.z.Regional_o(lowIdx,1), avg.z.Regional_o(highIdx,1));
    p_delta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_o(lowIdx,2), avg.z.Regional_o(highIdx,2));
    p_theta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_o(lowIdx,3), avg.z.Regional_o(highIdx,3));
    p_alpha_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_o(lowIdx,4), avg.z.Regional_o(highIdx,4));
    p_beta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_o(lowIdx,5), avg.z.Regional_o(highIdx,5));
    p_gamma_f(s) = p;
    
    p_analysis_z_o(s,1:5) = [p_delta_o(s), p_theta_o(s), p_alpha_o(s), p_beta_o(s), p_gamma_o(s)];
    p_analysis_z_o_min(s) = find(p_analysis_z_o(s,1:5)==min(p_analysis_z_o(s,1:5)));
    
    Result.ttest.regional_z(s,:) = [p_analysis_f(s,:) p_analysis_c(s,:) p_analysis_p(s,:) p_analysis_t(s,:) p_analysis_o(s,:)];

    
    
    
    
    
    % Regional Regional T-TEST 분석
    [h, p, ci, stats] = ttest2(avg.delta_f1(lowIdx), avg.delta_f1(highIdx));
    p_delta_f1(s) = p;
    [h, p, ci, stats] = ttest2(avg.theta_f1(lowIdx), avg.theta_f1(highIdx));
    p_theta_f1(s) = p;
    [h, p, ci, stats] = ttest2(avg.alpha_f1(lowIdx), avg.alpha_f1(highIdx));
    p_alpha_f1(s) = p;
    [h, p, ci, stats] = ttest2(avg.beta_f1(lowIdx), avg.beta_f1(highIdx));
    p_beta_f1(s) = p;
    [h, p, ci, stats] = ttest2(avg.gamma_f1(lowIdx), avg.gamma_f1(highIdx));
    p_gamma_f1(s) = p;
    
    p_analysis_f1(s,1:5) = [p_delta_f1(s), p_theta_f1(s), p_alpha_f1(s), p_beta_f1(s), p_gamma_f1(s)];
    p_analysis_f1_min(s) = find(p_analysis_f1(s,1:5)==min(p_analysis_f1(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.delta_f2(lowIdx), avg.delta_f2(highIdx));
    p_delta_f2(s) = p;
    [h, p, ci, stats] = ttest2(avg.theta_f2(lowIdx), avg.theta_f2(highIdx));
    p_theta_f2(s) = p;
    [h, p, ci, stats] = ttest2(avg.alpha_f2(lowIdx), avg.alpha_f2(highIdx));
    p_alpha_f2(s) = p;
    [h, p, ci, stats] = ttest2(avg.beta_f2(lowIdx), avg.beta_f2(highIdx));
    p_beta_f2(s) = p;
    [h, p, ci, stats] = ttest2(avg.gamma_f2(lowIdx), avg.gamma_f2(highIdx));
    p_gamma_f2(s) = p;
    
    p_analysis_f2(s,1:5) = [p_delta_f2(s), p_theta_f2(s), p_alpha_f2(s), p_beta_f2(s), p_gamma_f2(s)];
    p_analysis_f2_min(s) = find(p_analysis_f2(s,1:5)==min(p_analysis_f2(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.delta_f3(lowIdx), avg.delta_f3(highIdx));
    p_delta_f3(s) = p;
    [h, p, ci, stats] = ttest2(avg.theta_f3(lowIdx), avg.theta_f3(highIdx));
    p_theta_f3(s) = p;
    [h, p, ci, stats] = ttest2(avg.alpha_f3(lowIdx), avg.alpha_f3(highIdx));
    p_alpha_f3(s) = p;
    [h, p, ci, stats] = ttest2(avg.beta_f3(lowIdx), avg.beta_f3(highIdx));
    p_beta_f3(s) = p;
    [h, p, ci, stats] = ttest2(avg.gamma_f3(lowIdx), avg.gamma_f3(highIdx));
    p_gamma_f3(s) = p;
    
    p_analysis_f3(s,1:5) = [p_delta_f3(s), p_theta_f3(s), p_alpha_f3(s), p_beta_f3(s), p_gamma_f3(s)];
    p_analysis_f3_min(s) = find(p_analysis_f3(s,1:5)==min(p_analysis_f3(s,1:5)));
    
    
    
    [h, p, ci, stats] = ttest2(avg.delta_c1(lowIdx), avg.delta_c1(highIdx));
    p_delta_c1(s) = p;
    [h, p, ci, stats] = ttest2(avg.theta_c1(lowIdx), avg.theta_c1(highIdx));
    p_theta_c1(s) = p;
    [h, p, ci, stats] = ttest2(avg.alpha_c1(lowIdx), avg.alpha_c1(highIdx));
    p_alpha_c1(s) = p;
    [h, p, ci, stats] = ttest2(avg.beta_c1(lowIdx), avg.beta_c1(highIdx));
    p_beta_c1(s) = p;
    [h, p, ci, stats] = ttest2(avg.gamma_c1(lowIdx), avg.gamma_c1(highIdx));
    p_gamma_c1(s) = p;
    
    p_analysis_c1(s,1:5) = [p_delta_c1(s), p_theta_c1(s), p_alpha_c1(s), p_beta_c1(s), p_gamma_c1(s)];
    p_analysis_c1_min(s) = find(p_analysis_c1(s,1:5)==min(p_analysis_c1(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.delta_c2(lowIdx), avg.delta_c2(highIdx));
    p_delta_c2(s) = p;
    [h, p, ci, stats] = ttest2(avg.theta_c2(lowIdx), avg.theta_c2(highIdx));
    p_theta_c2(s) = p;
    [h, p, ci, stats] = ttest2(avg.alpha_c2(lowIdx), avg.alpha_c2(highIdx));
    p_alpha_c2(s) = p;
    [h, p, ci, stats] = ttest2(avg.beta_c2(lowIdx), avg.beta_c2(highIdx));
    p_beta_c2(s) = p;
    [h, p, ci, stats] = ttest2(avg.gamma_c2(lowIdx), avg.gamma_c2(highIdx));
    p_gamma_c2(s) = p;
    
    p_analysis_c2(s,1:5) = [p_delta_c2(s), p_theta_c2(s), p_alpha_c2(s), p_beta_c2(s), p_gamma_c2(s)];
    p_analysis_c2_min(s) = find(p_analysis_c2(s,1:5)==min(p_analysis_c2(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.delta_c3(lowIdx), avg.delta_c3(highIdx));
    p_delta_c3(s) = p;
    [h, p, ci, stats] = ttest2(avg.theta_c3(lowIdx), avg.theta_c3(highIdx));
    p_theta_c3(s) = p;
    [h, p, ci, stats] = ttest2(avg.alpha_c3(lowIdx), avg.alpha_c3(highIdx));
    p_alpha_c3(s) = p;
    [h, p, ci, stats] = ttest2(avg.beta_c3(lowIdx), avg.beta_c3(highIdx));
    p_beta_c3(s) = p;
    [h, p, ci, stats] = ttest2(avg.gamma_c3(lowIdx), avg.gamma_c3(highIdx));
    p_gamma_c3(s) = p;
    
    p_analysis_c3(s,1:5) = [p_delta_c3(s), p_theta_c3(s), p_alpha_c3(s), p_beta_c3(s), p_gamma_c3(s)];
    p_analysis_c3_min(s) = find(p_analysis_c3(s,1:5)==min(p_analysis_c3(s,1:5)));
    
    

    
    
    [h, p, ci, stats] = ttest2(avg.delta_p1(lowIdx), avg.delta_p1(highIdx));
    p_delta_p1(s) = p;
    [h, p, ci, stats] = ttest2(avg.theta_p1(lowIdx), avg.theta_p1(highIdx));
    p_theta_p1(s) = p;
    [h, p, ci, stats] = ttest2(avg.alpha_p1(lowIdx), avg.alpha_p1(highIdx));
    p_alpha_p1(s) = p;
    [h, p, ci, stats] = ttest2(avg.beta_p1(lowIdx), avg.beta_p1(highIdx));
    p_beta_p1(s) = p;
    [h, p, ci, stats] = ttest2(avg.gamma_p1(lowIdx), avg.gamma_p1(highIdx));
    p_gamma_p1(s) = p;
    
    p_analysis_p1(s,1:5) = [p_delta_p1(s), p_theta_p1(s), p_alpha_p1(s), p_beta_p1(s), p_gamma_p1(s)];
    p_analysis_p1_min(s) = find(p_analysis_p1(s,1:5)==min(p_analysis_p1(s,1:5)));
    
    
    
    [h, p, ci, stats] = ttest2(avg.delta_p2(lowIdx), avg.delta_p2(highIdx));
    p_delta_p2(s) = p;
    [h, p, ci, stats] = ttest2(avg.theta_p2(lowIdx), avg.theta_p2(highIdx));
    p_theta_p2(s) = p;
    [h, p, ci, stats] = ttest2(avg.alpha_p2(lowIdx), avg.alpha_p2(highIdx));
    p_alpha_p2(s) = p;
    [h, p, ci, stats] = ttest2(avg.beta_p2(lowIdx), avg.beta_p2(highIdx));
    p_beta_p2(s) = p;
    [h, p, ci, stats] = ttest2(avg.gamma_p2(lowIdx), avg.gamma_p2(highIdx));
    p_gamma_p2(s) = p;
    
    p_analysis_p2(s,1:5) = [p_delta_p2(s), p_theta_p2(s), p_alpha_p2(s), p_beta_p2(s), p_gamma_p2(s)];
    p_analysis_p2_min(s) = find(p_analysis_p2(s,1:5)==min(p_analysis_p2(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.delta_p3(lowIdx), avg.delta_p3(highIdx));
    p_delta_p3(s) = p;
    [h, p, ci, stats] = ttest2(avg.theta_p3(lowIdx), avg.theta_p3(highIdx));
    p_theta_p3(s) = p;
    [h, p, ci, stats] = ttest2(avg.alpha_p3(lowIdx), avg.alpha_p3(highIdx));
    p_alpha_p3(s) = p;
    [h, p, ci, stats] = ttest2(avg.beta_p3(lowIdx), avg.beta_p3(highIdx));
    p_beta_p3(s) = p;
    [h, p, ci, stats] = ttest2(avg.gamma_p3(lowIdx), avg.gamma_p3(highIdx));
    p_gamma_p3(s) = p;
    
    p_analysis_p3(s,1:5) = [p_delta_p3(s), p_theta_p3(s), p_alpha_p3(s), p_beta_p3(s), p_gamma_p3(s)];
    p_analysis_p3_min(s) = find(p_analysis_p3(s,1:5)==min(p_analysis_p3(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.delta_t1(lowIdx), avg.delta_t1(highIdx));
    p_delta_t1(s) = p;
    [h, p, ci, stats] = ttest2(avg.theta_t1(lowIdx), avg.theta_t1(highIdx));
    p_theta_t1(s) = p;
    [h, p, ci, stats] = ttest2(avg.alpha_t1(lowIdx), avg.alpha_t1(highIdx));
    p_alpha_t1(s) = p;
    [h, p, ci, stats] = ttest2(avg.beta_t1(lowIdx), avg.beta_t1(highIdx));
    p_beta_t1(s) = p;
    [h, p, ci, stats] = ttest2(avg.gamma_t1(lowIdx), avg.gamma_t1(highIdx));
    p_gamma_t1(s) = p;
    
    p_analysis_t1(s,1:5) = [p_delta_t1(s), p_theta_t1(s), p_alpha_t1(s), p_beta_t1(s), p_gamma_t1(s)];
    p_analysis_t1_min(s) = find(p_analysis_t1(s,1:5)==min(p_analysis_t1(s,1:5)));
    
    [h, p, ci, stats] = ttest2(avg.delta_t2(lowIdx), avg.delta_t2(highIdx));
    p_delta_t2(s) = p;
    [h, p, ci, stats] = ttest2(avg.theta_t2(lowIdx), avg.theta_t2(highIdx));
    p_theta_t2(s) = p;
    [h, p, ci, stats] = ttest2(avg.alpha_t2(lowIdx), avg.alpha_t2(highIdx));
    p_alpha_t2(s) = p;
    [h, p, ci, stats] = ttest2(avg.beta_t2(lowIdx), avg.beta_t2(highIdx));
    p_beta_t2(s) = p;
    [h, p, ci, stats] = ttest2(avg.gamma_t2(lowIdx), avg.gamma_t2(highIdx));
    p_gamma_t2(s) = p;
    
    p_analysis_t2(s,1:5) = [p_delta_t2(s), p_theta_t2(s), p_alpha_t2(s), p_beta_t2(s), p_gamma_t2(s)];
    p_analysis_t2_min(s) = find(p_analysis_t2(s,1:5)==min(p_analysis_t2(s,1:5)));
    
    
    
    [h, p, ci, stats] = ttest2(avg.delta_o1(lowIdx), avg.delta_o1(highIdx));
    p_delta_o1(s) = p;
    [h, p, ci, stats] = ttest2(avg.theta_o1(lowIdx), avg.theta_o1(highIdx));
    p_theta_o1(s) = p;
    [h, p, ci, stats] = ttest2(avg.alpha_o1(lowIdx), avg.alpha_o1(highIdx));
    p_alpha_o1(s) = p;
    [h, p, ci, stats] = ttest2(avg.beta_o1(lowIdx), avg.beta_o1(highIdx));
    p_beta_o1(s) = p;
    [h, p, ci, stats] = ttest2(avg.gamma_o1(lowIdx), avg.gamma_o1(highIdx));
    p_gamma_o1(s) = p;
    
    p_analysis_o1(s,1:5) = [p_delta_o1(s), p_theta_o1(s), p_alpha_o1(s), p_beta_o1(s), p_gamma_o1(s)];
    p_analysis_o1_min(s) = find(p_analysis_o1(s,1:5)==min(p_analysis_o1(s,1:5)));
    
    [h, p, ci, stats] = ttest2(avg.delta_o2(lowIdx), avg.delta_o2(highIdx));
    p_delta_o2(s) = p;
    [h, p, ci, stats] = ttest2(avg.theta_o2(lowIdx), avg.theta_o2(highIdx));
    p_theta_o2(s) = p;
    [h, p, ci, stats] = ttest2(avg.alpha_o2(lowIdx), avg.alpha_o2(highIdx));
    p_alpha_o2(s) = p;
    [h, p, ci, stats] = ttest2(avg.beta_o2(lowIdx), avg.beta_o2(highIdx));
    p_beta_o2(s) = p;
    [h, p, ci, stats] = ttest2(avg.gamma_o2(lowIdx), avg.gamma_o2(highIdx));
    p_gamma_o2(s) = p;
    
    p_analysis_o2(s,1:5) = [p_delta_o2(s), p_theta_o2(s), p_alpha_o2(s), p_beta_o2(s), p_gamma_o2(s)];
    p_analysis_o2_min(s) = find(p_analysis_o2(s,1:5)==min(p_analysis_o2(s,1:5)));
    
    [h, p, ci, stats] = ttest2(avg.delta_o3(lowIdx), avg.delta_o3(highIdx));
    p_delta_o3(s) = p;
    [h, p, ci, stats] = ttest2(avg.theta_o3(lowIdx), avg.theta_o3(highIdx));
    p_theta_o3(s) = p;
    [h, p, ci, stats] = ttest2(avg.alpha_o3(lowIdx), avg.alpha_o3(highIdx));
    p_alpha_o3(s) = p;
    [h, p, ci, stats] = ttest2(avg.beta_o3(lowIdx), avg.beta_o3(highIdx));
    p_beta_o3(s) = p;
    [h, p, ci, stats] = ttest2(avg.gamma_o3(lowIdx), avg.gamma_o3(highIdx));
    p_gamma_o3(s) = p;
    
    p_analysis_o3(s,1:5) = [p_delta_o3(s), p_theta_o3(s), p_alpha_o3(s), p_beta_o3(s), p_gamma_o3(s)];
    p_analysis_o3_min(s) = find(p_analysis_o3(s,1:5)==min(p_analysis_o3(s,1:5)));
    
    Result.ttest.vertical(s,:) = [p_analysis_f1(s,:) p_analysis_f2(s,:) p_analysis_f3(s,:) p_analysis_c1(s,:) p_analysis_c2(s,:) p_analysis_c3(s,:) p_analysis_p1(s,:) p_analysis_p2(s,:) p_analysis_p3(s,:) p_analysis_t1(s,:) p_analysis_t2(s,:) p_analysis_o1(s,:) p_analysis_o2(s,:) p_analysis_o3(s,:)];
    
    
    
        % Regional Vertical T-TEST 분석
    [h, p, ci, stats] = ttest2(avg.RPL.delta_f1(lowIdx), avg.RPL.delta_f1(highIdx));
    p_delta_f1(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.theta_f1(lowIdx), avg.RPL.theta_f1(highIdx));
    p_theta_f1(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.alpha_f1(lowIdx), avg.RPL.alpha_f1(highIdx));
    p_alpha_f1(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.beta_f1(lowIdx), avg.RPL.beta_f1(highIdx));
    p_beta_f1(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.gamma_f1(lowIdx), avg.RPL.gamma_f1(highIdx));
    p_gamma_f1(s) = p;
    
    p_analysis_f1_rpl(s,1:5) = [p_delta_f1(s), p_theta_f1(s), p_alpha_f1(s), p_beta_f1(s), p_gamma_f1(s)];
    p_analysis_f1_rpl_min(s) = find(p_analysis_f1_rpl(s,1:5)==min(p_analysis_f1_rpl(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.RPL.delta_f2(lowIdx), avg.RPL.delta_f2(highIdx));
    p_delta_f2(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.theta_f2(lowIdx), avg.RPL.theta_f2(highIdx));
    p_theta_f2(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.alpha_f2(lowIdx), avg.RPL.alpha_f2(highIdx));
    p_alpha_f2(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.beta_f2(lowIdx), avg.RPL.beta_f2(highIdx));
    p_beta_f2(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.gamma_f2(lowIdx), avg.RPL.gamma_f2(highIdx));
    p_gamma_f2(s) = p;
    
    p_analysis_f2_rpl(s,1:5) = [p_delta_f2(s), p_theta_f2(s), p_alpha_f2(s), p_beta_f2(s), p_gamma_f2(s)];
    p_analysis_f2_rpl_min(s) = find(p_analysis_f2_rpl(s,1:5)==min(p_analysis_f2_rpl(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.RPL.delta_f3(lowIdx), avg.RPL.delta_f3(highIdx));
    p_delta_f3(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.theta_f3(lowIdx), avg.RPL.theta_f3(highIdx));
    p_theta_f3(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.alpha_f3(lowIdx), avg.RPL.alpha_f3(highIdx));
    p_alpha_f3(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.beta_f3(lowIdx), avg.RPL.beta_f3(highIdx));
    p_beta_f3(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.gamma_f3(lowIdx), avg.RPL.gamma_f3(highIdx));
    p_gamma_f3(s) = p;
    
    p_analysis_f3_rpl(s,1:5) = [p_delta_f3(s), p_theta_f3(s), p_alpha_f3(s), p_beta_f3(s), p_gamma_f3(s)];
    p_analysis_f3_rpl_min(s) = find(p_analysis_f3_rpl(s,1:5)==min(p_analysis_f3_rpl(s,1:5)));
    
    
    
    [h, p, ci, stats] = ttest2(avg.RPL.delta_c1(lowIdx), avg.RPL.delta_c1(highIdx));
    p_delta_c1(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.theta_c1(lowIdx), avg.RPL.theta_c1(highIdx));
    p_theta_c1(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.alpha_c1(lowIdx), avg.RPL.alpha_c1(highIdx));
    p_alpha_c1(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.beta_c1(lowIdx), avg.RPL.beta_c1(highIdx));
    p_beta_c1(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.gamma_c1(lowIdx), avg.RPL.gamma_c1(highIdx));
    p_gamma_c1(s) = p;
    
    p_analysis_c1_rpl(s,1:5) = [p_delta_c1(s), p_theta_c1(s), p_alpha_c1(s), p_beta_c1(s), p_gamma_c1(s)];
    p_analysis_c1_rpl_min(s) = find(p_analysis_c1_rpl(s,1:5)==min(p_analysis_c1_rpl(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.RPL.delta_c2(lowIdx), avg.RPL.delta_c2(highIdx));
    p_delta_c2(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.theta_c2(lowIdx), avg.RPL.theta_c2(highIdx));
    p_theta_c2(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.alpha_c2(lowIdx), avg.RPL.alpha_c2(highIdx));
    p_alpha_c2(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.beta_c2(lowIdx), avg.RPL.beta_c2(highIdx));
    p_beta_c2(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.gamma_c2(lowIdx), avg.RPL.gamma_c2(highIdx));
    p_gamma_c2(s) = p;
    
    p_analysis_c2_rpl(s,1:5) = [p_delta_c2(s), p_theta_c2(s), p_alpha_c2(s), p_beta_c2(s), p_gamma_c2(s)];
    p_analysis_c2_rpl_min(s) = find(p_analysis_c2_rpl(s,1:5)==min(p_analysis_c2_rpl(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.RPL.delta_c3(lowIdx), avg.RPL.delta_c3(highIdx));
    p_delta_c3(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.theta_c3(lowIdx), avg.RPL.theta_c3(highIdx));
    p_theta_c3(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.alpha_c3(lowIdx), avg.RPL.alpha_c3(highIdx));
    p_alpha_c3(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.beta_c3(lowIdx), avg.RPL.beta_c3(highIdx));
    p_beta_c3(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.gamma_c3(lowIdx), avg.RPL.gamma_c3(highIdx));
    p_gamma_c3(s) = p;
    
    p_analysis_c3_rpl(s,1:5) = [p_delta_c3(s), p_theta_c3(s), p_alpha_c3(s), p_beta_c3(s), p_gamma_c3(s)];
    p_analysis_c3_rpl_min(s) = find(p_analysis_c3_rpl(s,1:5)==min(p_analysis_c3_rpl(s,1:5)));
    
    

    
    
    [h, p, ci, stats] = ttest2(avg.RPL.delta_p1(lowIdx), avg.RPL.delta_p1(highIdx));
    p_delta_p1(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.theta_p1(lowIdx), avg.RPL.theta_p1(highIdx));
    p_theta_p1(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.alpha_p1(lowIdx), avg.RPL.alpha_p1(highIdx));
    p_alpha_p1(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.beta_p1(lowIdx), avg.RPL.beta_p1(highIdx));
    p_beta_p1(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.gamma_p1(lowIdx), avg.RPL.gamma_p1(highIdx));
    p_gamma_p1(s) = p;
    
    p_analysis_p1_rpl(s,1:5) = [p_delta_p1(s), p_theta_p1(s), p_alpha_p1(s), p_beta_p1(s), p_gamma_p1(s)];
    p_analysis_p1_rpl_min(s) = find(p_analysis_p1_rpl(s,1:5)==min(p_analysis_p1_rpl(s,1:5)));
    
    
    
    [h, p, ci, stats] = ttest2(avg.RPL.delta_p2(lowIdx), avg.RPL.delta_p2(highIdx));
    p_delta_p2(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.theta_p2(lowIdx), avg.RPL.theta_p2(highIdx));
    p_theta_p2(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.alpha_p2(lowIdx), avg.RPL.alpha_p2(highIdx));
    p_alpha_p2(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.beta_p2(lowIdx), avg.RPL.beta_p2(highIdx));
    p_beta_p2(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.gamma_p2(lowIdx), avg.RPL.gamma_p2(highIdx));
    p_gamma_p2(s) = p;
    
    p_analysis_p2_rpl(s,1:5) = [p_delta_p2(s), p_theta_p2(s), p_alpha_p2(s), p_beta_p2(s), p_gamma_p2(s)];
    p_analysis_p2_rpl_min(s) = find(p_analysis_p2_rpl(s,1:5)==min(p_analysis_p2_rpl(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.RPL.delta_p3(lowIdx), avg.RPL.delta_p3(highIdx));
    p_delta_p3(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.theta_p3(lowIdx), avg.RPL.theta_p3(highIdx));
    p_theta_p3(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.alpha_p3(lowIdx), avg.RPL.alpha_p3(highIdx));
    p_alpha_p3(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.beta_p3(lowIdx), avg.RPL.beta_p3(highIdx));
    p_beta_p3(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.gamma_p3(lowIdx), avg.RPL.gamma_p3(highIdx));
    p_gamma_p3(s) = p;
    
    p_analysis_p3_rpl(s,1:5) = [p_delta_p3(s), p_theta_p3(s), p_alpha_p3(s), p_beta_p3(s), p_gamma_p3(s)];
    p_analysis_p3_rpl_min(s) = find(p_analysis_p3_rpl(s,1:5)==min(p_analysis_p3_rpl(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.RPL.delta_t1(lowIdx), avg.RPL.delta_t1(highIdx));
    p_delta_t1(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.theta_t1(lowIdx), avg.RPL.theta_t1(highIdx));
    p_theta_t1(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.alpha_t1(lowIdx), avg.RPL.alpha_t1(highIdx));
    p_alpha_t1(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.beta_t1(lowIdx), avg.RPL.beta_t1(highIdx));
    p_beta_t1(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.gamma_t1(lowIdx), avg.RPL.gamma_t1(highIdx));
    p_gamma_t1(s) = p;
    
    p_analysis_t1_rpl(s,1:5) = [p_delta_t1(s), p_theta_t1(s), p_alpha_t1(s), p_beta_t1(s), p_gamma_t1(s)];
    p_analysis_t1_rpl_min(s) = find(p_analysis_t1_rpl(s,1:5)==min(p_analysis_t1_rpl(s,1:5)));
    
    [h, p, ci, stats] = ttest2(avg.RPL.delta_t2(lowIdx), avg.RPL.delta_t2(highIdx));
    p_delta_t2(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.theta_t2(lowIdx), avg.RPL.theta_t2(highIdx));
    p_theta_t2(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.alpha_t2(lowIdx), avg.RPL.alpha_t2(highIdx));
    p_alpha_t2(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.beta_t2(lowIdx), avg.RPL.beta_t2(highIdx));
    p_beta_t2(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.gamma_t2(lowIdx), avg.RPL.gamma_t2(highIdx));
    p_gamma_t2(s) = p;
    
    p_analysis_t2_rpl(s,1:5) = [p_delta_t2(s), p_theta_t2(s), p_alpha_t2(s), p_beta_t2(s), p_gamma_t2(s)];
    p_analysis_t2_rpl_min(s) = find(p_analysis_t2_rpl(s,1:5)==min(p_analysis_t2_rpl(s,1:5)));
    
    
    
    [h, p, ci, stats] = ttest2(avg.RPL.delta_o1(lowIdx), avg.RPL.delta_o1(highIdx));
    p_delta_o1(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.theta_o1(lowIdx), avg.RPL.theta_o1(highIdx));
    p_theta_o1(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.alpha_o1(lowIdx), avg.RPL.alpha_o1(highIdx));
    p_alpha_o1(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.beta_o1(lowIdx), avg.RPL.beta_o1(highIdx));
    p_beta_o1(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.gamma_o1(lowIdx), avg.RPL.gamma_o1(highIdx));
    p_gamma_o1(s) = p;
    
    p_analysis_o1_rpl(s,1:5) = [p_delta_o1(s), p_theta_o1(s), p_alpha_o1(s), p_beta_o1(s), p_gamma_o1(s)];
    p_analysis_o1_rpl_min(s) = find(p_analysis_o1_rpl(s,1:5)==min(p_analysis_o1_rpl(s,1:5)));
    
    [h, p, ci, stats] = ttest2(avg.RPL.delta_o2(lowIdx), avg.RPL.delta_o2(highIdx));
    p_delta_o2(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.theta_o2(lowIdx), avg.RPL.theta_o2(highIdx));
    p_theta_o2(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.alpha_o2(lowIdx), avg.RPL.alpha_o2(highIdx));
    p_alpha_o2(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.beta_o2(lowIdx), avg.RPL.beta_o2(highIdx));
    p_beta_o2(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.gamma_o2(lowIdx), avg.RPL.gamma_o2(highIdx));
    p_gamma_o2(s) = p;
    
    p_analysis_o2_rpl(s,1:5) = [p_delta_o2(s), p_theta_o2(s), p_alpha_o2(s), p_beta_o2(s), p_gamma_o2(s)];
    p_analysis_o2_rpl_min(s) = find(p_analysis_o2_rpl(s,1:5)==min(p_analysis_o2_rpl(s,1:5)));
    
    [h, p, ci, stats] = ttest2(avg.RPL.delta_o3(lowIdx), avg.RPL.delta_o3(highIdx));
    p_delta_o3(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.theta_o3(lowIdx), avg.RPL.theta_o3(highIdx));
    p_theta_o3(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.alpha_o3(lowIdx), avg.RPL.alpha_o3(highIdx));
    p_alpha_o3(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.beta_o3(lowIdx), avg.RPL.beta_o3(highIdx));
    p_beta_o3(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.gamma_o3(lowIdx), avg.RPL.gamma_o3(highIdx));
    p_gamma_o3(s) = p;
    
    p_analysis_o3_rpl(s,1:5) = [p_delta_o3(s), p_theta_o3(s), p_alpha_o3(s), p_beta_o3(s), p_gamma_o3(s)];
    p_analysis_o3_rpl_min(s) = find(p_analysis_o3_rpl(s,1:5)==min(p_analysis_o3_rpl(s,1:5)));
    
    Result.ttest.vertical_rpl(s,:) = [p_analysis_f1(s,:) p_analysis_f2(s,:) p_analysis_f3(s,:) p_analysis_c1(s,:) p_analysis_c2(s,:) p_analysis_c3(s,:) p_analysis_p1(s,:) p_analysis_p2(s,:) p_analysis_p3(s,:) p_analysis_t1(s,:) p_analysis_t2(s,:) p_analysis_o1(s,:) p_analysis_o2(s,:) p_analysis_o3(s,:)];
    
    
    
    
    
    
    
            % Regional Vertical T-TEST 분석
    [h, p, ci, stats] = ttest2(avg.z.Regional_f1(lowIdx,1), avg.z.Regional_f1(highIdx,1));
    p_delta_f1(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_f1(lowIdx,2), avg.z.Regional_f1(highIdx,2));
    p_theta_f1(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_f1(lowIdx,3), avg.z.Regional_f1(highIdx,3));
    p_alpha_f1(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_f1(lowIdx,4), avg.z.Regional_f1(highIdx,4));
    p_beta_f1(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_f1(lowIdx,5), avg.z.Regional_f1(highIdx,5));
    p_gamma_f1(s) = p;
    
    p_analysis_f1_z(s,1:5) = [p_delta_f1(s), p_theta_f1(s), p_alpha_f1(s), p_beta_f1(s), p_gamma_f1(s)];
    p_analysis_f1_z_min(s) = find(p_analysis_f1_z(s,1:5)==min(p_analysis_f1_z(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.z.Regional_f2(lowIdx,1), avg.z.Regional_f2(highIdx,1));
    p_delta_f2(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_f2(lowIdx,2), avg.z.Regional_f2(highIdx,2));
    p_theta_f2(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_f2(lowIdx,3), avg.z.Regional_f2(highIdx,3));
    p_alpha_f2(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_f2(lowIdx,4), avg.z.Regional_f2(highIdx,4));
    p_beta_f2(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_f2(lowIdx,5), avg.z.Regional_f2(highIdx,5));
    p_gamma_f2(s) = p;
    
    p_analysis_f2_z(s,1:5) = [p_delta_f2(s), p_theta_f2(s), p_alpha_f2(s), p_beta_f2(s), p_gamma_f2(s)];
    p_analysis_f2_z_min(s) = find(p_analysis_f2_z(s,1:5)==min(p_analysis_f2_z(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.z.Regional_f3(lowIdx,1), avg.z.Regional_f3(highIdx,1));
    p_delta_f3(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_f3(lowIdx,2), avg.z.Regional_f3(highIdx,2));
    p_theta_f3(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_f3(lowIdx,3), avg.z.Regional_f3(highIdx,3));
    p_alpha_f3(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_f3(lowIdx,4), avg.z.Regional_f3(highIdx,4));
    p_beta_f3(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_f3(lowIdx,5), avg.z.Regional_f3(highIdx,5));
    p_gamma_f3(s) = p;
    
    p_analysis_f3_z(s,1:5) = [p_delta_f3(s), p_theta_f3(s), p_alpha_f3(s), p_beta_f3(s), p_gamma_f3(s)];
    p_analysis_f3_z_min(s) = find(p_analysis_f3_z(s,1:5)==min(p_analysis_f3_z(s,1:5)));
    
    
    
    [h, p, ci, stats] = ttest2(avg.z.Regional_c1(lowIdx,1), avg.z.Regional_c1(highIdx,1));
    p_delta_c1(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_c1(lowIdx,2), avg.z.Regional_c1(highIdx,2));
    p_theta_c1(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_c1(lowIdx,3), avg.z.Regional_c1(highIdx,3));
    p_alpha_c1(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_c1(lowIdx,4), avg.z.Regional_c1(highIdx,4));
    p_beta_c1(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_c1(lowIdx,5), avg.z.Regional_c1(highIdx,5));
    p_gamma_c1(s) = p;
    
    p_analysis_c1_z(s,1:5) = [p_delta_c1(s), p_theta_c1(s), p_alpha_c1(s), p_beta_c1(s), p_gamma_c1(s)];
    p_analysis_c1_z_min(s) = find(p_analysis_c1_z(s,1:5)==min(p_analysis_c1_z(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.z.Regional_c2(lowIdx,1), avg.z.Regional_c2(highIdx,1));
    p_delta_c2(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_c2(lowIdx,2), avg.z.Regional_c2(highIdx,2));
    p_theta_c2(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_c2(lowIdx,3), avg.z.Regional_c2(highIdx,3));
    p_alpha_c2(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_c2(lowIdx,4), avg.z.Regional_c2(highIdx,4));
    p_beta_c2(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_c2(lowIdx,5), avg.z.Regional_c2(highIdx,5));
    p_gamma_c2(s) = p;
    
    p_analysis_c2_z(s,1:5) = [p_delta_c2(s), p_theta_c2(s), p_alpha_c2(s), p_beta_c2(s), p_gamma_c2(s)];
    p_analysis_c2_z_min(s) = find(p_analysis_c2_z(s,1:5)==min(p_analysis_c2_z(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.z.Regional_c3(lowIdx,1), avg.z.Regional_c3(highIdx,1));
    p_delta_c3(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_c3(lowIdx,2), avg.z.Regional_c3(highIdx,2));
    p_theta_c3(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_c3(lowIdx,3), avg.z.Regional_c3(highIdx,3));
    p_alpha_c3(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_c3(lowIdx,4), avg.z.Regional_c3(highIdx,4));
    p_beta_c3(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_c3(lowIdx,5), avg.z.Regional_c3(highIdx,5));
    p_gamma_c3(s) = p;
    
    p_analysis_c3_z(s,1:5) = [p_delta_c3(s), p_theta_c3(s), p_alpha_c3(s), p_beta_c3(s), p_gamma_c3(s)];
    p_analysis_c3_z_min(s) = find(p_analysis_c3_z(s,1:5)==min(p_analysis_c3_z(s,1:5)));
    
    

    
    
    [h, p, ci, stats] = ttest2(avg.z.Regional_p1(lowIdx,1), avg.z.Regional_p1(highIdx,1));
    p_delta_p1(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_p1(lowIdx,2), avg.z.Regional_p1(highIdx,2));
    p_theta_p1(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_p1(lowIdx,3), avg.z.Regional_p1(highIdx,3));
    p_alpha_p1(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_p1(lowIdx,4), avg.z.Regional_p1(highIdx,4));
    p_beta_p1(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_p1(lowIdx,5), avg.z.Regional_p1(highIdx,5));
    p_gamma_p1(s) = p;
    
    p_analysis_p1_z(s,1:5) = [p_delta_p1(s), p_theta_p1(s), p_alpha_p1(s), p_beta_p1(s), p_gamma_p1(s)];
    p_analysis_p1_z_min(s) = find(p_analysis_p1_z(s,1:5)==min(p_analysis_p1_z(s,1:5)));
    
    
    
    [h, p, ci, stats] = ttest2(avg.z.Regional_p2(lowIdx,1), avg.z.Regional_p2(highIdx,1));
    p_delta_p2(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_p2(lowIdx,2), avg.z.Regional_p2(highIdx,2));
    p_theta_p2(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_p2(lowIdx,3), avg.z.Regional_p2(highIdx,3));
    p_alpha_p2(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_p2(lowIdx,4), avg.z.Regional_p2(highIdx,4));
    p_beta_p2(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_p2(lowIdx,5), avg.z.Regional_p2(highIdx,5));
    p_gamma_p2(s) = p;
    
    p_analysis_p2_z(s,1:5) = [p_delta_p2(s), p_theta_p2(s), p_alpha_p2(s), p_beta_p2(s), p_gamma_p2(s)];
    p_analysis_p2_z_min(s) = find(p_analysis_p2_z(s,1:5)==min(p_analysis_p2_z(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.z.Regional_p3(lowIdx,1), avg.z.Regional_p3(highIdx,1));
    p_delta_p3(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_p3(lowIdx,2), avg.z.Regional_p3(highIdx,2));
    p_theta_p3(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_p3(lowIdx,3), avg.z.Regional_p3(highIdx,3));
    p_alpha_p3(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_p3(lowIdx,4), avg.z.Regional_p3(highIdx,4));
    p_beta_p3(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_p3(lowIdx,5), avg.z.Regional_p3(highIdx,5));
    p_gamma_p3(s) = p;
    
    p_analysis_p3_z(s,1:5) = [p_delta_p3(s), p_theta_p3(s), p_alpha_p3(s), p_beta_p3(s), p_gamma_p3(s)];
    p_analysis_p3_z_min(s) = find(p_analysis_p3_z(s,1:5)==min(p_analysis_p3_z(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.z.Regional_t1(lowIdx,1), avg.z.Regional_t1(highIdx,1));
    p_delta_t1(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_t1(lowIdx,2), avg.z.Regional_t1(highIdx,2));
    p_theta_t1(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_t1(lowIdx,3), avg.z.Regional_t1(highIdx,3));
    p_alpha_t1(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_t1(lowIdx,4), avg.z.Regional_t1(highIdx,4));
    p_beta_t1(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_t1(lowIdx,5), avg.z.Regional_t1(highIdx,5));
    p_gamma_t1(s) = p;
    
    p_analysis_t1_z(s,1:5) = [p_delta_t1(s), p_theta_t1(s), p_alpha_t1(s), p_beta_t1(s), p_gamma_t1(s)];
    p_analysis_t1_z_min(s) = find(p_analysis_t1_z(s,1:5)==min(p_analysis_t1_z(s,1:5)));
    
    [h, p, ci, stats] = ttest2(avg.z.Regional_t2(lowIdx,1), avg.z.Regional_t2(highIdx,1));
    p_delta_t2(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_t2(lowIdx,2), avg.z.Regional_t2(highIdx,2));
    p_theta_t2(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_t2(lowIdx,3), avg.z.Regional_t2(highIdx,3));
    p_alpha_t2(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_t2(lowIdx,4), avg.z.Regional_t2(highIdx,4));
    p_beta_t2(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_t2(lowIdx,5), avg.z.Regional_t2(highIdx,5));
    p_gamma_t2(s) = p;
    
    p_analysis_t2_z(s,1:5) = [p_delta_t2(s), p_theta_t2(s), p_alpha_t2(s), p_beta_t2(s), p_gamma_t2(s)];
    p_analysis_t2_z_min(s) = find(p_analysis_t2_z(s,1:5)==min(p_analysis_t2_z(s,1:5)));
    
    
    
    [h, p, ci, stats] = ttest2(avg.z.Regional_o1(lowIdx,1), avg.z.Regional_o1(highIdx,1));
    p_delta_o1(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_o1(lowIdx,2), avg.z.Regional_o1(highIdx,2));
    p_theta_o1(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_o1(lowIdx,3), avg.z.Regional_o1(highIdx,3));
    p_alpha_o1(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_o1(lowIdx,4), avg.z.Regional_o1(highIdx,4));
    p_beta_o1(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_o1(lowIdx,5), avg.z.Regional_o1(highIdx,5));
    p_gamma_o1(s) = p;
    
    p_analysis_o1_z(s,1:5) = [p_delta_o1(s), p_theta_o1(s), p_alpha_o1(s), p_beta_o1(s), p_gamma_o1(s)];
    p_analysis_o1_z_min(s) = find(p_analysis_o1_z(s,1:5)==min(p_analysis_o1_z(s,1:5)));
    
    [h, p, ci, stats] = ttest2(avg.z.Regional_o2(lowIdx,1), avg.z.Regional_o2(highIdx,1));
    p_delta_o2(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_o2(lowIdx,2), avg.z.Regional_o2(highIdx,2));
    p_theta_o2(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_o2(lowIdx,3), avg.z.Regional_o2(highIdx,3));
    p_alpha_o2(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_o2(lowIdx,4), avg.z.Regional_o2(highIdx,4));
    p_beta_o2(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_o2(lowIdx,5), avg.z.Regional_o2(highIdx,5));
    p_gamma_o2(s) = p;
    
    p_analysis_o2_z(s,1:5) = [p_delta_o2(s), p_theta_o2(s), p_alpha_o2(s), p_beta_o2(s), p_gamma_o2(s)];
    p_analysis_o2_z_min(s) = find(p_analysis_o2_z(s,1:5)==min(p_analysis_o2_z(s,1:5)));
    
    [h, p, ci, stats] = ttest2(avg.z.Regional_o3(lowIdx,1), avg.z.Regional_o3(highIdx,1));
    p_delta_o3(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_o3(lowIdx,2), avg.z.Regional_o3(highIdx,2));
    p_theta_o3(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_o3(lowIdx,3), avg.z.Regional_o3(highIdx,3));
    p_alpha_o3(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_o3(lowIdx,4), avg.z.Regional_o3(highIdx,4));
    p_beta_o3(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_o3(lowIdx,5), avg.z.Regional_o3(highIdx,5));
    p_gamma_o3(s) = p;
    
    p_analysis_o3_z(s,1:5) = [p_delta_o3(s), p_theta_o3(s), p_alpha_o3(s), p_beta_o3(s), p_gamma_o3(s)];
    p_analysis_o3_z_min(s) = find(p_analysis_o3_z(s,1:5)==min(p_analysis_o3_z(s,1:5)));
    
    Result.ttest.vertical_z(s,:) = [p_analysis_f1(s,:) p_analysis_f2(s,:) p_analysis_f3(s,:) p_analysis_c1(s,:) p_analysis_c2(s,:) p_analysis_c3(s,:) p_analysis_p1(s,:) p_analysis_p2(s,:) p_analysis_p3(s,:) p_analysis_t1(s,:) p_analysis_t2(s,:) p_analysis_o1(s,:) p_analysis_o2(s,:) p_analysis_o3(s,:)];
    
    
    
    
    
    
    
    
    
    
    %% Feature Selection
    feature_ALL = avg.PSD;
    feature_PSD = avg.PSD(:,find(p_analysis(s,:)<0.05));
    feature_Regional = [avg.Regional_f(:,find(p_analysis_f(s,:)<0.05)),avg.Regional_c(:,find(p_analysis_c(s,:)<0.05)),avg.Regional_p(:,find(p_analysis_p(s,:)<0.05)),avg.Regional_t(:,find(p_analysis_t(s,:)<0.05)),avg.Regional_o(:,find(p_analysis_o(s,:)<0.05))];
    feature_Vertical = [avg.Regional_f1(:,find(p_analysis_f1(s,:)<0.05)),avg.Regional_f2(:,find(p_analysis_f2(s,:)<0.05)),avg.Regional_f3(:,find(p_analysis_f3(s,:)<0.05)),avg.Regional_c1(:,find(p_analysis_c1(s,:)<0.05)),avg.Regional_c2(:,find(p_analysis_c2(s,:)<0.05)),avg.Regional_c3(:,find(p_analysis_c3(s,:)<0.05)),avg.Regional_p1(:,find(p_analysis_p1(s,:)<0.05)),avg.Regional_p2(:,find(p_analysis_p2(s,:)<0.05)),avg.Regional_p3(:,find(p_analysis_p3(s,:)<0.05)),avg.Regional_t1(:,find(p_analysis_t1(s,:)<0.05)),avg.Regional_t2(:,find(p_analysis_t2(s,:)<0.05)),avg.Regional_o1(:,find(p_analysis_o1(s,:)<0.05)),avg.Regional_o2(:,find(p_analysis_o2(s,:)<0.05)),avg.Regional_o3(:,find(p_analysis_o3(s,:)<0.05))];
    
    feature_ALL_RPL = avg.RPL.PSD;
    feature_PSD_RPL = avg.RPL.PSD(:,find(p_analysis_RPL(s,:)<0.05));
    feature_Regional_RPL = [avg.RPL.Regional_f(:,find(p_analysis_RPL_f(s,:)<0.05)),avg.RPL.Regional_c(:,find(p_analysis_RPL_c(s,:)<0.05)),avg.RPL.Regional_p(:,find(p_analysis_RPL_p(s,:)<0.05)),avg.RPL.Regional_t(:,find(p_analysis_RPL_t(s,:)<0.05)),avg.RPL.Regional_o(:,find(p_analysis_RPL_o(s,:)<0.05))];
    feature_Vertical_RPL = [avg.RPL.Regional_f1(:,find(p_analysis_f1_rpl(s,:)<0.05)),avg.RPL.Regional_f2(:,find(p_analysis_f2_rpl(s,:)<0.05)),avg.RPL.Regional_f3(:,find(p_analysis_f3_rpl(s,:)<0.05)),avg.RPL.Regional_c1(:,find(p_analysis_c1_rpl(s,:)<0.05)),avg.RPL.Regional_c2(:,find(p_analysis_c2_rpl(s,:)<0.05)),avg.RPL.Regional_c3(:,find(p_analysis_c3_rpl(s,:)<0.05)),avg.RPL.Regional_p1(:,find(p_analysis_p1_rpl(s,:)<0.05)),avg.RPL.Regional_p2(:,find(p_analysis_p2_rpl(s,:)<0.05)),avg.RPL.Regional_p3(:,find(p_analysis_p3_rpl(s,:)<0.05)),avg.RPL.Regional_t1(:,find(p_analysis_t1_rpl(s,:)<0.05)),avg.RPL.Regional_t2(:,find(p_analysis_t2_rpl(s,:)<0.05)),avg.RPL.Regional_o1(:,find(p_analysis_o1_rpl(s,:)<0.05)),avg.RPL.Regional_o2(:,find(p_analysis_o2_rpl(s,:)<0.05)),avg.RPL.Regional_o3(:,find(p_analysis_o3_rpl(s,:)<0.05))];

    feature_ALL_z = avg.z.PSD;
    feature_PSD_z = avg.z.PSD(:,find(p_analysis_z(s,:)<0.05));
    feature_Regional_z = [avg.z.Regional_f(:,find(p_analysis_z_f(s,:)<0.05)),avg.z.Regional_c(:,find(p_analysis_z_c(s,:)<0.05)),avg.z.Regional_p(:,find(p_analysis_z_p(s,:)<0.05)),avg.z.Regional_t(:,find(p_analysis_z_t(s,:)<0.05)),avg.z.Regional_o(:,find(p_analysis_z_o(s,:)<0.05))];
    feature_Vertical_z = [avg.z.Regional_f1(:,find(p_analysis_f1_z(s,:)<0.05)),avg.z.Regional_f2(:,find(p_analysis_f2_z(s,:)<0.05)),avg.z.Regional_f3(:,find(p_analysis_f3_z(s,:)<0.05)),avg.z.Regional_c1(:,find(p_analysis_c1_z(s,:)<0.05)),avg.z.Regional_c2(:,find(p_analysis_c2_z(s,:)<0.05)),avg.z.Regional_c3(:,find(p_analysis_c3_z(s,:)<0.05)),avg.z.Regional_p1(:,find(p_analysis_p1_z(s,:)<0.05)),avg.z.Regional_p2(:,find(p_analysis_p2_z(s,:)<0.05)),avg.z.Regional_p3(:,find(p_analysis_p3_z(s,:)<0.05)),avg.z.Regional_t1(:,find(p_analysis_t1_z(s,:)<0.05)),avg.z.Regional_t2(:,find(p_analysis_t2_z(s,:)<0.05)),avg.z.Regional_o1(:,find(p_analysis_o1_z(s,:)<0.05)),avg.z.Regional_o2(:,find(p_analysis_o2_z(s,:)<0.05)),avg.z.Regional_o3(:,find(p_analysis_o3_z(s,:)<0.05))];
    
    %% Classification
    [Accuracy_ALL AUC_ALL] = LDA_4CV(feature_ALL, highIdx, lowIdx, s);
    [Accuracy_PSD AUC_PSD] = LDA_4CV(feature_PSD, highIdx, lowIdx, s);
    [Accuracy_Regional AUC_Regional] = LDA_4CV(feature_Regional, highIdx, lowIdx, s);
    [Accuracy_Vertical AUC_Vertical] = LDA_4CV(feature_Vertical, highIdx, lowIdx, s);
    
    [Accuracy_ALL_RPL AUC_ALL_RPL] = LDA_4CV(feature_ALL_RPL, highIdx, lowIdx, s);
    [Accuracy_PSD_RPL AUC_PSD_RPL] = LDA_4CV(feature_PSD_RPL, highIdx, lowIdx, s);
    [Accuracy_Regional_RPL AUC_Regional_RPL] = LDA_4CV(feature_Regional_RPL, highIdx, lowIdx, s);
    [Accuracy_Vertical_RPL AUC_Vertical_RPL] = LDA_4CV(feature_Vertical_RPL, highIdx, lowIdx, s);
    
    [Accuracy_ALL_z AUC_ALL_z] = LDA_4CV(feature_ALL_z, highIdx, lowIdx, s);
    [Accuracy_PSD_z AUC_PSD_z] = LDA_4CV(feature_PSD_z, highIdx, lowIdx, s);
    [Accuracy_Regional_z AUC_Regional_z] = LDA_4CV(feature_Regional_z, highIdx, lowIdx, s);
    [Accuracy_Vertical_z AUC_Vertical_z] = LDA_4CV(feature_Vertical_z, highIdx, lowIdx, s);
    
    Result.Fatigue_Accuracy(s,:) = [Accuracy_ALL Accuracy_PSD Accuracy_Regional Accuracy_Vertical];
    Result.Fatigue_AUC(s,:) = [AUC_ALL AUC_PSD AUC_Regional AUC_Vertical];
    
    Result.Fatigue_Accuracy_RPL(s,:) = [Accuracy_ALL_RPL Accuracy_PSD_RPL Accuracy_Regional_RPL Accuracy_Vertical_RPL];
    Result.Fatigue_AUC_RPL(s,:) = [AUC_ALL_RPL AUC_PSD_RPL AUC_Regional_RPL AUC_Vertical_RPL];
    
    Result.Fatigue_Accuracy_z(s,:) = [Accuracy_ALL_z Accuracy_PSD_z Accuracy_Regional_z Accuracy_Vertical_z];
    Result.Fatigue_AUC_z(s,:) = [AUC_ALL_z AUC_PSD_z AUC_Regional_z AUC_Vertical_z];
    
%     %% Regression
%     [Prediction_ALL, RMSE_ALL, Error_Rate_ALL] = MLR_4CV(feature_ALL, kss, s);
%     [Prediction_PSD, RMSE_PSD, Error_Rate_PSD] = MLR_4CV(feature_PSD, kss, s);
%     [Prediction_Regional, RMSE_Regional, Error_Rate_Regional] = MLR_4CV(feature_Regional, kss, s);
%     [Prediction_Vertical, RMSE_Vertical, Error_Rate_Vertical] = MLR_4CV(feature_Vertical, kss, s);
%     
%     [Prediction_ALL_RPL, RMSE_ALL_RPL, Error_Rate_ALL_RPL] = MLR_4CV(feature_ALL_RPL, kss, s);
%     [Prediction_PSD_RPL, RMSE_PSD_RPL, Error_Rate_PSD_RPL] = MLR_4CV(feature_PSD_RPL, kss, s);
%     [Prediction_Regional_RPL, RMSE_Regional_RPL, Error_Rate_Regional_RPL] = MLR_4CV(feature_Regional_RPL, kss, s);
%     [Prediction_Vertical_RPL, RMSE_Vertical_RPL, Error_Rate_Vertical_RPL] = MLR_4CV(feature_Vertical_RPL, kss, s);
%     
%     [Prediction_ALL_z, RMSE_ALL_z, Error_Rate_ALL_z] = MLR_4CV(feature_ALL_z, kss, s);
%     [Prediction_PSD_z, RMSE_PSD_z, Error_Rate_PSD_z] = MLR_4CV(feature_PSD_z, kss, s);
%     [Prediction_Regional_z, RMSE_Regional_z, Error_Rate_Regional_z] = MLR_4CV(feature_Regional_z, kss, s);
%     [Prediction_Vertical_z, RMSE_Vertical_z, Error_Rate_Vertical_z] = MLR_4CV(feature_Vertical_z, kss, s);
%  
%     Result.Fatigue_Prediction(s,:) = [Prediction_ALL Prediction_PSD Prediction_Regional Prediction_Vertical];
%     Result.Fatigue_RMSE(s,:) = [RMSE_ALL RMSE_PSD RMSE_Regional RMSE_Vertical];   
%     Result.Fatigue_Error(s,:) = [Error_Rate_ALL Error_Rate_PSD Error_Rate_Regional Error_Rate_Vertical];   
%     
%     Result.Fatigue_Prediction_RPL(s,:) = [Prediction_ALL_RPL Prediction_PSD_RPL Prediction_Regional_RPL Prediction_Vertical_RPL];
%     Result.Fatigue_RMSE_RPL(s,:) = [RMSE_ALL_RPL RMSE_PSD_RPL RMSE_Regional_RPL RMSE_Vertical_RPL];   
%     Result.Fatigue_Error_RPL(s,:) = [Error_Rate_ALL_RPL Error_Rate_PSD_RPL Error_Rate_Regional_RPL Error_Rate_Vertical_RPL];   
%     
%     Result.Fatigue_Prediction_z(s,:) = [Prediction_ALL_z Prediction_PSD_z Prediction_Regional_z Prediction_Vertical_z];
%     Result.Fatigue_RMSE_z(s,:) = [RMSE_ALL_z RMSE_PSD_z RMSE_Regional_z RMSE_Vertical_z];   
%     Result.Fatigue_Error_z(s,:) = [Error_Rate_ALL_z Error_Rate_PSD_z Error_Rate_Regional_z Error_Rate_Vertical_z];   

    %% Save
    Result.Fatigue_Accuracy_10(s,:) = Result.Fatigue_Accuracy(s,:);
    Result.Fatigue_AUC_10(s,:) = Result.Fatigue_AUC(s,:);  
%     Result.Fatigue_Prediction_10(s,:) = Result.Fatigue_Prediction(s,:);
%     Result.Fatigue_RMSE_10(s,:) = Result.Fatigue_RMSE(s,:);
%     Result.Fatigue_Error_10(s,:) = Result.Fatigue_Error(s,:);
    
    Result.Fatigue_Accuracy_10_RPL(s,:) = Result.Fatigue_Accuracy_RPL(s,:);
    Result.Fatigue_AUC_10_RPL(s,:) = Result.Fatigue_AUC_RPL(s,:);  
%     Result.Fatigue_Prediction_10_RPL(s,:) = Result.Fatigue_Prediction_RPL(s,:);
%     Result.Fatigue_RMSE_10_RPL(s,:) = Result.Fatigue_RMSE_RPL(s,:);
%     Result.Fatigue_Error_10_RPL(s,:) = Result.Fatigue_Error_RPL(s,:);
    
    Result.Fatigue_Accuracy_10_z(s,:) = Result.Fatigue_Accuracy_z(s,:);
    Result.Fatigue_AUC_10_z(s,:) = Result.Fatigue_AUC_z(s,:);  
%     Result.Fatigue_Prediction_10_z(s,:) = Result.Fatigue_Prediction_z(s,:);
%     Result.Fatigue_RMSE_10_z(s,:) = Result.Fatigue_RMSE_z(s,:);
%     Result.Fatigue_Error_10_z(s,:) = Result.Fatigue_Error_z(s,:);

    
    
    
    %% Segmentation of bio-signals with 20 seconds length of epoch
    epoch = segmentationFatigue_20(cnt, mrk, interKSS);
    kssIdx = find(strcmp(epoch.mClab, 'KSS'));
    kss = squeeze(mean(epoch.misc(:, kssIdx, :), 1));
    
    %% Regional Channel Selection
    channel_f=epoch.x(:,[1:5,31:36],:);
    channel_c=epoch.x(:,[6:9,11:13,16:19,39:40,43:46,48:50],:);
    channel_p=epoch.x(:,[21:25,52:55],:);
    channel_t=epoch.x(:,[10,14:15,20,37:38,41:42,47,51],:);
    channel_o=epoch.x(:,[26:30,56:60],:);
    
    channel_f1=epoch.x(:,[1,3,31,33],:);
    channel_f2=epoch.x(:,[2,5,32,36],:);
    channel_f3=epoch.x(:,[4,34:35],:);
    channel_c1=epoch.x(:,[6:7,11,16,17,39,43,44,48],:);
    channel_c2=epoch.x(:,[8:9,13,18,19,40,45,46,50],:);
    channel_c3=epoch.x(:,[12,49],:);
    channel_p1=epoch.x(:,[21:22,52,53],:);
    channel_p2=epoch.x(:,[24:25,54,55],:);
    channel_p3=epoch.x(:,[23],:);
    channel_t1=epoch.x(:,[10,15,37,38,47],:);
    channel_t2=epoch.x(:,[14,20,41,42,51],:);
    channel_o1=epoch.x(:,[26,27,56,57],:);
    channel_o2=epoch.x(:,[29,30,59,60],:);
    channel_o3=epoch.x(:,[28,58],:);
    
    %% Feature Extraction - Power spectrum analysis for EEG signals
    
    % PSD Feature Extraction
    [cca.delta, avg.delta] = Power_spectrum(epoch.x, [0.5 4], cnt.fs);
    [cca.theta, avg.theta] = Power_spectrum(epoch.x, [4 8], cnt.fs);
    [cca.alpha, avg.alpha] = Power_spectrum(epoch.x, [8 13], cnt.fs);
    [cca.beta, avg.beta] = Power_spectrum(epoch.x, [13 30], cnt.fs);
    [cca.gamma, avg.gamma] = Power_spectrum(epoch.x, [30 40], cnt.fs);
    avg.delta = avg.delta'; avg.theta=avg.theta'; avg.alpha=avg.alpha'; avg.beta = avg.beta'; avg.gamma = avg.gamma';
    
    cca.PSD = [cca.delta cca.theta cca.alpha cca.beta cca.gamma];
    avg.PSD = [avg.delta avg.theta avg.alpha avg.beta avg.gamma];

    % RPL Feature Extraction
    cca.RPL.delta = cca.delta ./ (cca.delta + cca.theta + cca.alpha + cca.beta + cca.gamma) ;
    avg.RPL.delta = avg.delta ./ sum(avg.PSD,2) ;
    cca.RPL.theta = cca.theta ./ (cca.delta + cca.theta + cca.alpha + cca.beta + cca.gamma) ;
    avg.RPL.theta = avg.theta ./ sum(avg.PSD,2) ;
    cca.RPL.alpha = cca.alpha ./ (cca.delta + cca.theta + cca.alpha + cca.beta + cca.gamma) ;
    avg.RPL.alpha = avg.alpha ./ sum(avg.PSD,2) ;
    cca.RPL.beta = cca.beta ./ (cca.delta + cca.theta + cca.alpha + cca.beta + cca.gamma) ;
    avg.RPL.beta = avg.beta ./ sum(avg.PSD,2) ;
    cca.RPL.gamma = cca.gamma ./ (cca.delta + cca.theta + cca.alpha + cca.beta + cca.gamma) ;
    avg.RPL.gamma = avg.gamma ./ sum(avg.PSD,2) ;
    
    cca.RPL.PSD = [cca.RPL.delta, cca.RPL.theta, cca.RPL.alpha, cca.RPL.beta, cca.RPL.gamma];
    avg.RPL.PSD = [avg.RPL.delta, avg.RPL.theta, avg.RPL.alpha, avg.RPL.beta, avg.RPL.gamma];
    
    % Z-SCORE Feature Extraction
    cca.z.PSD = [zscore(cca.delta) zscore(cca.theta) zscore(cca.alpha) zscore(cca.beta) zscore(cca.gamma)];
    avg.z.PSD = [zscore(avg.delta) zscore(avg.theta) zscore(avg.alpha) zscore(avg.beta) zscore(avg.gamma)];
    
    
    % Regional Feature Extraction
    [cca.delta_f,avg.delta_f] = Power_spectrum(channel_f, [0.5 4], cnt.fs);
    [cca.theta_f,avg.theta_f] = Power_spectrum(channel_f, [4 8], cnt.fs);
    [cca.alpha_f,avg.alpha_f] = Power_spectrum(channel_f, [8 13], cnt.fs);
    [cca.beta_f,avg.beta_f] = Power_spectrum(channel_f, [13 30], cnt.fs);
    [cca.gamma_f,avg.gamma_f] = Power_spectrum(channel_f, [30 40], cnt.fs);
    avg.delta_f = avg.delta_f'; avg.theta_f=avg.theta_f'; avg.alpha_f=avg.alpha_f'; avg.beta_f = avg.beta_f'; avg.gamma_f = avg.gamma_f';
    
    cca.Regional_f = [cca.delta_f cca.theta_f cca.alpha_f cca.beta_f cca.gamma_f];
    avg.Regional_f = [avg.delta_f avg.theta_f avg.alpha_f avg.beta_f avg.gamma_f];
    
    cca.RPL.delta_f = cca.delta_f ./ (cca.delta_f + cca.theta_f + cca.alpha_f + cca.beta_f + cca.gamma_f) ;
    avg.RPL.delta_f = avg.delta_f ./ sum(avg.Regional_f,2) ;
    cca.RPL.theta_f = cca.theta_f ./ (cca.delta_f + cca.theta_f + cca.alpha_f + cca.beta_f + cca.gamma_f) ;
    avg.RPL.theta_f = avg.theta_f ./ sum(avg.Regional_f,2) ;
    cca.RPL.alpha_f = cca.alpha_f ./ (cca.delta_f + cca.theta_f + cca.alpha_f + cca.beta_f + cca.gamma_f) ;
    avg.RPL.alpha_f = avg.alpha_f ./ sum(avg.Regional_f,2) ;
    cca.RPL.beta_f = cca.beta_f ./ (cca.delta_f + cca.theta_f + cca.alpha_f + cca.beta_f + cca.gamma_f) ;
    avg.RPL.beta_f = avg.beta_f ./ sum(avg.Regional_f,2) ;
    cca.RPL.gamma_f = cca.gamma_f ./ (cca.delta_f + cca.theta_f + cca.alpha_f + cca.beta_f + cca.gamma_f) ;
    avg.RPL.gamma_f = avg.gamma_f ./ sum(avg.Regional_f,2) ;    
    
    cca.RPL.Regional_f = [cca.RPL.delta_f cca.RPL.theta_f cca.RPL.alpha_f cca.RPL.beta_f cca.RPL.gamma_f]; 
    avg.RPL.Regional_f = [avg.RPL.delta_f avg.RPL.theta_f avg.RPL.alpha_f avg.RPL.beta_f avg.RPL.gamma_f];
    
    cca.z.Regional_f = [zscore(cca.delta_f) zscore(cca.theta_f) zscore(cca.alpha_f) zscore(cca.beta_f) zscore(cca.gamma_f)];
    avg.z.Regional_f = [zscore(avg.delta_f) zscore(avg.theta_f) zscore(avg.alpha_f) zscore(avg.beta_f) zscore(avg.gamma_f)];   
    
    
    
    [cca.delta_c,avg.delta_c] = Power_spectrum(channel_c, [0.5 4], cnt.fs);
    [cca.theta_c,avg.theta_c] = Power_spectrum(channel_c, [4 8], cnt.fs);
    [cca.alpha_c,avg.alpha_c] = Power_spectrum(channel_c, [8 13], cnt.fs);
    [cca.beta_c,avg.beta_c] = Power_spectrum(channel_c, [13 30], cnt.fs);
    [cca.gamma_c,avg.gamma_c] = Power_spectrum(channel_c, [30 40], cnt.fs);
    avg.delta_c = avg.delta_c'; avg.theta_c=avg.theta_c'; avg.alpha_c=avg.alpha_c'; avg.beta_c = avg.beta_c'; avg.gamma_c = avg.gamma_c';
    
    cca.Regional_c = [cca.delta_c cca.theta_c cca.alpha_c cca.beta_c cca.gamma_c];
    avg.Regional_c = [avg.delta_c avg.theta_c avg.alpha_c avg.beta_c avg.gamma_c];
    
    cca.RPL.delta_c = cca.delta_c ./ (cca.delta_c + cca.theta_c + cca.alpha_c + cca.beta_c + cca.gamma_c) ;
    avg.RPL.delta_c = avg.delta_c ./ sum(avg.Regional_c,2) ;
    cca.RPL.theta_c = cca.theta_c ./ (cca.delta_c + cca.theta_c + cca.alpha_c + cca.beta_c + cca.gamma_c) ;
    avg.RPL.theta_c = avg.theta_c ./ sum(avg.Regional_c,2) ;
    cca.RPL.alpha_c = cca.alpha_c ./ (cca.delta_c + cca.theta_c + cca.alpha_c + cca.beta_c + cca.gamma_c) ;
    avg.RPL.alpha_c = avg.alpha_c ./ sum(avg.Regional_c,2) ;
    cca.RPL.beta_c = cca.beta_c ./ (cca.delta_c + cca.theta_c + cca.alpha_c + cca.beta_c + cca.gamma_c) ;
    avg.RPL.beta_c = avg.beta_c ./ sum(avg.Regional_c,2) ;
    cca.RPL.gamma_c = cca.gamma_c ./ (cca.delta_c + cca.theta_c + cca.alpha_c + cca.beta_c + cca.gamma_c) ;
    avg.RPL.gamma_c = avg.gamma_c ./ sum(avg.Regional_c,2) ;     
    
    cca.RPL.Regional_c = [cca.RPL.delta_c cca.RPL.theta_c cca.RPL.alpha_c cca.RPL.beta_c cca.RPL.gamma_c]; 
    avg.RPL.Regional_c = [avg.RPL.delta_c avg.RPL.theta_c avg.RPL.alpha_c avg.RPL.beta_c avg.RPL.gamma_c];
    
    cca.z.Regional_c = [zscore(cca.delta_c) zscore(cca.theta_c) zscore(cca.alpha_c) zscore(cca.beta_c) zscore(cca.gamma_c)];
    avg.z.Regional_c = [zscore(avg.delta_c) zscore(avg.theta_c) zscore(avg.alpha_c) zscore(avg.beta_c) zscore(avg.gamma_c)];   
    
    
    [cca.delta_p,avg.delta_p] = Power_spectrum(channel_p, [0.5 4], cnt.fs);
    [cca.theta_p,avg.theta_p] = Power_spectrum(channel_p, [4 8], cnt.fs);
    [cca.alpha_p,avg.alpha_p] = Power_spectrum(channel_p, [8 13], cnt.fs);
    [cca.beta_p,avg.beta_p] = Power_spectrum(channel_p, [13 30], cnt.fs);
    [cca.gamma_p,avg.gamma_p] = Power_spectrum(channel_p, [30 40], cnt.fs);
    avg.delta_p = avg.delta_p'; avg.theta_p=avg.theta_p'; avg.alpha_p=avg.alpha_p'; avg.beta_p = avg.beta_p'; avg.gamma_p = avg.gamma_p';
    
    cca.Regional_p = [cca.delta_p cca.theta_p cca.alpha_p cca.beta_p cca.gamma_p];
    avg.Regional_p = [avg.delta_p avg.theta_p avg.alpha_p avg.beta_p avg.gamma_p];

    cca.RPL.delta_p = cca.delta_p ./ (cca.delta_p + cca.theta_p + cca.alpha_p + cca.beta_p + cca.gamma_p) ;
    avg.RPL.delta_p = avg.delta_p ./ sum(avg.Regional_p,2) ;
    cca.RPL.theta_p = cca.theta_p ./ (cca.delta_p + cca.theta_p + cca.alpha_p + cca.beta_p + cca.gamma_p) ;
    avg.RPL.theta_p = avg.theta_p ./ sum(avg.Regional_p,2) ;
    cca.RPL.alpha_p = cca.alpha_p ./ (cca.delta_p + cca.theta_p + cca.alpha_p + cca.beta_p + cca.gamma_p) ;
    avg.RPL.alpha_p = avg.alpha_p ./ sum(avg.Regional_p,2) ;
    cca.RPL.beta_p = cca.beta_p ./ (cca.delta_p + cca.theta_p + cca.alpha_p + cca.beta_p + cca.gamma_p) ;
    avg.RPL.beta_p = avg.beta_p ./ sum(avg.Regional_p,2) ;
    cca.RPL.gamma_p = cca.gamma_p ./ (cca.delta_p + cca.theta_p + cca.alpha_p + cca.beta_p + cca.gamma_p) ;
    avg.RPL.gamma_p = avg.gamma_p ./ sum(avg.Regional_p,2) ;     
    
    cca.RPL.Regional_p = [cca.RPL.delta_p cca.RPL.theta_p cca.RPL.alpha_p cca.RPL.beta_p cca.RPL.gamma_p]; 
    avg.RPL.Regional_p = [avg.RPL.delta_p avg.RPL.theta_p avg.RPL.alpha_p avg.RPL.beta_p avg.RPL.gamma_p];
    
    cca.z.Regional_p = [zscore(cca.delta_p) zscore(cca.theta_p) zscore(cca.alpha_p) zscore(cca.beta_p) zscore(cca.gamma_p)];
    avg.z.Regional_p = [zscore(avg.delta_p) zscore(avg.theta_p) zscore(avg.alpha_p) zscore(avg.beta_p) zscore(avg.gamma_p)];   
    
    
    [cca.delta_t,avg.delta_t] = Power_spectrum(channel_t, [0.5 4], cnt.fs);
    [cca.theta_t,avg.theta_t] = Power_spectrum(channel_t, [4 8], cnt.fs);
    [cca.alpha_t,avg.alpha_t] = Power_spectrum(channel_t, [8 13], cnt.fs);
    [cca.beta_t,avg.beta_t] = Power_spectrum(channel_t, [13 30], cnt.fs);
    [cca.gamma_t,avg.gamma_t] = Power_spectrum(channel_t, [30 40], cnt.fs);
    avg.delta_t = avg.delta_t'; avg.theta_t=avg.theta_t'; avg.alpha_t=avg.alpha_t'; avg.beta_t = avg.beta_t'; avg.gamma_t = avg.gamma_t';
    
    cca.Regional_t = [cca.delta_t cca.theta_t cca.alpha_t cca.beta_t cca.gamma_t];
    avg.Regional_t = [avg.delta_t avg.theta_t avg.alpha_t avg.beta_t avg.gamma_t];

    cca.RPL.delta_t = cca.delta_t ./ (cca.delta_t + cca.theta_t + cca.alpha_t + cca.beta_t + cca.gamma_t) ;
    avg.RPL.delta_t = avg.delta_t ./ sum(avg.Regional_t,2) ;
    cca.RPL.theta_t = cca.theta_t ./ (cca.delta_t + cca.theta_t + cca.alpha_t + cca.beta_t + cca.gamma_t) ;
    avg.RPL.theta_t = avg.theta_t ./ sum(avg.Regional_t,2) ;
    cca.RPL.alpha_t = cca.alpha_t ./ (cca.delta_t + cca.theta_t + cca.alpha_t + cca.beta_t + cca.gamma_t) ;
    avg.RPL.alpha_t = avg.alpha_t ./ sum(avg.Regional_t,2) ;
    cca.RPL.beta_t = cca.beta_t ./ (cca.delta_t + cca.theta_t + cca.alpha_t + cca.beta_t + cca.gamma_t) ;
    avg.RPL.beta_t = avg.beta_t ./ sum(avg.Regional_t,2) ;
    cca.RPL.gamma_t = cca.gamma_t ./ (cca.delta_t + cca.theta_t + cca.alpha_t + cca.beta_t + cca.gamma_t) ;
    avg.RPL.gamma_t = avg.gamma_t ./ sum(avg.Regional_t,2) ;     
    
    cca.RPL.Regional_t = [cca.RPL.delta_t cca.RPL.theta_t cca.RPL.alpha_t cca.RPL.beta_t cca.RPL.gamma_t]; 
    avg.RPL.Regional_t = [avg.RPL.delta_t avg.RPL.theta_t avg.RPL.alpha_t avg.RPL.beta_t avg.RPL.gamma_t];
    
    cca.z.Regional_t = [zscore(cca.delta_t) zscore(cca.theta_t) zscore(cca.alpha_t) zscore(cca.beta_t) zscore(cca.gamma_t)];
    avg.z.Regional_t = [zscore(avg.delta_t) zscore(avg.theta_t) zscore(avg.alpha_t) zscore(avg.beta_t) zscore(avg.gamma_t)];   
    
    
    [cca.delta_o,avg.delta_o] = Power_spectrum(channel_o, [0.5 4], cnt.fs);
    [cca.theta_o,avg.theta_o] = Power_spectrum(channel_o, [4 8], cnt.fs);
    [cca.alpha_o,avg.alpha_o] = Power_spectrum(channel_o, [8 13], cnt.fs);
    [cca.beta_o,avg.beta_o] = Power_spectrum(channel_o, [13 30], cnt.fs);
    [cca.gamma_o,avg.gamma_o] = Power_spectrum(channel_o, [30 40], cnt.fs);
    avg.delta_o = avg.delta_o'; avg.theta_o=avg.theta_o'; avg.alpha_o=avg.alpha_o'; avg.beta_o = avg.beta_o'; avg.gamma_o = avg.gamma_o';
    
    cca.Regional_o = [cca.delta_o cca.theta_o cca.alpha_o cca.beta_o cca.gamma_o];
    avg.Regional_o = [avg.delta_o avg.theta_o avg.alpha_o avg.beta_o avg.gamma_o];
    
    cca.RPL.delta_o = cca.delta_o ./ (cca.delta_o + cca.theta_o + cca.alpha_o + cca.beta_o + cca.gamma_o) ;
    avg.RPL.delta_o = avg.delta_o ./ sum(avg.Regional_o,2) ;
    cca.RPL.theta_o = cca.theta_o ./ (cca.delta_o + cca.theta_o + cca.alpha_o + cca.beta_o + cca.gamma_o) ;
    avg.RPL.theta_o = avg.theta_o ./ sum(avg.Regional_o,2) ;
    cca.RPL.alpha_o = cca.alpha_o ./ (cca.delta_o + cca.theta_o + cca.alpha_o + cca.beta_o + cca.gamma_o) ;
    avg.RPL.alpha_o = avg.alpha_o ./ sum(avg.Regional_o,2) ;
    cca.RPL.beta_o = cca.beta_o ./ (cca.delta_o + cca.theta_o + cca.alpha_o + cca.beta_o + cca.gamma_o) ;
    avg.RPL.beta_o = avg.beta_o ./ sum(avg.Regional_o,2) ;
    cca.RPL.gamma_o = cca.gamma_o ./ (cca.delta_o + cca.theta_o + cca.alpha_o + cca.beta_o + cca.gamma_o) ;
    avg.RPL.gamma_o = avg.gamma_o ./ sum(avg.Regional_o,2) ;     
    
    cca.RPL.Regional_o = [cca.RPL.delta_o cca.RPL.theta_o cca.RPL.alpha_o cca.RPL.beta_o cca.RPL.gamma_o]; 
    avg.RPL.Regional_o = [avg.RPL.delta_o avg.RPL.theta_o avg.RPL.alpha_o avg.RPL.beta_o avg.RPL.gamma_o];
    
    cca.z.Regional_o = [zscore(cca.delta_o) zscore(cca.theta_o) zscore(cca.alpha_o) zscore(cca.beta_o) zscore(cca.gamma_o)];
    avg.z.Regional_o = [zscore(avg.delta_o) zscore(avg.theta_o) zscore(avg.alpha_o) zscore(avg.beta_o) zscore(avg.gamma_o)];   
    
    
    
    
    % Regional Vertical Feature Extraction
    [cca.delta_f1,avg.delta_f1] = Power_spectrum(channel_f1, [0.5 4], cnt.fs);
    [cca.theta_f1,avg.theta_f1] = Power_spectrum(channel_f1, [4 8], cnt.fs);
    [cca.alpha_f1,avg.alpha_f1] = Power_spectrum(channel_f1, [8 13], cnt.fs);
    [cca.beta_f1,avg.beta_f1] = Power_spectrum(channel_f1, [13 30], cnt.fs);
    [cca.gamma_f1,avg.gamma_f1] = Power_spectrum(channel_f1, [30 40], cnt.fs);
    avg.delta_f1 = avg.delta_f1'; avg.theta_f1=avg.theta_f1'; avg.alpha_f1=avg.alpha_f1'; avg.beta_f1 = avg.beta_f1'; avg.gamma_f1 = avg.gamma_f1';
    
    cca.Regional_f1 = [cca.delta_f1 cca.theta_f1 cca.alpha_f1 cca.beta_f1 cca.gamma_f1];
    avg.Regional_f1 = [avg.delta_f1 avg.theta_f1 avg.alpha_f1 avg.beta_f1 avg.gamma_f1];
    
    cca.RPL.delta_f1 = cca.delta_f1 ./ (cca.delta_f1 + cca.theta_f1 + cca.alpha_f1 + cca.beta_f1 + cca.gamma_f1) ;
    avg.RPL.delta_f1 = avg.delta_f1 ./ sum(avg.Regional_f1,2) ;
    cca.RPL.theta_f1 = cca.theta_f1 ./ (cca.delta_f1 + cca.theta_f1 + cca.alpha_f1 + cca.beta_f1 + cca.gamma_f1) ;
    avg.RPL.theta_f1 = avg.theta_f1 ./ sum(avg.Regional_f1,2) ;
    cca.RPL.alpha_f1 = cca.alpha_f1 ./ (cca.delta_f1 + cca.theta_f1 + cca.alpha_f1 + cca.beta_f1 + cca.gamma_f1) ;
    avg.RPL.alpha_f1 = avg.alpha_f1 ./ sum(avg.Regional_f1,2) ;
    cca.RPL.beta_f1 = cca.beta_f1 ./ (cca.delta_f1 + cca.theta_f1 + cca.alpha_f1 + cca.beta_f1 + cca.gamma_f1) ;
    avg.RPL.beta_f1 = avg.beta_f1 ./ sum(avg.Regional_f1,2) ;
    cca.RPL.gamma_f1 = cca.gamma_f1 ./ (cca.delta_f1 + cca.theta_f1 + cca.alpha_f1 + cca.beta_f1 + cca.gamma_f1) ;
    avg.RPL.gamma_f1 = avg.gamma_f1 ./ sum(avg.Regional_f1,2) ;    
    
    cca.RPL.Regional_f1 = [cca.RPL.delta_f1 cca.RPL.theta_f1 cca.RPL.alpha_f1 cca.RPL.beta_f1 cca.RPL.gamma_f1]; 
    avg.RPL.Regional_f1 = [avg.RPL.delta_f1 avg.RPL.theta_f1 avg.RPL.alpha_f1 avg.RPL.beta_f1 avg.RPL.gamma_f1];
    
    cca.z.Regional_f1 = [zscore(cca.delta_f1) zscore(cca.theta_f1) zscore(cca.alpha_f1) zscore(cca.beta_f1) zscore(cca.gamma_f1)];
    avg.z.Regional_f1 = [zscore(avg.delta_f1) zscore(avg.theta_f1) zscore(avg.alpha_f1) zscore(avg.beta_f1) zscore(avg.gamma_f1)];   
    
    
    [cca.delta_f2,avg.delta_f2] = Power_spectrum(channel_f2, [0.5 4], cnt.fs);
    [cca.theta_f2,avg.theta_f2] = Power_spectrum(channel_f2, [4 8], cnt.fs);
    [cca.alpha_f2,avg.alpha_f2] = Power_spectrum(channel_f2, [8 13], cnt.fs);
    [cca.beta_f2,avg.beta_f2] = Power_spectrum(channel_f2, [13 30], cnt.fs);
    [cca.gamma_f2,avg.gamma_f2] = Power_spectrum(channel_f2, [30 40], cnt.fs);
    avg.delta_f2 = avg.delta_f2'; avg.theta_f2=avg.theta_f2'; avg.alpha_f2=avg.alpha_f2'; avg.beta_f2 = avg.beta_f2'; avg.gamma_f2 = avg.gamma_f2';
    
    cca.Regional_f2 = [cca.delta_f2 cca.theta_f2 cca.alpha_f2 cca.beta_f2 cca.gamma_f2];
    avg.Regional_f2 = [avg.delta_f2 avg.theta_f2 avg.alpha_f2 avg.beta_f2 avg.gamma_f2];
    
    cca.RPL.delta_f2 = cca.delta_f2 ./ (cca.delta_f2 + cca.theta_f2 + cca.alpha_f2 + cca.beta_f2 + cca.gamma_f2) ;
    avg.RPL.delta_f2 = avg.delta_f2 ./ sum(avg.Regional_f2,2) ;
    cca.RPL.theta_f2 = cca.theta_f2 ./ (cca.delta_f2 + cca.theta_f2 + cca.alpha_f2 + cca.beta_f2 + cca.gamma_f2) ;
    avg.RPL.theta_f2 = avg.theta_f2 ./ sum(avg.Regional_f2,2) ;
    cca.RPL.alpha_f2 = cca.alpha_f2 ./ (cca.delta_f2 + cca.theta_f2 + cca.alpha_f2 + cca.beta_f2 + cca.gamma_f2) ;
    avg.RPL.alpha_f2 = avg.alpha_f2 ./ sum(avg.Regional_f2,2) ;
    cca.RPL.beta_f2 = cca.beta_f2 ./ (cca.delta_f2 + cca.theta_f2 + cca.alpha_f2 + cca.beta_f2 + cca.gamma_f2) ;
    avg.RPL.beta_f2 = avg.beta_f2 ./ sum(avg.Regional_f2,2) ;
    cca.RPL.gamma_f2 = cca.gamma_f2 ./ (cca.delta_f2 + cca.theta_f2 + cca.alpha_f2 + cca.beta_f2 + cca.gamma_f2) ;
    avg.RPL.gamma_f2 = avg.gamma_f2 ./ sum(avg.Regional_f2,2) ;    
    
    cca.RPL.Regional_f2 = [cca.RPL.delta_f2 cca.RPL.theta_f2 cca.RPL.alpha_f2 cca.RPL.beta_f2 cca.RPL.gamma_f2]; 
    avg.RPL.Regional_f2 = [avg.RPL.delta_f2 avg.RPL.theta_f2 avg.RPL.alpha_f2 avg.RPL.beta_f2 avg.RPL.gamma_f2];
    
    cca.z.Regional_f2 = [zscore(cca.delta_f2) zscore(cca.theta_f2) zscore(cca.alpha_f2) zscore(cca.beta_f2) zscore(cca.gamma_f2)];
    avg.z.Regional_f2 = [zscore(avg.delta_f2) zscore(avg.theta_f2) zscore(avg.alpha_f2) zscore(avg.beta_f2) zscore(avg.gamma_f2)];       
    
    
    [cca.delta_f3,avg.delta_f3] = Power_spectrum(channel_f3, [0.5 4], cnt.fs);
    [cca.theta_f3,avg.theta_f3] = Power_spectrum(channel_f3, [4 8], cnt.fs);
    [cca.alpha_f3,avg.alpha_f3] = Power_spectrum(channel_f3, [8 13], cnt.fs);
    [cca.beta_f3,avg.beta_f3] = Power_spectrum(channel_f3, [13 30], cnt.fs);
    [cca.gamma_f3,avg.gamma_f3] = Power_spectrum(channel_f3, [30 40], cnt.fs);
    avg.delta_f3 = avg.delta_f3'; avg.theta_f3=avg.theta_f3'; avg.alpha_f3=avg.alpha_f3'; avg.beta_f3 = avg.beta_f3'; avg.gamma_f3 = avg.gamma_f3';
    
    cca.Regional_f3 = [cca.delta_f3 cca.theta_f3 cca.alpha_f3 cca.beta_f3 cca.gamma_f3];
    avg.Regional_f3 = [avg.delta_f3 avg.theta_f3 avg.alpha_f3 avg.beta_f3 avg.gamma_f3];
    
    cca.RPL.delta_f3 = cca.delta_f3 ./ (cca.delta_f3 + cca.theta_f3 + cca.alpha_f3 + cca.beta_f3 + cca.gamma_f3) ;
    avg.RPL.delta_f3 = avg.delta_f3 ./ sum(avg.Regional_f3,2) ;
    cca.RPL.theta_f3 = cca.theta_f3 ./ (cca.delta_f3 + cca.theta_f3 + cca.alpha_f3 + cca.beta_f3 + cca.gamma_f3) ;
    avg.RPL.theta_f3 = avg.theta_f3 ./ sum(avg.Regional_f3,2) ;
    cca.RPL.alpha_f3 = cca.alpha_f3 ./ (cca.delta_f3 + cca.theta_f3 + cca.alpha_f3 + cca.beta_f3 + cca.gamma_f3) ;
    avg.RPL.alpha_f3 = avg.alpha_f3 ./ sum(avg.Regional_f3,2) ;
    cca.RPL.beta_f3 = cca.beta_f3 ./ (cca.delta_f3 + cca.theta_f3 + cca.alpha_f3 + cca.beta_f3 + cca.gamma_f3) ;
    avg.RPL.beta_f3 = avg.beta_f3 ./ sum(avg.Regional_f3,2) ;
    cca.RPL.gamma_f3 = cca.gamma_f3 ./ (cca.delta_f3 + cca.theta_f3 + cca.alpha_f3 + cca.beta_f3 + cca.gamma_f3) ;
    avg.RPL.gamma_f3 = avg.gamma_f3 ./ sum(avg.Regional_f3,2) ;    
    
    cca.RPL.Regional_f3 = [cca.RPL.delta_f3 cca.RPL.theta_f3 cca.RPL.alpha_f3 cca.RPL.beta_f3 cca.RPL.gamma_f3]; 
    avg.RPL.Regional_f3 = [avg.RPL.delta_f3 avg.RPL.theta_f3 avg.RPL.alpha_f3 avg.RPL.beta_f3 avg.RPL.gamma_f3];
    
    cca.z.Regional_f3 = [zscore(cca.delta_f3) zscore(cca.theta_f3) zscore(cca.alpha_f3) zscore(cca.beta_f3) zscore(cca.gamma_f3)];
    avg.z.Regional_f3 = [zscore(avg.delta_f3) zscore(avg.theta_f3) zscore(avg.alpha_f3) zscore(avg.beta_f3) zscore(avg.gamma_f3)];           
    
    
    [cca.delta_c1,avg.delta_c1] = Power_spectrum(channel_c1, [0.5 4], cnt.fs);
    [cca.theta_c1,avg.theta_c1] = Power_spectrum(channel_c1, [4 8], cnt.fs);
    [cca.alpha_c1,avg.alpha_c1] = Power_spectrum(channel_c1, [8 13], cnt.fs);
    [cca.beta_c1,avg.beta_c1] = Power_spectrum(channel_c1, [13 30], cnt.fs);
    [cca.gamma_c1,avg.gamma_c1] = Power_spectrum(channel_c1, [30 40], cnt.fs);
    avg.delta_c1 = avg.delta_c1'; avg.theta_c1=avg.theta_c1'; avg.alpha_c1=avg.alpha_c1'; avg.beta_c1 = avg.beta_c1'; avg.gamma_c1 = avg.gamma_c1';
    
    cca.Regional_c1 = [cca.delta_c1 cca.theta_c1 cca.alpha_c1 cca.beta_c1 cca.gamma_c1];
    avg.Regional_c1 = [avg.delta_c1 avg.theta_c1 avg.alpha_c1 avg.beta_c1 avg.gamma_c1];
    
    cca.RPL.delta_c1 = cca.delta_c1 ./ (cca.delta_c1 + cca.theta_c1 + cca.alpha_c1 + cca.beta_c1 + cca.gamma_c1) ;
    avg.RPL.delta_c1 = avg.delta_c1 ./ sum(avg.Regional_c1,2) ;
    cca.RPL.theta_c1 = cca.theta_c1 ./ (cca.delta_c1 + cca.theta_c1 + cca.alpha_c1 + cca.beta_c1 + cca.gamma_c1) ;
    avg.RPL.theta_c1 = avg.theta_c1 ./ sum(avg.Regional_c1,2) ;
    cca.RPL.alpha_c1 = cca.alpha_c1 ./ (cca.delta_c1 + cca.theta_c1 + cca.alpha_c1 + cca.beta_c1 + cca.gamma_c1) ;
    avg.RPL.alpha_c1 = avg.alpha_c1 ./ sum(avg.Regional_c1,2) ;
    cca.RPL.beta_c1 = cca.beta_c1 ./ (cca.delta_c1 + cca.theta_c1 + cca.alpha_c1 + cca.beta_c1 + cca.gamma_c1) ;
    avg.RPL.beta_c1 = avg.beta_c1 ./ sum(avg.Regional_c1,2) ;
    cca.RPL.gamma_c1 = cca.gamma_c1 ./ (cca.delta_c1 + cca.theta_c1 + cca.alpha_c1 + cca.beta_c1 + cca.gamma_c1) ;
    avg.RPL.gamma_c1 = avg.gamma_c1 ./ sum(avg.Regional_c1,2) ;     
    
    cca.RPL.Regional_c1 = [cca.RPL.delta_c1 cca.RPL.theta_c1 cca.RPL.alpha_c1 cca.RPL.beta_c1 cca.RPL.gamma_c1]; 
    avg.RPL.Regional_c1 = [avg.RPL.delta_c1 avg.RPL.theta_c1 avg.RPL.alpha_c1 avg.RPL.beta_c1 avg.RPL.gamma_c1];
    
    cca.z.Regional_c1 = [zscore(cca.delta_c1) zscore(cca.theta_c1) zscore(cca.alpha_c1) zscore(cca.beta_c1) zscore(cca.gamma_c1)];
    avg.z.Regional_c1 = [zscore(avg.delta_c1) zscore(avg.theta_c1) zscore(avg.alpha_c1) zscore(avg.beta_c1) zscore(avg.gamma_c1)];               
    
    [cca.delta_c2,avg.delta_c2] = Power_spectrum(channel_c2, [0.5 4], cnt.fs);
    [cca.theta_c2,avg.theta_c2] = Power_spectrum(channel_c2, [4 8], cnt.fs);
    [cca.alpha_c2,avg.alpha_c2] = Power_spectrum(channel_c2, [8 13], cnt.fs);
    [cca.beta_c2,avg.beta_c2] = Power_spectrum(channel_c2, [13 30], cnt.fs);
    [cca.gamma_c2,avg.gamma_c2] = Power_spectrum(channel_c2, [30 40], cnt.fs);
    avg.delta_c2 = avg.delta_c2'; avg.theta_c2=avg.theta_c2'; avg.alpha_c2=avg.alpha_c2'; avg.beta_c2 = avg.beta_c2'; avg.gamma_c2 = avg.gamma_c2';
    
    cca.Regional_c2 = [cca.delta_c2 cca.theta_c2 cca.alpha_c2 cca.beta_c2 cca.gamma_c2];
    avg.Regional_c2 = [avg.delta_c2 avg.theta_c2 avg.alpha_c2 avg.beta_c2 avg.gamma_c2];
    
    cca.RPL.delta_c2 = cca.delta_c2 ./ (cca.delta_c2 + cca.theta_c2 + cca.alpha_c2 + cca.beta_c2 + cca.gamma_c2) ;
    avg.RPL.delta_c2 = avg.delta_c2 ./ sum(avg.Regional_c2,2) ;
    cca.RPL.theta_c2 = cca.theta_c2 ./ (cca.delta_c2 + cca.theta_c2 + cca.alpha_c2 + cca.beta_c2 + cca.gamma_c2) ;
    avg.RPL.theta_c2 = avg.theta_c2 ./ sum(avg.Regional_c2,2) ;
    cca.RPL.alpha_c2 = cca.alpha_c2 ./ (cca.delta_c2 + cca.theta_c2 + cca.alpha_c2 + cca.beta_c2 + cca.gamma_c2) ;
    avg.RPL.alpha_c2 = avg.alpha_c2 ./ sum(avg.Regional_c2,2) ;
    cca.RPL.beta_c2 = cca.beta_c2 ./ (cca.delta_c2 + cca.theta_c2 + cca.alpha_c2 + cca.beta_c2 + cca.gamma_c2) ;
    avg.RPL.beta_c2 = avg.beta_c2 ./ sum(avg.Regional_c2,2) ;
    cca.RPL.gamma_c2 = cca.gamma_c2 ./ (cca.delta_c2 + cca.theta_c2 + cca.alpha_c2 + cca.beta_c2 + cca.gamma_c2) ;
    avg.RPL.gamma_c2 = avg.gamma_c2 ./ sum(avg.Regional_c2,2) ;     
    
    cca.RPL.Regional_c2 = [cca.RPL.delta_c2 cca.RPL.theta_c2 cca.RPL.alpha_c2 cca.RPL.beta_c2 cca.RPL.gamma_c2]; 
    avg.RPL.Regional_c2 = [avg.RPL.delta_c2 avg.RPL.theta_c2 avg.RPL.alpha_c2 avg.RPL.beta_c2 avg.RPL.gamma_c2];
    
    cca.z.Regional_c2 = [zscore(cca.delta_c2) zscore(cca.theta_c2) zscore(cca.alpha_c2) zscore(cca.beta_c2) zscore(cca.gamma_c2)];
    avg.z.Regional_c2 = [zscore(avg.delta_c2) zscore(avg.theta_c2) zscore(avg.alpha_c2) zscore(avg.beta_c2) zscore(avg.gamma_c2)];                   
    
    [cca.delta_c3,avg.delta_c3] = Power_spectrum(channel_c3, [0.5 4], cnt.fs);
    [cca.theta_c3,avg.theta_c3] = Power_spectrum(channel_c3, [4 8], cnt.fs);
    [cca.alpha_c3,avg.alpha_c3] = Power_spectrum(channel_c3, [8 13], cnt.fs);
    [cca.beta_c3,avg.beta_c3] = Power_spectrum(channel_c3, [13 30], cnt.fs);
    [cca.gamma_c3,avg.gamma_c3] = Power_spectrum(channel_c3, [30 40], cnt.fs);
    avg.delta_c3 = avg.delta_c3'; avg.theta_c3=avg.theta_c3'; avg.alpha_c3=avg.alpha_c3'; avg.beta_c3 = avg.beta_c3'; avg.gamma_c3 = avg.gamma_c3';
    
    cca.Regional_c3 = [cca.delta_c3 cca.theta_c3 cca.alpha_c3 cca.beta_c3 cca.gamma_c3];
    avg.Regional_c3 = [avg.delta_c3 avg.theta_c3 avg.alpha_c3 avg.beta_c3 avg.gamma_c3];
    
    cca.RPL.delta_c3 = cca.delta_c3 ./ (cca.delta_c3 + cca.theta_c3 + cca.alpha_c3 + cca.beta_c3 + cca.gamma_c3) ;
    avg.RPL.delta_c3 = avg.delta_c3 ./ sum(avg.Regional_c3,2) ;
    cca.RPL.theta_c3 = cca.theta_c3 ./ (cca.delta_c3 + cca.theta_c3 + cca.alpha_c3 + cca.beta_c3 + cca.gamma_c3) ;
    avg.RPL.theta_c3 = avg.theta_c3 ./ sum(avg.Regional_c3,2) ;
    cca.RPL.alpha_c3 = cca.alpha_c3 ./ (cca.delta_c3 + cca.theta_c3 + cca.alpha_c3 + cca.beta_c3 + cca.gamma_c3) ;
    avg.RPL.alpha_c3 = avg.alpha_c3 ./ sum(avg.Regional_c3,2) ;
    cca.RPL.beta_c3 = cca.beta_c3 ./ (cca.delta_c3 + cca.theta_c3 + cca.alpha_c3 + cca.beta_c3 + cca.gamma_c3) ;
    avg.RPL.beta_c3 = avg.beta_c3 ./ sum(avg.Regional_c3,2) ;
    cca.RPL.gamma_c3 = cca.gamma_c3 ./ (cca.delta_c3 + cca.theta_c3 + cca.alpha_c3 + cca.beta_c3 + cca.gamma_c3) ;
    avg.RPL.gamma_c3 = avg.gamma_c3 ./ sum(avg.Regional_c3,2) ;     
    
    cca.RPL.Regional_c3 = [cca.RPL.delta_c3 cca.RPL.theta_c3 cca.RPL.alpha_c3 cca.RPL.beta_c3 cca.RPL.gamma_c3]; 
    avg.RPL.Regional_c3 = [avg.RPL.delta_c3 avg.RPL.theta_c3 avg.RPL.alpha_c3 avg.RPL.beta_c3 avg.RPL.gamma_c3];
    
    cca.z.Regional_c3 = [zscore(cca.delta_c3) zscore(cca.theta_c3) zscore(cca.alpha_c3) zscore(cca.beta_c3) zscore(cca.gamma_c3)];
    avg.z.Regional_c3 = [zscore(avg.delta_c3) zscore(avg.theta_c3) zscore(avg.alpha_c3) zscore(avg.beta_c3) zscore(avg.gamma_c3)];                   
    
    [cca.delta_p1,avg.delta_p1] = Power_spectrum(channel_p1, [0.5 4], cnt.fs);
    [cca.theta_p1,avg.theta_p1] = Power_spectrum(channel_p1, [4 8], cnt.fs);
    [cca.alpha_p1,avg.alpha_p1] = Power_spectrum(channel_p1, [8 13], cnt.fs);
    [cca.beta_p1,avg.beta_p1] = Power_spectrum(channel_p1, [13 30], cnt.fs);
    [cca.gamma_p1,avg.gamma_p1] = Power_spectrum(channel_p1, [30 40], cnt.fs);
    avg.delta_p1 = avg.delta_p1'; avg.theta_p1=avg.theta_p1'; avg.alpha_p1=avg.alpha_p1'; avg.beta_p1 = avg.beta_p1'; avg.gamma_p1 = avg.gamma_p1';
    
    cca.Regional_p1 = [cca.delta_p1 cca.theta_p1 cca.alpha_p1 cca.beta_p1 cca.gamma_p1];
    avg.Regional_p1 = [avg.delta_p1 avg.theta_p1 avg.alpha_p1 avg.beta_p1 avg.gamma_p1];
    
    cca.RPL.delta_p1 = cca.delta_p1 ./ (cca.delta_p1 + cca.theta_p1 + cca.alpha_p1 + cca.beta_p1 + cca.gamma_p1) ;
    avg.RPL.delta_p1 = avg.delta_p1 ./ sum(avg.Regional_p1,2) ;
    cca.RPL.theta_p1 = cca.theta_p1 ./ (cca.delta_p1 + cca.theta_p1 + cca.alpha_p1 + cca.beta_p1 + cca.gamma_p1) ;
    avg.RPL.theta_p1 = avg.theta_p1 ./ sum(avg.Regional_p1,2) ;
    cca.RPL.alpha_p1 = cca.alpha_p1 ./ (cca.delta_p1 + cca.theta_p1 + cca.alpha_p1 + cca.beta_p1 + cca.gamma_p1) ;
    avg.RPL.alpha_p1 = avg.alpha_p1 ./ sum(avg.Regional_p1,2) ;
    cca.RPL.beta_p1 = cca.beta_p1 ./ (cca.delta_p1 + cca.theta_p1 + cca.alpha_p1 + cca.beta_p1 + cca.gamma_p1) ;
    avg.RPL.beta_p1 = avg.beta_p1 ./ sum(avg.Regional_p1,2) ;
    cca.RPL.gamma_p1 = cca.gamma_p1 ./ (cca.delta_p1 + cca.theta_p1 + cca.alpha_p1 + cca.beta_p1 + cca.gamma_p1) ;
    avg.RPL.gamma_p1 = avg.gamma_p1 ./ sum(avg.Regional_p1,2) ;     
    
    cca.RPL.Regional_p1 = [cca.RPL.delta_p1 cca.RPL.theta_p1 cca.RPL.alpha_p1 cca.RPL.beta_p1 cca.RPL.gamma_p1]; 
    avg.RPL.Regional_p1 = [avg.RPL.delta_p1 avg.RPL.theta_p1 avg.RPL.alpha_p1 avg.RPL.beta_p1 avg.RPL.gamma_p1];
    
    cca.z.Regional_p1 = [zscore(cca.delta_p1) zscore(cca.theta_p1) zscore(cca.alpha_p1) zscore(cca.beta_p1) zscore(cca.gamma_p1)];
    avg.z.Regional_p1 = [zscore(avg.delta_p1) zscore(avg.theta_p1) zscore(avg.alpha_p1) zscore(avg.beta_p1) zscore(avg.gamma_p1)];               
    
    [cca.delta_p2,avg.delta_p2] = Power_spectrum(channel_p2, [0.5 4], cnt.fs);
    [cca.theta_p2,avg.theta_p2] = Power_spectrum(channel_p2, [4 8], cnt.fs);
    [cca.alpha_p2,avg.alpha_p2] = Power_spectrum(channel_p2, [8 13], cnt.fs);
    [cca.beta_p2,avg.beta_p2] = Power_spectrum(channel_p2, [13 30], cnt.fs);
    [cca.gamma_p2,avg.gamma_p2] = Power_spectrum(channel_p2, [30 40], cnt.fs);
    avg.delta_p2 = avg.delta_p2'; avg.theta_p2=avg.theta_p2'; avg.alpha_p2=avg.alpha_p2'; avg.beta_p2 = avg.beta_p2'; avg.gamma_p2 = avg.gamma_p2';
    
    cca.Regional_p2 = [cca.delta_p2 cca.theta_p2 cca.alpha_p2 cca.beta_p2 cca.gamma_p2];
    avg.Regional_p2 = [avg.delta_p2 avg.theta_p2 avg.alpha_p2 avg.beta_p2 avg.gamma_p2];
    
    cca.RPL.delta_p2 = cca.delta_p2 ./ (cca.delta_p2 + cca.theta_p2 + cca.alpha_p2 + cca.beta_p2 + cca.gamma_p2) ;
    avg.RPL.delta_p2 = avg.delta_p2 ./ sum(avg.Regional_p2,2) ;
    cca.RPL.theta_p2 = cca.theta_p2 ./ (cca.delta_p2 + cca.theta_p2 + cca.alpha_p2 + cca.beta_p2 + cca.gamma_p2) ;
    avg.RPL.theta_p2 = avg.theta_p2 ./ sum(avg.Regional_p2,2) ;
    cca.RPL.alpha_p2 = cca.alpha_p2 ./ (cca.delta_p2 + cca.theta_p2 + cca.alpha_p2 + cca.beta_p2 + cca.gamma_p2) ;
    avg.RPL.alpha_p2 = avg.alpha_p2 ./ sum(avg.Regional_p2,2) ;
    cca.RPL.beta_p2 = cca.beta_p2 ./ (cca.delta_p2 + cca.theta_p2 + cca.alpha_p2 + cca.beta_p2 + cca.gamma_p2) ;
    avg.RPL.beta_p2 = avg.beta_p2 ./ sum(avg.Regional_p2,2) ;
    cca.RPL.gamma_p2 = cca.gamma_p2 ./ (cca.delta_p2 + cca.theta_p2 + cca.alpha_p2 + cca.beta_p2 + cca.gamma_p2) ;
    avg.RPL.gamma_p2 = avg.gamma_p2 ./ sum(avg.Regional_p2,2) ;     
    
    cca.RPL.Regional_p2 = [cca.RPL.delta_p2 cca.RPL.theta_p2 cca.RPL.alpha_p2 cca.RPL.beta_p2 cca.RPL.gamma_p2]; 
    avg.RPL.Regional_p2 = [avg.RPL.delta_p2 avg.RPL.theta_p2 avg.RPL.alpha_p2 avg.RPL.beta_p2 avg.RPL.gamma_p2];
    
    cca.z.Regional_p2 = [zscore(cca.delta_p2) zscore(cca.theta_p2) zscore(cca.alpha_p2) zscore(cca.beta_p2) zscore(cca.gamma_p2)];
    avg.z.Regional_p2 = [zscore(avg.delta_p2) zscore(avg.theta_p2) zscore(avg.alpha_p2) zscore(avg.beta_p2) zscore(avg.gamma_p2)];               
    
    
    [cca.delta_p3,avg.delta_p3] = Power_spectrum(channel_p3, [0.5 4], cnt.fs);
    [cca.theta_p3,avg.theta_p3] = Power_spectrum(channel_p3, [4 8], cnt.fs);
    [cca.alpha_p3,avg.alpha_p3] = Power_spectrum(channel_p3, [8 13], cnt.fs);
    [cca.beta_p3,avg.beta_p3] = Power_spectrum(channel_p3, [13 30], cnt.fs);
    [cca.gamma_p3,avg.gamma_p3] = Power_spectrum(channel_p3, [30 40], cnt.fs);
    avg.delta_p3 = avg.delta_p3'; avg.theta_p3=avg.theta_p3'; avg.alpha_p3=avg.alpha_p3'; avg.beta_p3 = avg.beta_p3'; avg.gamma_p3 = avg.gamma_p3';
    
    cca.Regional_p3 = [cca.delta_p3 cca.theta_p3 cca.alpha_p3 cca.beta_p3 cca.gamma_p3];
    avg.Regional_p3 = [avg.delta_p3 avg.theta_p3 avg.alpha_p3 avg.beta_p3 avg.gamma_p3];
    
    cca.RPL.delta_p3 = cca.delta_p3 ./ (cca.delta_p3 + cca.theta_p3 + cca.alpha_p3 + cca.beta_p3 + cca.gamma_p3) ;
    avg.RPL.delta_p3 = avg.delta_p3 ./ sum(avg.Regional_p3,2) ;
    cca.RPL.theta_p3 = cca.theta_p3 ./ (cca.delta_p3 + cca.theta_p3 + cca.alpha_p3 + cca.beta_p3 + cca.gamma_p3) ;
    avg.RPL.theta_p3 = avg.theta_p3 ./ sum(avg.Regional_p3,2) ;
    cca.RPL.alpha_p3 = cca.alpha_p3 ./ (cca.delta_p3 + cca.theta_p3 + cca.alpha_p3 + cca.beta_p3 + cca.gamma_p3) ;
    avg.RPL.alpha_p3 = avg.alpha_p3 ./ sum(avg.Regional_p3,2) ;
    cca.RPL.beta_p3 = cca.beta_p3 ./ (cca.delta_p3 + cca.theta_p3 + cca.alpha_p3 + cca.beta_p3 + cca.gamma_p3) ;
    avg.RPL.beta_p3 = avg.beta_p3 ./ sum(avg.Regional_p3,2) ;
    cca.RPL.gamma_p3 = cca.gamma_p3 ./ (cca.delta_p3 + cca.theta_p3 + cca.alpha_p3 + cca.beta_p3 + cca.gamma_p3) ;
    avg.RPL.gamma_p3 = avg.gamma_p3 ./ sum(avg.Regional_p3,2) ;     
    
    cca.RPL.Regional_p3 = [cca.RPL.delta_p3 cca.RPL.theta_p3 cca.RPL.alpha_p3 cca.RPL.beta_p3 cca.RPL.gamma_p3]; 
    avg.RPL.Regional_p3 = [avg.RPL.delta_p3 avg.RPL.theta_p3 avg.RPL.alpha_p3 avg.RPL.beta_p3 avg.RPL.gamma_p3];
    
    cca.z.Regional_p3 = [zscore(cca.delta_p3) zscore(cca.theta_p3) zscore(cca.alpha_p3) zscore(cca.beta_p3) zscore(cca.gamma_p3)];
    avg.z.Regional_p3 = [zscore(avg.delta_p3) zscore(avg.theta_p3) zscore(avg.alpha_p3) zscore(avg.beta_p3) zscore(avg.gamma_p3)];               
    
    
    [cca.delta_t1,avg.delta_t1] = Power_spectrum(channel_t1, [0.5 4], cnt.fs);
    [cca.theta_t1,avg.theta_t1] = Power_spectrum(channel_t1, [4 8], cnt.fs);
    [cca.alpha_t1,avg.alpha_t1] = Power_spectrum(channel_t1, [8 13], cnt.fs);
    [cca.beta_t1,avg.beta_t1] = Power_spectrum(channel_t1, [13 30], cnt.fs);
    [cca.gamma_t1,avg.gamma_t1] = Power_spectrum(channel_t1, [30 40], cnt.fs);
    avg.delta_t1 = avg.delta_t1'; avg.theta_t1=avg.theta_t1'; avg.alpha_t1=avg.alpha_t1'; avg.beta_t1 = avg.beta_t1'; avg.gamma_t1 = avg.gamma_t1';
    
    cca.Regional_t1 = [cca.delta_t1 cca.theta_t1 cca.alpha_t1 cca.beta_t1 cca.gamma_t1];
    avg.Regional_t1 = [avg.delta_t1 avg.theta_t1 avg.alpha_t1 avg.beta_t1 avg.gamma_t1];
    
    cca.RPL.delta_t1 = cca.delta_t1 ./ (cca.delta_t1 + cca.theta_t1 + cca.alpha_t1 + cca.beta_t1 + cca.gamma_t1) ;
    avg.RPL.delta_t1 = avg.delta_t1 ./ sum(avg.Regional_t1,2) ;
    cca.RPL.theta_t1 = cca.theta_t1 ./ (cca.delta_t1 + cca.theta_t1 + cca.alpha_t1 + cca.beta_t1 + cca.gamma_t1) ;
    avg.RPL.theta_t1 = avg.theta_t1 ./ sum(avg.Regional_t1,2) ;
    cca.RPL.alpha_t1 = cca.alpha_t1 ./ (cca.delta_t1 + cca.theta_t1 + cca.alpha_t1 + cca.beta_t1 + cca.gamma_t1) ;
    avg.RPL.alpha_t1 = avg.alpha_t1 ./ sum(avg.Regional_t1,2) ;
    cca.RPL.beta_t1 = cca.beta_t1 ./ (cca.delta_t1 + cca.theta_t1 + cca.alpha_t1 + cca.beta_t1 + cca.gamma_t1) ;
    avg.RPL.beta_t1 = avg.beta_t1 ./ sum(avg.Regional_t1,2) ;
    cca.RPL.gamma_t1 = cca.gamma_t1 ./ (cca.delta_t1 + cca.theta_t1 + cca.alpha_t1 + cca.beta_t1 + cca.gamma_t1) ;
    avg.RPL.gamma_t1 = avg.gamma_t1 ./ sum(avg.Regional_t1,2) ;     

    cca.RPL.Regional_t1 = [cca.RPL.delta_t1 cca.RPL.theta_t1 cca.RPL.alpha_t1 cca.RPL.beta_t1 cca.RPL.gamma_t1]; 
    avg.RPL.Regional_t1 = [avg.RPL.delta_t1 avg.RPL.theta_t1 avg.RPL.alpha_t1 avg.RPL.beta_t1 avg.RPL.gamma_t1];
    
    cca.z.Regional_t1 = [zscore(cca.delta_t1) zscore(cca.theta_t1) zscore(cca.alpha_t1) zscore(cca.beta_t1) zscore(cca.gamma_t1)];
    avg.z.Regional_t1 = [zscore(avg.delta_t1) zscore(avg.theta_t1) zscore(avg.alpha_t1) zscore(avg.beta_t1) zscore(avg.gamma_t1)];                   
    
    
    [cca.delta_t2,avg.delta_t2] = Power_spectrum(channel_t2, [0.5 4], cnt.fs);
    [cca.theta_t2,avg.theta_t2] = Power_spectrum(channel_t2, [4 8], cnt.fs);
    [cca.alpha_t2,avg.alpha_t2] = Power_spectrum(channel_t2, [8 13], cnt.fs);
    [cca.beta_t2,avg.beta_t2] = Power_spectrum(channel_t2, [13 30], cnt.fs);
    [cca.gamma_t2,avg.gamma_t2] = Power_spectrum(channel_t2, [30 40], cnt.fs);
    avg.delta_t2 = avg.delta_t2'; avg.theta_t2=avg.theta_t2'; avg.alpha_t2=avg.alpha_t2'; avg.beta_t2 = avg.beta_t2'; avg.gamma_t2 = avg.gamma_t2';
    
    cca.Regional_t2 = [cca.delta_t2 cca.theta_t2 cca.alpha_t2 cca.beta_t2 cca.gamma_t2];
    avg.Regional_t2 = [avg.delta_t2 avg.theta_t2 avg.alpha_t2 avg.beta_t2 avg.gamma_t2];
    
    cca.RPL.delta_t2 = cca.delta_t2 ./ (cca.delta_t2 + cca.theta_t2 + cca.alpha_t2 + cca.beta_t2 + cca.gamma_t2) ;
    avg.RPL.delta_t2 = avg.delta_t2 ./ sum(avg.Regional_t2,2) ;
    cca.RPL.theta_t2 = cca.theta_t2 ./ (cca.delta_t2 + cca.theta_t2 + cca.alpha_t2 + cca.beta_t2 + cca.gamma_t2) ;
    avg.RPL.theta_t2 = avg.theta_t2 ./ sum(avg.Regional_t2,2) ;
    cca.RPL.alpha_t2 = cca.alpha_t2 ./ (cca.delta_t2 + cca.theta_t2 + cca.alpha_t2 + cca.beta_t2 + cca.gamma_t2) ;
    avg.RPL.alpha_t2 = avg.alpha_t2 ./ sum(avg.Regional_t2,2) ;
    cca.RPL.beta_t2 = cca.beta_t2 ./ (cca.delta_t2 + cca.theta_t2 + cca.alpha_t2 + cca.beta_t2 + cca.gamma_t2) ;
    avg.RPL.beta_t2 = avg.beta_t2 ./ sum(avg.Regional_t2,2) ;
    cca.RPL.gamma_t2 = cca.gamma_t2 ./ (cca.delta_t2 + cca.theta_t2 + cca.alpha_t2 + cca.beta_t2 + cca.gamma_t2) ;
    avg.RPL.gamma_t2 = avg.gamma_t2 ./ sum(avg.Regional_t2,2) ;     
    
    cca.RPL.Regional_t2 = [cca.RPL.delta_t2 cca.RPL.theta_t2 cca.RPL.alpha_t2 cca.RPL.beta_t2 cca.RPL.gamma_t2]; 
    avg.RPL.Regional_t2 = [avg.RPL.delta_t2 avg.RPL.theta_t2 avg.RPL.alpha_t2 avg.RPL.beta_t2 avg.RPL.gamma_t2];
    
    cca.z.Regional_t2 = [zscore(cca.delta_t2) zscore(cca.theta_t2) zscore(cca.alpha_t2) zscore(cca.beta_t2) zscore(cca.gamma_t2)];
    avg.z.Regional_t2 = [zscore(avg.delta_t2) zscore(avg.theta_t2) zscore(avg.alpha_t2) zscore(avg.beta_t2) zscore(avg.gamma_t2)];                   
    
    
    
    
    
    [cca.delta_o1,avg.delta_o1] = Power_spectrum(channel_o1, [0.5 4], cnt.fs);
    [cca.theta_o1,avg.theta_o1] = Power_spectrum(channel_o1, [4 8], cnt.fs);
    [cca.alpha_o1,avg.alpha_o1] = Power_spectrum(channel_o1, [8 13], cnt.fs);
    [cca.beta_o1,avg.beta_o1] = Power_spectrum(channel_o1, [13 30], cnt.fs);
    [cca.gamma_o1,avg.gamma_o1] = Power_spectrum(channel_o1, [30 40], cnt.fs);
    avg.delta_o1 = avg.delta_o1'; avg.theta_o1=avg.theta_o1'; avg.alpha_o1=avg.alpha_o1'; avg.beta_o1 = avg.beta_o1'; avg.gamma_o1 = avg.gamma_o1';
    
    cca.Regional_o1 = [cca.delta_o1 cca.theta_o1 cca.alpha_o1 cca.beta_o1 cca.gamma_o1];
    avg.Regional_o1 = [avg.delta_o1 avg.theta_o1 avg.alpha_o1 avg.beta_o1 avg.gamma_o1];

    cca.RPL.delta_o1 = cca.delta_o1 ./ (cca.delta_o1 + cca.theta_o1 + cca.alpha_o1 + cca.beta_o1 + cca.gamma_o1) ;
    avg.RPL.delta_o1 = avg.delta_o1 ./ sum(avg.Regional_o1,2) ;
    cca.RPL.theta_o1 = cca.theta_o1 ./ (cca.delta_o1 + cca.theta_o1 + cca.alpha_o1 + cca.beta_o1 + cca.gamma_o1) ;
    avg.RPL.theta_o1 = avg.theta_o1 ./ sum(avg.Regional_o1,2) ;
    cca.RPL.alpha_o1 = cca.alpha_o1 ./ (cca.delta_o1 + cca.theta_o1 + cca.alpha_o1 + cca.beta_o1 + cca.gamma_o1) ;
    avg.RPL.alpha_o1 = avg.alpha_o1 ./ sum(avg.Regional_o1,2) ;
    cca.RPL.beta_o1 = cca.beta_o1 ./ (cca.delta_o1 + cca.theta_o1 + cca.alpha_o1 + cca.beta_o1 + cca.gamma_o1) ;
    avg.RPL.beta_o1 = avg.beta_o1 ./ sum(avg.Regional_o1,2) ;
    cca.RPL.gamma_o1 = cca.gamma_o1 ./ (cca.delta_o1 + cca.theta_o1 + cca.alpha_o1 + cca.beta_o1 + cca.gamma_o1) ;
    avg.RPL.gamma_o1 = avg.gamma_o1 ./ sum(avg.Regional_o1,2) ;     

    cca.RPL.Regional_o1 = [cca.RPL.delta_o1 cca.RPL.theta_o1 cca.RPL.alpha_o1 cca.RPL.beta_o1 cca.RPL.gamma_o1]; 
    avg.RPL.Regional_o1 = [avg.RPL.delta_o1 avg.RPL.theta_o1 avg.RPL.alpha_o1 avg.RPL.beta_o1 avg.RPL.gamma_o1];
    
    cca.z.Regional_o1 = [zscore(cca.delta_o1) zscore(cca.theta_o1) zscore(cca.alpha_o1) zscore(cca.beta_o1) zscore(cca.gamma_o1)];
    avg.z.Regional_o1 = [zscore(avg.delta_o1) zscore(avg.theta_o1) zscore(avg.alpha_o1) zscore(avg.beta_o1) zscore(avg.gamma_o1)];                   

    
    
    
    [cca.delta_o2,avg.delta_o2] = Power_spectrum(channel_o2, [0.5 4], cnt.fs);
    [cca.theta_o2,avg.theta_o2] = Power_spectrum(channel_o2, [4 8], cnt.fs);
    [cca.alpha_o2,avg.alpha_o2] = Power_spectrum(channel_o2, [8 13], cnt.fs);
    [cca.beta_o2,avg.beta_o2] = Power_spectrum(channel_o2, [13 30], cnt.fs);
    [cca.gamma_o2,avg.gamma_o2] = Power_spectrum(channel_o2, [30 40], cnt.fs);
    avg.delta_o2 = avg.delta_o2'; avg.theta_o2=avg.theta_o2'; avg.alpha_o2=avg.alpha_o2'; avg.beta_o2 = avg.beta_o2'; avg.gamma_o2 = avg.gamma_o2';
    
    cca.Regional_o2 = [cca.delta_o2 cca.theta_o2 cca.alpha_o2 cca.beta_o2 cca.gamma_o2];
    avg.Regional_o2 = [avg.delta_o2 avg.theta_o2 avg.alpha_o2 avg.beta_o2 avg.gamma_o2];
    
    cca.RPL.delta_o2 = cca.delta_o2 ./ (cca.delta_o2 + cca.theta_o2 + cca.alpha_o2 + cca.beta_o2 + cca.gamma_o2) ;
    avg.RPL.delta_o2 = avg.delta_o2 ./ sum(avg.Regional_o2,2) ;
    cca.RPL.theta_o2 = cca.theta_o2 ./ (cca.delta_o2 + cca.theta_o2 + cca.alpha_o2 + cca.beta_o2 + cca.gamma_o2) ;
    avg.RPL.theta_o2 = avg.theta_o2 ./ sum(avg.Regional_o2,2) ;
    cca.RPL.alpha_o2 = cca.alpha_o2 ./ (cca.delta_o2 + cca.theta_o2 + cca.alpha_o2 + cca.beta_o2 + cca.gamma_o2) ;
    avg.RPL.alpha_o2 = avg.alpha_o2 ./ sum(avg.Regional_o2,2) ;
    cca.RPL.beta_o2 = cca.beta_o2 ./ (cca.delta_o2 + cca.theta_o2 + cca.alpha_o2 + cca.beta_o2 + cca.gamma_o2) ;
    avg.RPL.beta_o2 = avg.beta_o2 ./ sum(avg.Regional_o2,2) ;
    cca.RPL.gamma_o2 = cca.gamma_o2 ./ (cca.delta_o2 + cca.theta_o2 + cca.alpha_o2 + cca.beta_o2 + cca.gamma_o2) ;
    avg.RPL.gamma_o2 = avg.gamma_o2 ./ sum(avg.Regional_o2,2) ;     

    cca.RPL.Regional_o2 = [cca.RPL.delta_o2 cca.RPL.theta_o2 cca.RPL.alpha_o2 cca.RPL.beta_o2 cca.RPL.gamma_o2]; 
    avg.RPL.Regional_o2 = [avg.RPL.delta_o2 avg.RPL.theta_o2 avg.RPL.alpha_o2 avg.RPL.beta_o2 avg.RPL.gamma_o2];
        
    cca.z.Regional_o2 = [zscore(cca.delta_o2) zscore(cca.theta_o2) zscore(cca.alpha_o2) zscore(cca.beta_o2) zscore(cca.gamma_o2)];
    avg.z.Regional_o2 = [zscore(avg.delta_o2) zscore(avg.theta_o2) zscore(avg.alpha_o2) zscore(avg.beta_o2) zscore(avg.gamma_o2)];                   
    
    
    
    [cca.delta_o3,avg.delta_o3] = Power_spectrum(channel_o3, [0.5 4], cnt.fs);
    [cca.theta_o3,avg.theta_o3] = Power_spectrum(channel_o3, [4 8], cnt.fs);
    [cca.alpha_o3,avg.alpha_o3] = Power_spectrum(channel_o3, [8 13], cnt.fs);
    [cca.beta_o3,avg.beta_o3] = Power_spectrum(channel_o3, [13 30], cnt.fs);
    [cca.gamma_o3,avg.gamma_o3] = Power_spectrum(channel_o3, [30 40], cnt.fs);
    avg.delta_o3 = avg.delta_o3'; avg.theta_o3=avg.theta_o3'; avg.alpha_o3=avg.alpha_o3'; avg.beta_o3 = avg.beta_o3'; avg.gamma_o3 = avg.gamma_o3';
    
    cca.Regional_o3 = [cca.delta_o3 cca.theta_o3 cca.alpha_o3 cca.beta_o3 cca.gamma_o3];
    avg.Regional_o3 = [avg.delta_o3 avg.theta_o3 avg.alpha_o3 avg.beta_o3 avg.gamma_o3];

    cca.RPL.delta_o3 = cca.delta_o3 ./ (cca.delta_o3 + cca.theta_o3 + cca.alpha_o3 + cca.beta_o3 + cca.gamma_o3) ;
    avg.RPL.delta_o3 = avg.delta_o3 ./ sum(avg.Regional_o3,2) ;
    cca.RPL.theta_o3 = cca.theta_o3 ./ (cca.delta_o3 + cca.theta_o3 + cca.alpha_o3 + cca.beta_o3 + cca.gamma_o3) ;
    avg.RPL.theta_o3 = avg.theta_o3 ./ sum(avg.Regional_o3,2) ;
    cca.RPL.alpha_o3 = cca.alpha_o3 ./ (cca.delta_o3 + cca.theta_o3 + cca.alpha_o3 + cca.beta_o3 + cca.gamma_o3) ;
    avg.RPL.alpha_o3 = avg.alpha_o3 ./ sum(avg.Regional_o3,2) ;
    cca.RPL.beta_o3 = cca.beta_o3 ./ (cca.delta_o3 + cca.theta_o3 + cca.alpha_o3 + cca.beta_o3 + cca.gamma_o3) ;
    avg.RPL.beta_o3 = avg.beta_o3 ./ sum(avg.Regional_o3,2) ;
    cca.RPL.gamma_o3 = cca.gamma_o3 ./ (cca.delta_o3 + cca.theta_o3 + cca.alpha_o3 + cca.beta_o3 + cca.gamma_o3) ;
    avg.RPL.gamma_o3 = avg.gamma_o3 ./ sum(avg.Regional_o3,2) ;     

    cca.RPL.Regional_o3 = [cca.RPL.delta_o3 cca.RPL.theta_o3 cca.RPL.alpha_o3 cca.RPL.beta_o3 cca.RPL.gamma_o3]; 
    avg.RPL.Regional_o3 = [avg.RPL.delta_o3 avg.RPL.theta_o3 avg.RPL.alpha_o3 avg.RPL.beta_o3 avg.RPL.gamma_o3];
        
    cca.z.Regional_o3 = [zscore(cca.delta_o3) zscore(cca.theta_o3) zscore(cca.alpha_o3) zscore(cca.beta_o3) zscore(cca.gamma_o3)];
    avg.z.Regional_o3 = [zscore(avg.delta_o3) zscore(avg.theta_o3) zscore(avg.alpha_o3) zscore(avg.beta_o3) zscore(avg.gamma_o3)];                   
    
    
    %% Statistical Analysis
%         lowIdx = find(kss==min(kss)):1:find(kss==min(kss))+200;
%         lowIdx = lowIdx';
%         highIdx = find(kss==max(kss))-200:1:find(kss==max(kss));
%         highIdx = highIdx';
    
    %         최소+1, 최대-1
    lowIdx = min(kss)<kss & min(kss)+1>kss;
    highIdx = max(kss)>kss & max(kss)-1<kss;
    
    % PSD T-TEST 분석
    [h, p, ci, stats] = ttest2(avg.delta(lowIdx), avg.delta(highIdx));
    p_delta(s) = p;
    [h, p, ci, stats] = ttest2(avg.theta(lowIdx), avg.theta(highIdx));
    p_theta(s) = p;
    [h, p, ci, stats] = ttest2(avg.alpha(lowIdx), avg.alpha(highIdx));
    p_alpha(s) = p;
    [h, p, ci, stats] = ttest2(avg.beta(lowIdx), avg.beta(highIdx));
    p_beta(s) = p;
    [h, p, ci, stats] = ttest2(avg.gamma(lowIdx), avg.gamma(highIdx));
    p_gamma(s) = p;
    
    p_analysis(s,1:5) = [p_delta(s), p_theta(s), p_alpha(s), p_beta(s), p_gamma(s)];
    p_analysis_min(s) = find(p_analysis(s,1:5)==min(p_analysis(s,1:5)));
    
    Result.ttest.total(s,:) = p_analysis(s,1:5);
    
    % RPL T-TEST 분석
    [h, p, ci, stats] = ttest2(avg.RPL.delta(lowIdx), avg.RPL.delta(highIdx));
    p_delta(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.theta(lowIdx), avg.RPL.theta(highIdx));
    p_theta(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.alpha(lowIdx), avg.RPL.alpha(highIdx));
    p_alpha(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.beta(lowIdx), avg.RPL.beta(highIdx));
    p_beta(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.gamma(lowIdx), avg.RPL.gamma(highIdx));
    p_gamma(s) = p;
    
    p_analysis_RPL(s,1:5) = [p_delta(s), p_theta(s), p_alpha(s), p_beta(s), p_gamma(s)];
    p_analysis_RPL_min(s) = find(p_analysis_RPL(s,1:5)==min(p_analysis_RPL(s,1:5)));
    
    Result.ttest.total_RPL(s,:) = p_analysis_RPL(s,1:5);    
    
    
    % z-score T-TEST 분석
    [h, p, ci, stats] = ttest2(avg.z.PSD(lowIdx,1), avg.z.PSD(highIdx,1));
    p_delta(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.PSD(lowIdx,2), avg.z.PSD(lowIdx,2));
    p_theta(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.PSD(lowIdx,3), avg.z.PSD(lowIdx,3));
    p_alpha(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.PSD(lowIdx,4), avg.z.PSD(lowIdx,4));
    p_beta(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.PSD(lowIdx,5), avg.z.PSD(lowIdx,5));
    p_gamma(s) = p;
    
    p_analysis_z(s,1:5) = [p_delta(s), p_theta(s), p_alpha(s), p_beta(s), p_gamma(s)];
    p_analysis_z_min(s) = find(p_analysis_z(s,1:5)==min(p_analysis_z(s,1:5)));
    
    Result.ttest.total_z(s,:) = p_analysis_z(s,1:5);    
    
    
    
    % Regional PSD T-TEST 분석
    [h, p, ci, stats] = ttest2(avg.delta_f(lowIdx), avg.delta_f(highIdx));
    p_delta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.theta_f(lowIdx), avg.theta_f(highIdx));
    p_theta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.alpha_f(lowIdx), avg.alpha_f(highIdx));
    p_alpha_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.beta_f(lowIdx), avg.beta_f(highIdx));
    p_beta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.gamma_f(lowIdx), avg.gamma_f(highIdx));
    p_gamma_f(s) = p;
    
    p_analysis_f(s,1:5) = [p_delta_f(s), p_theta_f(s), p_alpha_f(s), p_beta_f(s), p_gamma_f(s)];
    p_analysis_f_min(s) = find(p_analysis_f(s,1:5)==min(p_analysis_f(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.delta_c(lowIdx), avg.delta_c(highIdx));
    p_delta_c(s) = p;
    [h, p, ci, stats] = ttest2(avg.theta_c(lowIdx), avg.theta_c(highIdx));
    p_theta_c(s) = p;
    [h, p, ci, stats] = ttest2(avg.alpha_c(lowIdx), avg.alpha_c(highIdx));
    p_alpha_c(s) = p;
    [h, p, ci, stats] = ttest2(avg.beta_c(lowIdx), avg.beta_c(highIdx));
    p_beta_c(s) = p;
    [h, p, ci, stats] = ttest2(avg.gamma_c(lowIdx), avg.gamma_c(highIdx));
    p_gamma_c(s) = p;
    
    p_analysis_c(s,1:5) = [p_delta_c(s), p_theta_c(s), p_alpha_c(s), p_beta_c(s), p_gamma_c(s)];
    p_analysis_c_min(s) = find(p_analysis_c(s,1:5)==min(p_analysis_c(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.delta_p(lowIdx), avg.delta_p(highIdx));
    p_delta_p(s) = p;
    [h, p, ci, stats] = ttest2(avg.theta_p(lowIdx), avg.theta_p(highIdx));
    p_theta_p(s) = p;
    [h, p, ci, stats] = ttest2(avg.alpha_p(lowIdx), avg.alpha_p(highIdx));
    p_alpha_p(s) = p;
    [h, p, ci, stats] = ttest2(avg.beta_p(lowIdx), avg.beta_p(highIdx));
    p_beta_p(s) = p;
    [h, p, ci, stats] = ttest2(avg.gamma_p(lowIdx), avg.gamma_p(highIdx));
    p_gamma_p(s) = p;
    
    p_analysis_p(s,1:5) = [p_delta_p(s), p_theta_p(s), p_alpha_p(s), p_beta_p(s), p_gamma_p(s)];
    p_analysis_p_min(s) = find(p_analysis_p(s,1:5)==min(p_analysis_p(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.delta_t(lowIdx), avg.delta_t(highIdx));
    p_delta_t(s) = p;
    [h, p, ci, stats] = ttest2(avg.theta_t(lowIdx), avg.theta_t(highIdx));
    p_theta_t(s) = p;
    [h, p, ci, stats] = ttest2(avg.alpha_t(lowIdx), avg.alpha_t(highIdx));
    p_alpha_t(s) = p;
    [h, p, ci, stats] = ttest2(avg.beta_t(lowIdx), avg.beta_t(highIdx));
    p_beta_t(s) = p;
    [h, p, ci, stats] = ttest2(avg.gamma_t(lowIdx), avg.gamma_t(highIdx));
    p_gamma_t(s) = p;
    
    p_analysis_t(s,1:5) = [p_delta_t(s), p_theta_t(s), p_alpha_t(s), p_beta_t(s), p_gamma_t(s)];
    p_analysis_t_min(s) = find(p_analysis_t(s,1:5)==min(p_analysis_t(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.delta_o(lowIdx), avg.delta_o(highIdx));
    p_delta_o(s) = p;
    [h, p, ci, stats] = ttest2(avg.theta_o(lowIdx), avg.theta_o(highIdx));
    p_theta_o(s) = p;
    [h, p, ci, stats] = ttest2(avg.alpha_o(lowIdx), avg.alpha_o(highIdx));
    p_alpha_o(s) = p;
    [h, p, ci, stats] = ttest2(avg.beta_o(lowIdx), avg.beta_o(highIdx));
    p_beta_o(s) = p;
    [h, p, ci, stats] = ttest2(avg.gamma_o(lowIdx), avg.gamma_o(highIdx));
    p_gamma_o(s) = p;
    
    p_analysis_o(s,1:5) = [p_delta_o(s), p_theta_o(s), p_alpha_o(s), p_beta_o(s), p_gamma_o(s)];
    p_analysis_o_min(s) = find(p_analysis_o(s,1:5)==min(p_analysis_o(s,1:5)));
    
    Result.ttest.regional(s,:) = [p_analysis_f(s,:) p_analysis_c(s,:) p_analysis_p(s,:) p_analysis_t(s,:) p_analysis_o(s,:)];

    
    
    [h, p, ci, stats] = ttest2(avg.RPL.delta_f(lowIdx), avg.RPL.delta_f(highIdx));
    p_delta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.theta_f(lowIdx), avg.RPL.theta_f(highIdx));
    p_theta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.alpha_f(lowIdx), avg.RPL.alpha_f(highIdx));
    p_alpha_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.beta_f(lowIdx), avg.RPL.beta_f(highIdx));
    p_beta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.gamma_f(lowIdx), avg.RPL.gamma_f(highIdx));
    p_gamma_f(s) = p;
    
    p_analysis_RPL_f(s,1:5) = [p_delta_f(s), p_theta_f(s), p_alpha_f(s), p_beta_f(s), p_gamma_f(s)];
    p_analysis_RPL_f_min(s) = find(p_analysis_RPL_f(s,1:5)==min(p_analysis_RPL_f(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.RPL.delta_c(lowIdx), avg.RPL.delta_c(highIdx));
    p_delta_c(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.theta_c(lowIdx), avg.RPL.theta_c(highIdx));
    p_theta_c(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.alpha_c(lowIdx), avg.RPL.alpha_c(highIdx));
    p_alpha_c(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.beta_c(lowIdx), avg.RPL.beta_c(highIdx));
    p_beta_c(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.gamma_c(lowIdx), avg.RPL.gamma_c(highIdx));
    p_gamma_c(s) = p;
    
    p_analysis_RPL_c(s,1:5) = [p_delta_c(s), p_theta_c(s), p_alpha_c(s), p_beta_c(s), p_gamma_c(s)];
    p_analysis_RPL_c_min(s) = find(p_analysis_RPL_c(s,1:5)==min(p_analysis_RPL_c(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.RPL.delta_p(lowIdx), avg.RPL.delta_p(highIdx));
    p_delta_p(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.theta_p(lowIdx), avg.RPL.theta_p(highIdx));
    p_theta_p(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.alpha_p(lowIdx), avg.RPL.alpha_p(highIdx));
    p_alpha_p(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.beta_p(lowIdx), avg.RPL.beta_p(highIdx));
    p_beta_p(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.gamma_p(lowIdx), avg.RPL.gamma_p(highIdx));
    p_gamma_p(s) = p;
    
    p_analysis_RPL_p(s,1:5) = [p_delta_p(s), p_theta_p(s), p_alpha_p(s), p_beta_p(s), p_gamma_p(s)];
    p_analysis_RPL_p_min(s) = find(p_analysis_RPL_p(s,1:5)==min(p_analysis_RPL_p(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.RPL.delta_t(lowIdx), avg.RPL.delta_t(highIdx));
    p_delta_t(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.theta_t(lowIdx), avg.RPL.theta_t(highIdx));
    p_theta_t(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.alpha_t(lowIdx), avg.RPL.alpha_t(highIdx));
    p_alpha_t(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.beta_t(lowIdx), avg.RPL.beta_t(highIdx));
    p_beta_t(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.gamma_t(lowIdx), avg.RPL.gamma_t(highIdx));
    p_gamma_t(s) = p;
    
    p_analysis_RPL_t(s,1:5) = [p_delta_t(s), p_theta_t(s), p_alpha_t(s), p_beta_t(s), p_gamma_t(s)];
    p_analysis_RPL_t_min(s) = find(p_analysis_RPL_t(s,1:5)==min(p_analysis_RPL_t(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.RPL.delta_o(lowIdx), avg.RPL.delta_o(highIdx));
    p_delta_o(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.theta_o(lowIdx), avg.RPL.theta_o(highIdx));
    p_theta_o(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.alpha_o(lowIdx), avg.RPL.alpha_o(highIdx));
    p_alpha_o(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.beta_o(lowIdx), avg.RPL.beta_o(highIdx));
    p_beta_o(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.gamma_o(lowIdx), avg.RPL.gamma_o(highIdx));
    p_gamma_o(s) = p;
    
    p_analysis_RPL_o(s,1:5) = [p_delta_o(s), p_theta_o(s), p_alpha_o(s), p_beta_o(s), p_gamma_o(s)];
    p_analysis_RPL_o_min(s) = find(p_analysis_RPL_o(s,1:5)==min(p_analysis_RPL_o(s,1:5)));
    
    Result.ttest.regional_RPL(s,:) = [p_analysis_RPL_f(s,:) p_analysis_RPL_c(s,:) p_analysis_RPL_p(s,:) p_analysis_RPL_t(s,:) p_analysis_RPL_o(s,:)];

    
    
    [h, p, ci, stats] = ttest2(avg.z.Regional_f(lowIdx,1), avg.z.Regional_f(highIdx,1));
    p_delta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_f(lowIdx,2), avg.z.Regional_f(highIdx,2));
    p_theta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_f(lowIdx,3), avg.z.Regional_f(highIdx,3));
    p_alpha_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_f(lowIdx,4), avg.z.Regional_f(highIdx,4));
    p_beta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_f(lowIdx,5), avg.z.Regional_f(highIdx,5));
    p_gamma_f(s) = p;
    
    p_analysis_z_f(s,1:5) = [p_delta_f(s), p_theta_f(s), p_alpha_f(s), p_beta_f(s), p_gamma_f(s)];
    p_analysis_z_f_min(s) = find(p_analysis_z_f(s,1:5)==min(p_analysis_z_f(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.z.Regional_c(lowIdx,1), avg.z.Regional_c(highIdx,1));
    p_delta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_c(lowIdx,2), avg.z.Regional_c(highIdx,2));
    p_theta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_c(lowIdx,3), avg.z.Regional_c(highIdx,3));
    p_alpha_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_c(lowIdx,4), avg.z.Regional_c(highIdx,4));
    p_beta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_c(lowIdx,5), avg.z.Regional_c(highIdx,5));
    p_gamma_f(s) = p;
    
    p_analysis_z_c(s,1:5) = [p_delta_c(s), p_theta_c(s), p_alpha_c(s), p_beta_c(s), p_gamma_c(s)];
    p_analysis_z_c_min(s) = find(p_analysis_z_c(s,1:5)==min(p_analysis_z_c(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.z.Regional_p(lowIdx,1), avg.z.Regional_p(highIdx,1));
    p_delta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_p(lowIdx,2), avg.z.Regional_p(highIdx,2));
    p_theta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_p(lowIdx,3), avg.z.Regional_p(highIdx,3));
    p_alpha_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_p(lowIdx,4), avg.z.Regional_p(highIdx,4));
    p_beta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_p(lowIdx,5), avg.z.Regional_p(highIdx,5));
    p_gamma_f(s) = p;
    
    p_analysis_z_p(s,1:5) = [p_delta_p(s), p_theta_p(s), p_alpha_p(s), p_beta_p(s), p_gamma_p(s)];
    p_analysis_z_p_min(s) = find(p_analysis_z_p(s,1:5)==min(p_analysis_z_p(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.z.Regional_t(lowIdx,1), avg.z.Regional_t(highIdx,1));
    p_delta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_t(lowIdx,2), avg.z.Regional_t(highIdx,2));
    p_theta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_t(lowIdx,3), avg.z.Regional_t(highIdx,3));
    p_alpha_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_t(lowIdx,4), avg.z.Regional_t(highIdx,4));
    p_beta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_t(lowIdx,5), avg.z.Regional_t(highIdx,5));
    p_gamma_f(s) = p;
    
    p_analysis_z_t(s,1:5) = [p_delta_t(s), p_theta_t(s), p_alpha_t(s), p_beta_t(s), p_gamma_t(s)];
    p_analysis_z_t_min(s) = find(p_analysis_z_t(s,1:5)==min(p_analysis_z_t(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.z.Regional_o(lowIdx,1), avg.z.Regional_o(highIdx,1));
    p_delta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_o(lowIdx,2), avg.z.Regional_o(highIdx,2));
    p_theta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_o(lowIdx,3), avg.z.Regional_o(highIdx,3));
    p_alpha_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_o(lowIdx,4), avg.z.Regional_o(highIdx,4));
    p_beta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_o(lowIdx,5), avg.z.Regional_o(highIdx,5));
    p_gamma_f(s) = p;
    
    p_analysis_z_o(s,1:5) = [p_delta_o(s), p_theta_o(s), p_alpha_o(s), p_beta_o(s), p_gamma_o(s)];
    p_analysis_z_o_min(s) = find(p_analysis_z_o(s,1:5)==min(p_analysis_z_o(s,1:5)));
    
    Result.ttest.regional_z(s,:) = [p_analysis_f(s,:) p_analysis_c(s,:) p_analysis_p(s,:) p_analysis_t(s,:) p_analysis_o(s,:)];

    
    
    
    
    
    % Regional Regional T-TEST 분석
    [h, p, ci, stats] = ttest2(avg.delta_f1(lowIdx), avg.delta_f1(highIdx));
    p_delta_f1(s) = p;
    [h, p, ci, stats] = ttest2(avg.theta_f1(lowIdx), avg.theta_f1(highIdx));
    p_theta_f1(s) = p;
    [h, p, ci, stats] = ttest2(avg.alpha_f1(lowIdx), avg.alpha_f1(highIdx));
    p_alpha_f1(s) = p;
    [h, p, ci, stats] = ttest2(avg.beta_f1(lowIdx), avg.beta_f1(highIdx));
    p_beta_f1(s) = p;
    [h, p, ci, stats] = ttest2(avg.gamma_f1(lowIdx), avg.gamma_f1(highIdx));
    p_gamma_f1(s) = p;
    
    p_analysis_f1(s,1:5) = [p_delta_f1(s), p_theta_f1(s), p_alpha_f1(s), p_beta_f1(s), p_gamma_f1(s)];
    p_analysis_f1_min(s) = find(p_analysis_f1(s,1:5)==min(p_analysis_f1(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.delta_f2(lowIdx), avg.delta_f2(highIdx));
    p_delta_f2(s) = p;
    [h, p, ci, stats] = ttest2(avg.theta_f2(lowIdx), avg.theta_f2(highIdx));
    p_theta_f2(s) = p;
    [h, p, ci, stats] = ttest2(avg.alpha_f2(lowIdx), avg.alpha_f2(highIdx));
    p_alpha_f2(s) = p;
    [h, p, ci, stats] = ttest2(avg.beta_f2(lowIdx), avg.beta_f2(highIdx));
    p_beta_f2(s) = p;
    [h, p, ci, stats] = ttest2(avg.gamma_f2(lowIdx), avg.gamma_f2(highIdx));
    p_gamma_f2(s) = p;
    
    p_analysis_f2(s,1:5) = [p_delta_f2(s), p_theta_f2(s), p_alpha_f2(s), p_beta_f2(s), p_gamma_f2(s)];
    p_analysis_f2_min(s) = find(p_analysis_f2(s,1:5)==min(p_analysis_f2(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.delta_f3(lowIdx), avg.delta_f3(highIdx));
    p_delta_f3(s) = p;
    [h, p, ci, stats] = ttest2(avg.theta_f3(lowIdx), avg.theta_f3(highIdx));
    p_theta_f3(s) = p;
    [h, p, ci, stats] = ttest2(avg.alpha_f3(lowIdx), avg.alpha_f3(highIdx));
    p_alpha_f3(s) = p;
    [h, p, ci, stats] = ttest2(avg.beta_f3(lowIdx), avg.beta_f3(highIdx));
    p_beta_f3(s) = p;
    [h, p, ci, stats] = ttest2(avg.gamma_f3(lowIdx), avg.gamma_f3(highIdx));
    p_gamma_f3(s) = p;
    
    p_analysis_f3(s,1:5) = [p_delta_f3(s), p_theta_f3(s), p_alpha_f3(s), p_beta_f3(s), p_gamma_f3(s)];
    p_analysis_f3_min(s) = find(p_analysis_f3(s,1:5)==min(p_analysis_f3(s,1:5)));
    
    
    
    [h, p, ci, stats] = ttest2(avg.delta_c1(lowIdx), avg.delta_c1(highIdx));
    p_delta_c1(s) = p;
    [h, p, ci, stats] = ttest2(avg.theta_c1(lowIdx), avg.theta_c1(highIdx));
    p_theta_c1(s) = p;
    [h, p, ci, stats] = ttest2(avg.alpha_c1(lowIdx), avg.alpha_c1(highIdx));
    p_alpha_c1(s) = p;
    [h, p, ci, stats] = ttest2(avg.beta_c1(lowIdx), avg.beta_c1(highIdx));
    p_beta_c1(s) = p;
    [h, p, ci, stats] = ttest2(avg.gamma_c1(lowIdx), avg.gamma_c1(highIdx));
    p_gamma_c1(s) = p;
    
    p_analysis_c1(s,1:5) = [p_delta_c1(s), p_theta_c1(s), p_alpha_c1(s), p_beta_c1(s), p_gamma_c1(s)];
    p_analysis_c1_min(s) = find(p_analysis_c1(s,1:5)==min(p_analysis_c1(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.delta_c2(lowIdx), avg.delta_c2(highIdx));
    p_delta_c2(s) = p;
    [h, p, ci, stats] = ttest2(avg.theta_c2(lowIdx), avg.theta_c2(highIdx));
    p_theta_c2(s) = p;
    [h, p, ci, stats] = ttest2(avg.alpha_c2(lowIdx), avg.alpha_c2(highIdx));
    p_alpha_c2(s) = p;
    [h, p, ci, stats] = ttest2(avg.beta_c2(lowIdx), avg.beta_c2(highIdx));
    p_beta_c2(s) = p;
    [h, p, ci, stats] = ttest2(avg.gamma_c2(lowIdx), avg.gamma_c2(highIdx));
    p_gamma_c2(s) = p;
    
    p_analysis_c2(s,1:5) = [p_delta_c2(s), p_theta_c2(s), p_alpha_c2(s), p_beta_c2(s), p_gamma_c2(s)];
    p_analysis_c2_min(s) = find(p_analysis_c2(s,1:5)==min(p_analysis_c2(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.delta_c3(lowIdx), avg.delta_c3(highIdx));
    p_delta_c3(s) = p;
    [h, p, ci, stats] = ttest2(avg.theta_c3(lowIdx), avg.theta_c3(highIdx));
    p_theta_c3(s) = p;
    [h, p, ci, stats] = ttest2(avg.alpha_c3(lowIdx), avg.alpha_c3(highIdx));
    p_alpha_c3(s) = p;
    [h, p, ci, stats] = ttest2(avg.beta_c3(lowIdx), avg.beta_c3(highIdx));
    p_beta_c3(s) = p;
    [h, p, ci, stats] = ttest2(avg.gamma_c3(lowIdx), avg.gamma_c3(highIdx));
    p_gamma_c3(s) = p;
    
    p_analysis_c3(s,1:5) = [p_delta_c3(s), p_theta_c3(s), p_alpha_c3(s), p_beta_c3(s), p_gamma_c3(s)];
    p_analysis_c3_min(s) = find(p_analysis_c3(s,1:5)==min(p_analysis_c3(s,1:5)));
    
    

    
    
    [h, p, ci, stats] = ttest2(avg.delta_p1(lowIdx), avg.delta_p1(highIdx));
    p_delta_p1(s) = p;
    [h, p, ci, stats] = ttest2(avg.theta_p1(lowIdx), avg.theta_p1(highIdx));
    p_theta_p1(s) = p;
    [h, p, ci, stats] = ttest2(avg.alpha_p1(lowIdx), avg.alpha_p1(highIdx));
    p_alpha_p1(s) = p;
    [h, p, ci, stats] = ttest2(avg.beta_p1(lowIdx), avg.beta_p1(highIdx));
    p_beta_p1(s) = p;
    [h, p, ci, stats] = ttest2(avg.gamma_p1(lowIdx), avg.gamma_p1(highIdx));
    p_gamma_p1(s) = p;
    
    p_analysis_p1(s,1:5) = [p_delta_p1(s), p_theta_p1(s), p_alpha_p1(s), p_beta_p1(s), p_gamma_p1(s)];
    p_analysis_p1_min(s) = find(p_analysis_p1(s,1:5)==min(p_analysis_p1(s,1:5)));
    
    
    
    [h, p, ci, stats] = ttest2(avg.delta_p2(lowIdx), avg.delta_p2(highIdx));
    p_delta_p2(s) = p;
    [h, p, ci, stats] = ttest2(avg.theta_p2(lowIdx), avg.theta_p2(highIdx));
    p_theta_p2(s) = p;
    [h, p, ci, stats] = ttest2(avg.alpha_p2(lowIdx), avg.alpha_p2(highIdx));
    p_alpha_p2(s) = p;
    [h, p, ci, stats] = ttest2(avg.beta_p2(lowIdx), avg.beta_p2(highIdx));
    p_beta_p2(s) = p;
    [h, p, ci, stats] = ttest2(avg.gamma_p2(lowIdx), avg.gamma_p2(highIdx));
    p_gamma_p2(s) = p;
    
    p_analysis_p2(s,1:5) = [p_delta_p2(s), p_theta_p2(s), p_alpha_p2(s), p_beta_p2(s), p_gamma_p2(s)];
    p_analysis_p2_min(s) = find(p_analysis_p2(s,1:5)==min(p_analysis_p2(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.delta_p3(lowIdx), avg.delta_p3(highIdx));
    p_delta_p3(s) = p;
    [h, p, ci, stats] = ttest2(avg.theta_p3(lowIdx), avg.theta_p3(highIdx));
    p_theta_p3(s) = p;
    [h, p, ci, stats] = ttest2(avg.alpha_p3(lowIdx), avg.alpha_p3(highIdx));
    p_alpha_p3(s) = p;
    [h, p, ci, stats] = ttest2(avg.beta_p3(lowIdx), avg.beta_p3(highIdx));
    p_beta_p3(s) = p;
    [h, p, ci, stats] = ttest2(avg.gamma_p3(lowIdx), avg.gamma_p3(highIdx));
    p_gamma_p3(s) = p;
    
    p_analysis_p3(s,1:5) = [p_delta_p3(s), p_theta_p3(s), p_alpha_p3(s), p_beta_p3(s), p_gamma_p3(s)];
    p_analysis_p3_min(s) = find(p_analysis_p3(s,1:5)==min(p_analysis_p3(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.delta_t1(lowIdx), avg.delta_t1(highIdx));
    p_delta_t1(s) = p;
    [h, p, ci, stats] = ttest2(avg.theta_t1(lowIdx), avg.theta_t1(highIdx));
    p_theta_t1(s) = p;
    [h, p, ci, stats] = ttest2(avg.alpha_t1(lowIdx), avg.alpha_t1(highIdx));
    p_alpha_t1(s) = p;
    [h, p, ci, stats] = ttest2(avg.beta_t1(lowIdx), avg.beta_t1(highIdx));
    p_beta_t1(s) = p;
    [h, p, ci, stats] = ttest2(avg.gamma_t1(lowIdx), avg.gamma_t1(highIdx));
    p_gamma_t1(s) = p;
    
    p_analysis_t1(s,1:5) = [p_delta_t1(s), p_theta_t1(s), p_alpha_t1(s), p_beta_t1(s), p_gamma_t1(s)];
    p_analysis_t1_min(s) = find(p_analysis_t1(s,1:5)==min(p_analysis_t1(s,1:5)));
    
    [h, p, ci, stats] = ttest2(avg.delta_t2(lowIdx), avg.delta_t2(highIdx));
    p_delta_t2(s) = p;
    [h, p, ci, stats] = ttest2(avg.theta_t2(lowIdx), avg.theta_t2(highIdx));
    p_theta_t2(s) = p;
    [h, p, ci, stats] = ttest2(avg.alpha_t2(lowIdx), avg.alpha_t2(highIdx));
    p_alpha_t2(s) = p;
    [h, p, ci, stats] = ttest2(avg.beta_t2(lowIdx), avg.beta_t2(highIdx));
    p_beta_t2(s) = p;
    [h, p, ci, stats] = ttest2(avg.gamma_t2(lowIdx), avg.gamma_t2(highIdx));
    p_gamma_t2(s) = p;
    
    p_analysis_t2(s,1:5) = [p_delta_t2(s), p_theta_t2(s), p_alpha_t2(s), p_beta_t2(s), p_gamma_t2(s)];
    p_analysis_t2_min(s) = find(p_analysis_t2(s,1:5)==min(p_analysis_t2(s,1:5)));
    
    
    
    [h, p, ci, stats] = ttest2(avg.delta_o1(lowIdx), avg.delta_o1(highIdx));
    p_delta_o1(s) = p;
    [h, p, ci, stats] = ttest2(avg.theta_o1(lowIdx), avg.theta_o1(highIdx));
    p_theta_o1(s) = p;
    [h, p, ci, stats] = ttest2(avg.alpha_o1(lowIdx), avg.alpha_o1(highIdx));
    p_alpha_o1(s) = p;
    [h, p, ci, stats] = ttest2(avg.beta_o1(lowIdx), avg.beta_o1(highIdx));
    p_beta_o1(s) = p;
    [h, p, ci, stats] = ttest2(avg.gamma_o1(lowIdx), avg.gamma_o1(highIdx));
    p_gamma_o1(s) = p;
    
    p_analysis_o1(s,1:5) = [p_delta_o1(s), p_theta_o1(s), p_alpha_o1(s), p_beta_o1(s), p_gamma_o1(s)];
    p_analysis_o1_min(s) = find(p_analysis_o1(s,1:5)==min(p_analysis_o1(s,1:5)));
    
    [h, p, ci, stats] = ttest2(avg.delta_o2(lowIdx), avg.delta_o2(highIdx));
    p_delta_o2(s) = p;
    [h, p, ci, stats] = ttest2(avg.theta_o2(lowIdx), avg.theta_o2(highIdx));
    p_theta_o2(s) = p;
    [h, p, ci, stats] = ttest2(avg.alpha_o2(lowIdx), avg.alpha_o2(highIdx));
    p_alpha_o2(s) = p;
    [h, p, ci, stats] = ttest2(avg.beta_o2(lowIdx), avg.beta_o2(highIdx));
    p_beta_o2(s) = p;
    [h, p, ci, stats] = ttest2(avg.gamma_o2(lowIdx), avg.gamma_o2(highIdx));
    p_gamma_o2(s) = p;
    
    p_analysis_o2(s,1:5) = [p_delta_o2(s), p_theta_o2(s), p_alpha_o2(s), p_beta_o2(s), p_gamma_o2(s)];
    p_analysis_o2_min(s) = find(p_analysis_o2(s,1:5)==min(p_analysis_o2(s,1:5)));
    
    [h, p, ci, stats] = ttest2(avg.delta_o3(lowIdx), avg.delta_o3(highIdx));
    p_delta_o3(s) = p;
    [h, p, ci, stats] = ttest2(avg.theta_o3(lowIdx), avg.theta_o3(highIdx));
    p_theta_o3(s) = p;
    [h, p, ci, stats] = ttest2(avg.alpha_o3(lowIdx), avg.alpha_o3(highIdx));
    p_alpha_o3(s) = p;
    [h, p, ci, stats] = ttest2(avg.beta_o3(lowIdx), avg.beta_o3(highIdx));
    p_beta_o3(s) = p;
    [h, p, ci, stats] = ttest2(avg.gamma_o3(lowIdx), avg.gamma_o3(highIdx));
    p_gamma_o3(s) = p;
    
    p_analysis_o3(s,1:5) = [p_delta_o3(s), p_theta_o3(s), p_alpha_o3(s), p_beta_o3(s), p_gamma_o3(s)];
    p_analysis_o3_min(s) = find(p_analysis_o3(s,1:5)==min(p_analysis_o3(s,1:5)));
    
    Result.ttest.vertical(s,:) = [p_analysis_f1(s,:) p_analysis_f2(s,:) p_analysis_f3(s,:) p_analysis_c1(s,:) p_analysis_c2(s,:) p_analysis_c3(s,:) p_analysis_p1(s,:) p_analysis_p2(s,:) p_analysis_p3(s,:) p_analysis_t1(s,:) p_analysis_t2(s,:) p_analysis_o1(s,:) p_analysis_o2(s,:) p_analysis_o3(s,:)];
    
    
    
        % Regional Vertical T-TEST 분석
    [h, p, ci, stats] = ttest2(avg.RPL.delta_f1(lowIdx), avg.RPL.delta_f1(highIdx));
    p_delta_f1(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.theta_f1(lowIdx), avg.RPL.theta_f1(highIdx));
    p_theta_f1(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.alpha_f1(lowIdx), avg.RPL.alpha_f1(highIdx));
    p_alpha_f1(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.beta_f1(lowIdx), avg.RPL.beta_f1(highIdx));
    p_beta_f1(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.gamma_f1(lowIdx), avg.RPL.gamma_f1(highIdx));
    p_gamma_f1(s) = p;
    
    p_analysis_f1_rpl(s,1:5) = [p_delta_f1(s), p_theta_f1(s), p_alpha_f1(s), p_beta_f1(s), p_gamma_f1(s)];
    p_analysis_f1_rpl_min(s) = find(p_analysis_f1_rpl(s,1:5)==min(p_analysis_f1_rpl(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.RPL.delta_f2(lowIdx), avg.RPL.delta_f2(highIdx));
    p_delta_f2(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.theta_f2(lowIdx), avg.RPL.theta_f2(highIdx));
    p_theta_f2(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.alpha_f2(lowIdx), avg.RPL.alpha_f2(highIdx));
    p_alpha_f2(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.beta_f2(lowIdx), avg.RPL.beta_f2(highIdx));
    p_beta_f2(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.gamma_f2(lowIdx), avg.RPL.gamma_f2(highIdx));
    p_gamma_f2(s) = p;
    
    p_analysis_f2_rpl(s,1:5) = [p_delta_f2(s), p_theta_f2(s), p_alpha_f2(s), p_beta_f2(s), p_gamma_f2(s)];
    p_analysis_f2_rpl_min(s) = find(p_analysis_f2_rpl(s,1:5)==min(p_analysis_f2_rpl(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.RPL.delta_f3(lowIdx), avg.RPL.delta_f3(highIdx));
    p_delta_f3(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.theta_f3(lowIdx), avg.RPL.theta_f3(highIdx));
    p_theta_f3(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.alpha_f3(lowIdx), avg.RPL.alpha_f3(highIdx));
    p_alpha_f3(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.beta_f3(lowIdx), avg.RPL.beta_f3(highIdx));
    p_beta_f3(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.gamma_f3(lowIdx), avg.RPL.gamma_f3(highIdx));
    p_gamma_f3(s) = p;
    
    p_analysis_f3_rpl(s,1:5) = [p_delta_f3(s), p_theta_f3(s), p_alpha_f3(s), p_beta_f3(s), p_gamma_f3(s)];
    p_analysis_f3_rpl_min(s) = find(p_analysis_f3_rpl(s,1:5)==min(p_analysis_f3_rpl(s,1:5)));
    
    
    
    [h, p, ci, stats] = ttest2(avg.RPL.delta_c1(lowIdx), avg.RPL.delta_c1(highIdx));
    p_delta_c1(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.theta_c1(lowIdx), avg.RPL.theta_c1(highIdx));
    p_theta_c1(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.alpha_c1(lowIdx), avg.RPL.alpha_c1(highIdx));
    p_alpha_c1(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.beta_c1(lowIdx), avg.RPL.beta_c1(highIdx));
    p_beta_c1(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.gamma_c1(lowIdx), avg.RPL.gamma_c1(highIdx));
    p_gamma_c1(s) = p;
    
    p_analysis_c1_rpl(s,1:5) = [p_delta_c1(s), p_theta_c1(s), p_alpha_c1(s), p_beta_c1(s), p_gamma_c1(s)];
    p_analysis_c1_rpl_min(s) = find(p_analysis_c1_rpl(s,1:5)==min(p_analysis_c1_rpl(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.RPL.delta_c2(lowIdx), avg.RPL.delta_c2(highIdx));
    p_delta_c2(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.theta_c2(lowIdx), avg.RPL.theta_c2(highIdx));
    p_theta_c2(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.alpha_c2(lowIdx), avg.RPL.alpha_c2(highIdx));
    p_alpha_c2(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.beta_c2(lowIdx), avg.RPL.beta_c2(highIdx));
    p_beta_c2(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.gamma_c2(lowIdx), avg.RPL.gamma_c2(highIdx));
    p_gamma_c2(s) = p;
    
    p_analysis_c2_rpl(s,1:5) = [p_delta_c2(s), p_theta_c2(s), p_alpha_c2(s), p_beta_c2(s), p_gamma_c2(s)];
    p_analysis_c2_rpl_min(s) = find(p_analysis_c2_rpl(s,1:5)==min(p_analysis_c2_rpl(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.RPL.delta_c3(lowIdx), avg.RPL.delta_c3(highIdx));
    p_delta_c3(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.theta_c3(lowIdx), avg.RPL.theta_c3(highIdx));
    p_theta_c3(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.alpha_c3(lowIdx), avg.RPL.alpha_c3(highIdx));
    p_alpha_c3(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.beta_c3(lowIdx), avg.RPL.beta_c3(highIdx));
    p_beta_c3(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.gamma_c3(lowIdx), avg.RPL.gamma_c3(highIdx));
    p_gamma_c3(s) = p;
    
    p_analysis_c3_rpl(s,1:5) = [p_delta_c3(s), p_theta_c3(s), p_alpha_c3(s), p_beta_c3(s), p_gamma_c3(s)];
    p_analysis_c3_rpl_min(s) = find(p_analysis_c3_rpl(s,1:5)==min(p_analysis_c3_rpl(s,1:5)));
    
    

    
    
    [h, p, ci, stats] = ttest2(avg.RPL.delta_p1(lowIdx), avg.RPL.delta_p1(highIdx));
    p_delta_p1(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.theta_p1(lowIdx), avg.RPL.theta_p1(highIdx));
    p_theta_p1(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.alpha_p1(lowIdx), avg.RPL.alpha_p1(highIdx));
    p_alpha_p1(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.beta_p1(lowIdx), avg.RPL.beta_p1(highIdx));
    p_beta_p1(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.gamma_p1(lowIdx), avg.RPL.gamma_p1(highIdx));
    p_gamma_p1(s) = p;
    
    p_analysis_p1_rpl(s,1:5) = [p_delta_p1(s), p_theta_p1(s), p_alpha_p1(s), p_beta_p1(s), p_gamma_p1(s)];
    p_analysis_p1_rpl_min(s) = find(p_analysis_p1_rpl(s,1:5)==min(p_analysis_p1_rpl(s,1:5)));
    
    
    
    [h, p, ci, stats] = ttest2(avg.RPL.delta_p2(lowIdx), avg.RPL.delta_p2(highIdx));
    p_delta_p2(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.theta_p2(lowIdx), avg.RPL.theta_p2(highIdx));
    p_theta_p2(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.alpha_p2(lowIdx), avg.RPL.alpha_p2(highIdx));
    p_alpha_p2(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.beta_p2(lowIdx), avg.RPL.beta_p2(highIdx));
    p_beta_p2(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.gamma_p2(lowIdx), avg.RPL.gamma_p2(highIdx));
    p_gamma_p2(s) = p;
    
    p_analysis_p2_rpl(s,1:5) = [p_delta_p2(s), p_theta_p2(s), p_alpha_p2(s), p_beta_p2(s), p_gamma_p2(s)];
    p_analysis_p2_rpl_min(s) = find(p_analysis_p2_rpl(s,1:5)==min(p_analysis_p2_rpl(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.RPL.delta_p3(lowIdx), avg.RPL.delta_p3(highIdx));
    p_delta_p3(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.theta_p3(lowIdx), avg.RPL.theta_p3(highIdx));
    p_theta_p3(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.alpha_p3(lowIdx), avg.RPL.alpha_p3(highIdx));
    p_alpha_p3(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.beta_p3(lowIdx), avg.RPL.beta_p3(highIdx));
    p_beta_p3(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.gamma_p3(lowIdx), avg.RPL.gamma_p3(highIdx));
    p_gamma_p3(s) = p;
    
    p_analysis_p3_rpl(s,1:5) = [p_delta_p3(s), p_theta_p3(s), p_alpha_p3(s), p_beta_p3(s), p_gamma_p3(s)];
    p_analysis_p3_rpl_min(s) = find(p_analysis_p3_rpl(s,1:5)==min(p_analysis_p3_rpl(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.RPL.delta_t1(lowIdx), avg.RPL.delta_t1(highIdx));
    p_delta_t1(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.theta_t1(lowIdx), avg.RPL.theta_t1(highIdx));
    p_theta_t1(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.alpha_t1(lowIdx), avg.RPL.alpha_t1(highIdx));
    p_alpha_t1(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.beta_t1(lowIdx), avg.RPL.beta_t1(highIdx));
    p_beta_t1(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.gamma_t1(lowIdx), avg.RPL.gamma_t1(highIdx));
    p_gamma_t1(s) = p;
    
    p_analysis_t1_rpl(s,1:5) = [p_delta_t1(s), p_theta_t1(s), p_alpha_t1(s), p_beta_t1(s), p_gamma_t1(s)];
    p_analysis_t1_rpl_min(s) = find(p_analysis_t1_rpl(s,1:5)==min(p_analysis_t1_rpl(s,1:5)));
    
    [h, p, ci, stats] = ttest2(avg.RPL.delta_t2(lowIdx), avg.RPL.delta_t2(highIdx));
    p_delta_t2(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.theta_t2(lowIdx), avg.RPL.theta_t2(highIdx));
    p_theta_t2(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.alpha_t2(lowIdx), avg.RPL.alpha_t2(highIdx));
    p_alpha_t2(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.beta_t2(lowIdx), avg.RPL.beta_t2(highIdx));
    p_beta_t2(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.gamma_t2(lowIdx), avg.RPL.gamma_t2(highIdx));
    p_gamma_t2(s) = p;
    
    p_analysis_t2_rpl(s,1:5) = [p_delta_t2(s), p_theta_t2(s), p_alpha_t2(s), p_beta_t2(s), p_gamma_t2(s)];
    p_analysis_t2_rpl_min(s) = find(p_analysis_t2_rpl(s,1:5)==min(p_analysis_t2_rpl(s,1:5)));
    
    
    
    [h, p, ci, stats] = ttest2(avg.RPL.delta_o1(lowIdx), avg.RPL.delta_o1(highIdx));
    p_delta_o1(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.theta_o1(lowIdx), avg.RPL.theta_o1(highIdx));
    p_theta_o1(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.alpha_o1(lowIdx), avg.RPL.alpha_o1(highIdx));
    p_alpha_o1(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.beta_o1(lowIdx), avg.RPL.beta_o1(highIdx));
    p_beta_o1(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.gamma_o1(lowIdx), avg.RPL.gamma_o1(highIdx));
    p_gamma_o1(s) = p;
    
    p_analysis_o1_rpl(s,1:5) = [p_delta_o1(s), p_theta_o1(s), p_alpha_o1(s), p_beta_o1(s), p_gamma_o1(s)];
    p_analysis_o1_rpl_min(s) = find(p_analysis_o1_rpl(s,1:5)==min(p_analysis_o1_rpl(s,1:5)));
    
    [h, p, ci, stats] = ttest2(avg.RPL.delta_o2(lowIdx), avg.RPL.delta_o2(highIdx));
    p_delta_o2(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.theta_o2(lowIdx), avg.RPL.theta_o2(highIdx));
    p_theta_o2(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.alpha_o2(lowIdx), avg.RPL.alpha_o2(highIdx));
    p_alpha_o2(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.beta_o2(lowIdx), avg.RPL.beta_o2(highIdx));
    p_beta_o2(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.gamma_o2(lowIdx), avg.RPL.gamma_o2(highIdx));
    p_gamma_o2(s) = p;
    
    p_analysis_o2_rpl(s,1:5) = [p_delta_o2(s), p_theta_o2(s), p_alpha_o2(s), p_beta_o2(s), p_gamma_o2(s)];
    p_analysis_o2_rpl_min(s) = find(p_analysis_o2_rpl(s,1:5)==min(p_analysis_o2_rpl(s,1:5)));
    
    [h, p, ci, stats] = ttest2(avg.RPL.delta_o3(lowIdx), avg.RPL.delta_o3(highIdx));
    p_delta_o3(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.theta_o3(lowIdx), avg.RPL.theta_o3(highIdx));
    p_theta_o3(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.alpha_o3(lowIdx), avg.RPL.alpha_o3(highIdx));
    p_alpha_o3(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.beta_o3(lowIdx), avg.RPL.beta_o3(highIdx));
    p_beta_o3(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.gamma_o3(lowIdx), avg.RPL.gamma_o3(highIdx));
    p_gamma_o3(s) = p;
    
    p_analysis_o3_rpl(s,1:5) = [p_delta_o3(s), p_theta_o3(s), p_alpha_o3(s), p_beta_o3(s), p_gamma_o3(s)];
    p_analysis_o3_rpl_min(s) = find(p_analysis_o3_rpl(s,1:5)==min(p_analysis_o3_rpl(s,1:5)));
    
    Result.ttest.vertical_rpl(s,:) = [p_analysis_f1(s,:) p_analysis_f2(s,:) p_analysis_f3(s,:) p_analysis_c1(s,:) p_analysis_c2(s,:) p_analysis_c3(s,:) p_analysis_p1(s,:) p_analysis_p2(s,:) p_analysis_p3(s,:) p_analysis_t1(s,:) p_analysis_t2(s,:) p_analysis_o1(s,:) p_analysis_o2(s,:) p_analysis_o3(s,:)];
    
    
    
    
    
    
    
            % Regional Vertical T-TEST 분석
    [h, p, ci, stats] = ttest2(avg.z.Regional_f1(lowIdx,1), avg.z.Regional_f1(highIdx,1));
    p_delta_f1(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_f1(lowIdx,2), avg.z.Regional_f1(highIdx,2));
    p_theta_f1(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_f1(lowIdx,3), avg.z.Regional_f1(highIdx,3));
    p_alpha_f1(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_f1(lowIdx,4), avg.z.Regional_f1(highIdx,4));
    p_beta_f1(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_f1(lowIdx,5), avg.z.Regional_f1(highIdx,5));
    p_gamma_f1(s) = p;
    
    p_analysis_f1_z(s,1:5) = [p_delta_f1(s), p_theta_f1(s), p_alpha_f1(s), p_beta_f1(s), p_gamma_f1(s)];
    p_analysis_f1_z_min(s) = find(p_analysis_f1_z(s,1:5)==min(p_analysis_f1_z(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.z.Regional_f2(lowIdx,1), avg.z.Regional_f2(highIdx,1));
    p_delta_f2(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_f2(lowIdx,2), avg.z.Regional_f2(highIdx,2));
    p_theta_f2(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_f2(lowIdx,3), avg.z.Regional_f2(highIdx,3));
    p_alpha_f2(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_f2(lowIdx,4), avg.z.Regional_f2(highIdx,4));
    p_beta_f2(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_f2(lowIdx,5), avg.z.Regional_f2(highIdx,5));
    p_gamma_f2(s) = p;
    
    p_analysis_f2_z(s,1:5) = [p_delta_f2(s), p_theta_f2(s), p_alpha_f2(s), p_beta_f2(s), p_gamma_f2(s)];
    p_analysis_f2_z_min(s) = find(p_analysis_f2_z(s,1:5)==min(p_analysis_f2_z(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.z.Regional_f3(lowIdx,1), avg.z.Regional_f3(highIdx,1));
    p_delta_f3(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_f3(lowIdx,2), avg.z.Regional_f3(highIdx,2));
    p_theta_f3(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_f3(lowIdx,3), avg.z.Regional_f3(highIdx,3));
    p_alpha_f3(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_f3(lowIdx,4), avg.z.Regional_f3(highIdx,4));
    p_beta_f3(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_f3(lowIdx,5), avg.z.Regional_f3(highIdx,5));
    p_gamma_f3(s) = p;
    
    p_analysis_f3_z(s,1:5) = [p_delta_f3(s), p_theta_f3(s), p_alpha_f3(s), p_beta_f3(s), p_gamma_f3(s)];
    p_analysis_f3_z_min(s) = find(p_analysis_f3_z(s,1:5)==min(p_analysis_f3_z(s,1:5)));
    
    
    
    [h, p, ci, stats] = ttest2(avg.z.Regional_c1(lowIdx,1), avg.z.Regional_c1(highIdx,1));
    p_delta_c1(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_c1(lowIdx,2), avg.z.Regional_c1(highIdx,2));
    p_theta_c1(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_c1(lowIdx,3), avg.z.Regional_c1(highIdx,3));
    p_alpha_c1(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_c1(lowIdx,4), avg.z.Regional_c1(highIdx,4));
    p_beta_c1(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_c1(lowIdx,5), avg.z.Regional_c1(highIdx,5));
    p_gamma_c1(s) = p;
    
    p_analysis_c1_z(s,1:5) = [p_delta_c1(s), p_theta_c1(s), p_alpha_c1(s), p_beta_c1(s), p_gamma_c1(s)];
    p_analysis_c1_z_min(s) = find(p_analysis_c1_z(s,1:5)==min(p_analysis_c1_z(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.z.Regional_c2(lowIdx,1), avg.z.Regional_c2(highIdx,1));
    p_delta_c2(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_c2(lowIdx,2), avg.z.Regional_c2(highIdx,2));
    p_theta_c2(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_c2(lowIdx,3), avg.z.Regional_c2(highIdx,3));
    p_alpha_c2(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_c2(lowIdx,4), avg.z.Regional_c2(highIdx,4));
    p_beta_c2(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_c2(lowIdx,5), avg.z.Regional_c2(highIdx,5));
    p_gamma_c2(s) = p;
    
    p_analysis_c2_z(s,1:5) = [p_delta_c2(s), p_theta_c2(s), p_alpha_c2(s), p_beta_c2(s), p_gamma_c2(s)];
    p_analysis_c2_z_min(s) = find(p_analysis_c2_z(s,1:5)==min(p_analysis_c2_z(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.z.Regional_c3(lowIdx,1), avg.z.Regional_c3(highIdx,1));
    p_delta_c3(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_c3(lowIdx,2), avg.z.Regional_c3(highIdx,2));
    p_theta_c3(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_c3(lowIdx,3), avg.z.Regional_c3(highIdx,3));
    p_alpha_c3(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_c3(lowIdx,4), avg.z.Regional_c3(highIdx,4));
    p_beta_c3(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_c3(lowIdx,5), avg.z.Regional_c3(highIdx,5));
    p_gamma_c3(s) = p;
    
    p_analysis_c3_z(s,1:5) = [p_delta_c3(s), p_theta_c3(s), p_alpha_c3(s), p_beta_c3(s), p_gamma_c3(s)];
    p_analysis_c3_z_min(s) = find(p_analysis_c3_z(s,1:5)==min(p_analysis_c3_z(s,1:5)));
    
    

    
    
    [h, p, ci, stats] = ttest2(avg.z.Regional_p1(lowIdx,1), avg.z.Regional_p1(highIdx,1));
    p_delta_p1(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_p1(lowIdx,2), avg.z.Regional_p1(highIdx,2));
    p_theta_p1(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_p1(lowIdx,3), avg.z.Regional_p1(highIdx,3));
    p_alpha_p1(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_p1(lowIdx,4), avg.z.Regional_p1(highIdx,4));
    p_beta_p1(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_p1(lowIdx,5), avg.z.Regional_p1(highIdx,5));
    p_gamma_p1(s) = p;
    
    p_analysis_p1_z(s,1:5) = [p_delta_p1(s), p_theta_p1(s), p_alpha_p1(s), p_beta_p1(s), p_gamma_p1(s)];
    p_analysis_p1_z_min(s) = find(p_analysis_p1_z(s,1:5)==min(p_analysis_p1_z(s,1:5)));
    
    
    
    [h, p, ci, stats] = ttest2(avg.z.Regional_p2(lowIdx,1), avg.z.Regional_p2(highIdx,1));
    p_delta_p2(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_p2(lowIdx,2), avg.z.Regional_p2(highIdx,2));
    p_theta_p2(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_p2(lowIdx,3), avg.z.Regional_p2(highIdx,3));
    p_alpha_p2(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_p2(lowIdx,4), avg.z.Regional_p2(highIdx,4));
    p_beta_p2(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_p2(lowIdx,5), avg.z.Regional_p2(highIdx,5));
    p_gamma_p2(s) = p;
    
    p_analysis_p2_z(s,1:5) = [p_delta_p2(s), p_theta_p2(s), p_alpha_p2(s), p_beta_p2(s), p_gamma_p2(s)];
    p_analysis_p2_z_min(s) = find(p_analysis_p2_z(s,1:5)==min(p_analysis_p2_z(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.z.Regional_p3(lowIdx,1), avg.z.Regional_p3(highIdx,1));
    p_delta_p3(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_p3(lowIdx,2), avg.z.Regional_p3(highIdx,2));
    p_theta_p3(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_p3(lowIdx,3), avg.z.Regional_p3(highIdx,3));
    p_alpha_p3(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_p3(lowIdx,4), avg.z.Regional_p3(highIdx,4));
    p_beta_p3(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_p3(lowIdx,5), avg.z.Regional_p3(highIdx,5));
    p_gamma_p3(s) = p;
    
    p_analysis_p3_z(s,1:5) = [p_delta_p3(s), p_theta_p3(s), p_alpha_p3(s), p_beta_p3(s), p_gamma_p3(s)];
    p_analysis_p3_z_min(s) = find(p_analysis_p3_z(s,1:5)==min(p_analysis_p3_z(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.z.Regional_t1(lowIdx,1), avg.z.Regional_t1(highIdx,1));
    p_delta_t1(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_t1(lowIdx,2), avg.z.Regional_t1(highIdx,2));
    p_theta_t1(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_t1(lowIdx,3), avg.z.Regional_t1(highIdx,3));
    p_alpha_t1(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_t1(lowIdx,4), avg.z.Regional_t1(highIdx,4));
    p_beta_t1(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_t1(lowIdx,5), avg.z.Regional_t1(highIdx,5));
    p_gamma_t1(s) = p;
    
    p_analysis_t1_z(s,1:5) = [p_delta_t1(s), p_theta_t1(s), p_alpha_t1(s), p_beta_t1(s), p_gamma_t1(s)];
    p_analysis_t1_z_min(s) = find(p_analysis_t1_z(s,1:5)==min(p_analysis_t1_z(s,1:5)));
    
    [h, p, ci, stats] = ttest2(avg.z.Regional_t2(lowIdx,1), avg.z.Regional_t2(highIdx,1));
    p_delta_t2(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_t2(lowIdx,2), avg.z.Regional_t2(highIdx,2));
    p_theta_t2(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_t2(lowIdx,3), avg.z.Regional_t2(highIdx,3));
    p_alpha_t2(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_t2(lowIdx,4), avg.z.Regional_t2(highIdx,4));
    p_beta_t2(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_t2(lowIdx,5), avg.z.Regional_t2(highIdx,5));
    p_gamma_t2(s) = p;
    
    p_analysis_t2_z(s,1:5) = [p_delta_t2(s), p_theta_t2(s), p_alpha_t2(s), p_beta_t2(s), p_gamma_t2(s)];
    p_analysis_t2_z_min(s) = find(p_analysis_t2_z(s,1:5)==min(p_analysis_t2_z(s,1:5)));
    
    
    
    [h, p, ci, stats] = ttest2(avg.z.Regional_o1(lowIdx,1), avg.z.Regional_o1(highIdx,1));
    p_delta_o1(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_o1(lowIdx,2), avg.z.Regional_o1(highIdx,2));
    p_theta_o1(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_o1(lowIdx,3), avg.z.Regional_o1(highIdx,3));
    p_alpha_o1(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_o1(lowIdx,4), avg.z.Regional_o1(highIdx,4));
    p_beta_o1(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_o1(lowIdx,5), avg.z.Regional_o1(highIdx,5));
    p_gamma_o1(s) = p;
    
    p_analysis_o1_z(s,1:5) = [p_delta_o1(s), p_theta_o1(s), p_alpha_o1(s), p_beta_o1(s), p_gamma_o1(s)];
    p_analysis_o1_z_min(s) = find(p_analysis_o1_z(s,1:5)==min(p_analysis_o1_z(s,1:5)));
    
    [h, p, ci, stats] = ttest2(avg.z.Regional_o2(lowIdx,1), avg.z.Regional_o2(highIdx,1));
    p_delta_o2(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_o2(lowIdx,2), avg.z.Regional_o2(highIdx,2));
    p_theta_o2(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_o2(lowIdx,3), avg.z.Regional_o2(highIdx,3));
    p_alpha_o2(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_o2(lowIdx,4), avg.z.Regional_o2(highIdx,4));
    p_beta_o2(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_o2(lowIdx,5), avg.z.Regional_o2(highIdx,5));
    p_gamma_o2(s) = p;
    
    p_analysis_o2_z(s,1:5) = [p_delta_o2(s), p_theta_o2(s), p_alpha_o2(s), p_beta_o2(s), p_gamma_o2(s)];
    p_analysis_o2_z_min(s) = find(p_analysis_o2_z(s,1:5)==min(p_analysis_o2_z(s,1:5)));
    
    [h, p, ci, stats] = ttest2(avg.z.Regional_o3(lowIdx,1), avg.z.Regional_o3(highIdx,1));
    p_delta_o3(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_o3(lowIdx,2), avg.z.Regional_o3(highIdx,2));
    p_theta_o3(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_o3(lowIdx,3), avg.z.Regional_o3(highIdx,3));
    p_alpha_o3(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_o3(lowIdx,4), avg.z.Regional_o3(highIdx,4));
    p_beta_o3(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_o3(lowIdx,5), avg.z.Regional_o3(highIdx,5));
    p_gamma_o3(s) = p;
    
    p_analysis_o3_z(s,1:5) = [p_delta_o3(s), p_theta_o3(s), p_alpha_o3(s), p_beta_o3(s), p_gamma_o3(s)];
    p_analysis_o3_z_min(s) = find(p_analysis_o3_z(s,1:5)==min(p_analysis_o3_z(s,1:5)));
    
    Result.ttest.vertical_z(s,:) = [p_analysis_f1(s,:) p_analysis_f2(s,:) p_analysis_f3(s,:) p_analysis_c1(s,:) p_analysis_c2(s,:) p_analysis_c3(s,:) p_analysis_p1(s,:) p_analysis_p2(s,:) p_analysis_p3(s,:) p_analysis_t1(s,:) p_analysis_t2(s,:) p_analysis_o1(s,:) p_analysis_o2(s,:) p_analysis_o3(s,:)];
    
    
    
    
    
    
    
    
    
    
    %% Feature Selection
    feature_ALL = avg.PSD;
    feature_PSD = avg.PSD(:,find(p_analysis(s,:)<0.05));
    feature_Regional = [avg.Regional_f(:,find(p_analysis_f(s,:)<0.05)),avg.Regional_c(:,find(p_analysis_c(s,:)<0.05)),avg.Regional_p(:,find(p_analysis_p(s,:)<0.05)),avg.Regional_t(:,find(p_analysis_t(s,:)<0.05)),avg.Regional_o(:,find(p_analysis_o(s,:)<0.05))];
    feature_Vertical = [avg.Regional_f1(:,find(p_analysis_f1(s,:)<0.05)),avg.Regional_f2(:,find(p_analysis_f2(s,:)<0.05)),avg.Regional_f3(:,find(p_analysis_f3(s,:)<0.05)),avg.Regional_c1(:,find(p_analysis_c1(s,:)<0.05)),avg.Regional_c2(:,find(p_analysis_c2(s,:)<0.05)),avg.Regional_c3(:,find(p_analysis_c3(s,:)<0.05)),avg.Regional_p1(:,find(p_analysis_p1(s,:)<0.05)),avg.Regional_p2(:,find(p_analysis_p2(s,:)<0.05)),avg.Regional_p3(:,find(p_analysis_p3(s,:)<0.05)),avg.Regional_t1(:,find(p_analysis_t1(s,:)<0.05)),avg.Regional_t2(:,find(p_analysis_t2(s,:)<0.05)),avg.Regional_o1(:,find(p_analysis_o1(s,:)<0.05)),avg.Regional_o2(:,find(p_analysis_o2(s,:)<0.05)),avg.Regional_o3(:,find(p_analysis_o3(s,:)<0.05))];
    
    feature_ALL_RPL = avg.RPL.PSD;
    feature_PSD_RPL = avg.RPL.PSD(:,find(p_analysis_RPL(s,:)<0.05));
    feature_Regional_RPL = [avg.RPL.Regional_f(:,find(p_analysis_RPL_f(s,:)<0.05)),avg.RPL.Regional_c(:,find(p_analysis_RPL_c(s,:)<0.05)),avg.RPL.Regional_p(:,find(p_analysis_RPL_p(s,:)<0.05)),avg.RPL.Regional_t(:,find(p_analysis_RPL_t(s,:)<0.05)),avg.RPL.Regional_o(:,find(p_analysis_RPL_o(s,:)<0.05))];
    feature_Vertical_RPL = [avg.RPL.Regional_f1(:,find(p_analysis_f1_rpl(s,:)<0.05)),avg.RPL.Regional_f2(:,find(p_analysis_f2_rpl(s,:)<0.05)),avg.RPL.Regional_f3(:,find(p_analysis_f3_rpl(s,:)<0.05)),avg.RPL.Regional_c1(:,find(p_analysis_c1_rpl(s,:)<0.05)),avg.RPL.Regional_c2(:,find(p_analysis_c2_rpl(s,:)<0.05)),avg.RPL.Regional_c3(:,find(p_analysis_c3_rpl(s,:)<0.05)),avg.RPL.Regional_p1(:,find(p_analysis_p1_rpl(s,:)<0.05)),avg.RPL.Regional_p2(:,find(p_analysis_p2_rpl(s,:)<0.05)),avg.RPL.Regional_p3(:,find(p_analysis_p3_rpl(s,:)<0.05)),avg.RPL.Regional_t1(:,find(p_analysis_t1_rpl(s,:)<0.05)),avg.RPL.Regional_t2(:,find(p_analysis_t2_rpl(s,:)<0.05)),avg.RPL.Regional_o1(:,find(p_analysis_o1_rpl(s,:)<0.05)),avg.RPL.Regional_o2(:,find(p_analysis_o2_rpl(s,:)<0.05)),avg.RPL.Regional_o3(:,find(p_analysis_o3_rpl(s,:)<0.05))];

    feature_ALL_z = avg.z.PSD;
    feature_PSD_z = avg.z.PSD(:,find(p_analysis_z(s,:)<0.05));
    feature_Regional_z = [avg.z.Regional_f(:,find(p_analysis_z_f(s,:)<0.05)),avg.z.Regional_c(:,find(p_analysis_z_c(s,:)<0.05)),avg.z.Regional_p(:,find(p_analysis_z_p(s,:)<0.05)),avg.z.Regional_t(:,find(p_analysis_z_t(s,:)<0.05)),avg.z.Regional_o(:,find(p_analysis_z_o(s,:)<0.05))];
    feature_Vertical_z = [avg.z.Regional_f1(:,find(p_analysis_f1_z(s,:)<0.05)),avg.z.Regional_f2(:,find(p_analysis_f2_z(s,:)<0.05)),avg.z.Regional_f3(:,find(p_analysis_f3_z(s,:)<0.05)),avg.z.Regional_c1(:,find(p_analysis_c1_z(s,:)<0.05)),avg.z.Regional_c2(:,find(p_analysis_c2_z(s,:)<0.05)),avg.z.Regional_c3(:,find(p_analysis_c3_z(s,:)<0.05)),avg.z.Regional_p1(:,find(p_analysis_p1_z(s,:)<0.05)),avg.z.Regional_p2(:,find(p_analysis_p2_z(s,:)<0.05)),avg.z.Regional_p3(:,find(p_analysis_p3_z(s,:)<0.05)),avg.z.Regional_t1(:,find(p_analysis_t1_z(s,:)<0.05)),avg.z.Regional_t2(:,find(p_analysis_t2_z(s,:)<0.05)),avg.z.Regional_o1(:,find(p_analysis_o1_z(s,:)<0.05)),avg.z.Regional_o2(:,find(p_analysis_o2_z(s,:)<0.05)),avg.z.Regional_o3(:,find(p_analysis_o3_z(s,:)<0.05))];
    
    %% Classification
    [Accuracy_ALL AUC_ALL] = LDA_4CV(feature_ALL, highIdx, lowIdx, s);
    [Accuracy_PSD AUC_PSD] = LDA_4CV(feature_PSD, highIdx, lowIdx, s);
    [Accuracy_Regional AUC_Regional] = LDA_4CV(feature_Regional, highIdx, lowIdx, s);
    [Accuracy_Vertical AUC_Vertical] = LDA_4CV(feature_Vertical, highIdx, lowIdx, s);
    
    [Accuracy_ALL_RPL AUC_ALL_RPL] = LDA_4CV(feature_ALL_RPL, highIdx, lowIdx, s);
    [Accuracy_PSD_RPL AUC_PSD_RPL] = LDA_4CV(feature_PSD_RPL, highIdx, lowIdx, s);
    [Accuracy_Regional_RPL AUC_Regional_RPL] = LDA_4CV(feature_Regional_RPL, highIdx, lowIdx, s);
    [Accuracy_Vertical_RPL AUC_Vertical_RPL] = LDA_4CV(feature_Vertical_RPL, highIdx, lowIdx, s);
    
    [Accuracy_ALL_z AUC_ALL_z] = LDA_4CV(feature_ALL_z, highIdx, lowIdx, s);
    [Accuracy_PSD_z AUC_PSD_z] = LDA_4CV(feature_PSD_z, highIdx, lowIdx, s);
    [Accuracy_Regional_z AUC_Regional_z] = LDA_4CV(feature_Regional_z, highIdx, lowIdx, s);
    [Accuracy_Vertical_z AUC_Vertical_z] = LDA_4CV(feature_Vertical_z, highIdx, lowIdx, s);
    
    Result.Fatigue_Accuracy(s,:) = [Accuracy_ALL Accuracy_PSD Accuracy_Regional Accuracy_Vertical];
    Result.Fatigue_AUC(s,:) = [AUC_ALL AUC_PSD AUC_Regional AUC_Vertical];
    
    Result.Fatigue_Accuracy_RPL(s,:) = [Accuracy_ALL_RPL Accuracy_PSD_RPL Accuracy_Regional_RPL Accuracy_Vertical_RPL];
    Result.Fatigue_AUC_RPL(s,:) = [AUC_ALL_RPL AUC_PSD_RPL AUC_Regional_RPL AUC_Vertical_RPL];
    
    Result.Fatigue_Accuracy_z(s,:) = [Accuracy_ALL_z Accuracy_PSD_z Accuracy_Regional_z Accuracy_Vertical_z];
    Result.Fatigue_AUC_z(s,:) = [AUC_ALL_z AUC_PSD_z AUC_Regional_z AUC_Vertical_z];
    
%     %% Regression
%     [Prediction_ALL, RMSE_ALL, Error_Rate_ALL] = MLR_4CV(feature_ALL, kss, s);
%     [Prediction_PSD, RMSE_PSD, Error_Rate_PSD] = MLR_4CV(feature_PSD, kss, s);
%     [Prediction_Regional, RMSE_Regional, Error_Rate_Regional] = MLR_4CV(feature_Regional, kss, s);
%     [Prediction_Vertical, RMSE_Vertical, Error_Rate_Vertical] = MLR_4CV(feature_Vertical, kss, s);
%     
%     [Prediction_ALL_RPL, RMSE_ALL_RPL, Error_Rate_ALL_RPL] = MLR_4CV(feature_ALL_RPL, kss, s);
%     [Prediction_PSD_RPL, RMSE_PSD_RPL, Error_Rate_PSD_RPL] = MLR_4CV(feature_PSD_RPL, kss, s);
%     [Prediction_Regional_RPL, RMSE_Regional_RPL, Error_Rate_Regional_RPL] = MLR_4CV(feature_Regional_RPL, kss, s);
%     [Prediction_Vertical_RPL, RMSE_Vertical_RPL, Error_Rate_Vertical_RPL] = MLR_4CV(feature_Vertical_RPL, kss, s);
%     
%     [Prediction_ALL_z, RMSE_ALL_z, Error_Rate_ALL_z] = MLR_4CV(feature_ALL_z, kss, s);
%     [Prediction_PSD_z, RMSE_PSD_z, Error_Rate_PSD_z] = MLR_4CV(feature_PSD_z, kss, s);
%     [Prediction_Regional_z, RMSE_Regional_z, Error_Rate_Regional_z] = MLR_4CV(feature_Regional_z, kss, s);
%     [Prediction_Vertical_z, RMSE_Vertical_z, Error_Rate_Vertical_z] = MLR_4CV(feature_Vertical_z, kss, s);
%  
%     Result.Fatigue_Prediction(s,:) = [Prediction_ALL Prediction_PSD Prediction_Regional Prediction_Vertical];
%     Result.Fatigue_RMSE(s,:) = [RMSE_ALL RMSE_PSD RMSE_Regional RMSE_Vertical];   
%     Result.Fatigue_Error(s,:) = [Error_Rate_ALL Error_Rate_PSD Error_Rate_Regional Error_Rate_Vertical];   
%     
%     Result.Fatigue_Prediction_RPL(s,:) = [Prediction_ALL_RPL Prediction_PSD_RPL Prediction_Regional_RPL Prediction_Vertical_RPL];
%     Result.Fatigue_RMSE_RPL(s,:) = [RMSE_ALL_RPL RMSE_PSD_RPL RMSE_Regional_RPL RMSE_Vertical_RPL];   
%     Result.Fatigue_Error_RPL(s,:) = [Error_Rate_ALL_RPL Error_Rate_PSD_RPL Error_Rate_Regional_RPL Error_Rate_Vertical_RPL];   
%     
%     Result.Fatigue_Prediction_z(s,:) = [Prediction_ALL_z Prediction_PSD_z Prediction_Regional_z Prediction_Vertical_z];
%     Result.Fatigue_RMSE_z(s,:) = [RMSE_ALL_z RMSE_PSD_z RMSE_Regional_z RMSE_Vertical_z];   
%     Result.Fatigue_Error_z(s,:) = [Error_Rate_ALL_z Error_Rate_PSD_z Error_Rate_Regional_z Error_Rate_Vertical_z];   


    %% Save
    Result.Fatigue_Accuracy_20(s,:) = Result.Fatigue_Accuracy(s,:);
    Result.Fatigue_AUC_20(s,:) = Result.Fatigue_AUC(s,:);  
%     Result.Fatigue_Prediction_20(s,:) = Result.Fatigue_Prediction(s,:);
%     Result.Fatigue_RMSE_20(s,:) = Result.Fatigue_RMSE(s,:);
%     Result.Fatigue_Error_20(s,:) = Result.Fatigue_Error(s,:);
    
    Result.Fatigue_Accuracy_20_RPL(s,:) = Result.Fatigue_Accuracy_RPL(s,:);
    Result.Fatigue_AUC_20_RPL(s,:) = Result.Fatigue_AUC_RPL(s,:);  
%     Result.Fatigue_Prediction_20_RPL(s,:) = Result.Fatigue_Prediction_RPL(s,:);
%     Result.Fatigue_RMSE_20_RPL(s,:) = Result.Fatigue_RMSE_RPL(s,:);
%     Result.Fatigue_Error_20_RPL(s,:) = Result.Fatigue_Error_RPL(s,:);
    
    Result.Fatigue_Accuracy_20_z(s,:) = Result.Fatigue_Accuracy_z(s,:);
    Result.Fatigue_AUC_20_z(s,:) = Result.Fatigue_AUC_z(s,:);  
%     Result.Fatigue_Prediction_20_z(s,:) = Result.Fatigue_Prediction_z(s,:);
%     Result.Fatigue_RMSE_20_z(s,:) = Result.Fatigue_RMSE_z(s,:);
%     Result.Fatigue_Error_20_z(s,:) = Result.Fatigue_Error_z(s,:);

    
    
    
    
    %% Segmentation of bio-signals with 1 seconds length of epoch
    epoch = segmentationFatigue_30(cnt, mrk, interKSS);
    kssIdx = find(strcmp(epoch.mClab, 'KSS'));
    kss = squeeze(mean(epoch.misc(:, kssIdx, :), 1));
    
    %% Regional Channel Selection
    channel_f=epoch.x(:,[1:5,31:36],:);
    channel_c=epoch.x(:,[6:9,11:13,16:19,39:40,43:46,48:50],:);
    channel_p=epoch.x(:,[21:25,52:55],:);
    channel_t=epoch.x(:,[10,14:15,20,37:38,41:42,47,51],:);
    channel_o=epoch.x(:,[26:30,56:60],:);
    
    channel_f1=epoch.x(:,[1,3,31,33],:);
    channel_f2=epoch.x(:,[2,5,32,36],:);
    channel_f3=epoch.x(:,[4,34:35],:);
    channel_c1=epoch.x(:,[6:7,11,16,17,39,43,44,48],:);
    channel_c2=epoch.x(:,[8:9,13,18,19,40,45,46,50],:);
    channel_c3=epoch.x(:,[12,49],:);
    channel_p1=epoch.x(:,[21:22,52,53],:);
    channel_p2=epoch.x(:,[24:25,54,55],:);
    channel_p3=epoch.x(:,[23],:);
    channel_t1=epoch.x(:,[10,15,37,38,47],:);
    channel_t2=epoch.x(:,[14,20,41,42,51],:);
    channel_o1=epoch.x(:,[26,27,56,57],:);
    channel_o2=epoch.x(:,[29,30,59,60],:);
    channel_o3=epoch.x(:,[28,58],:);
    
    %% Feature Extraction - Power spectrum analysis for EEG signals
    
    % PSD Feature Extraction
    [cca.delta, avg.delta] = Power_spectrum(epoch.x, [0.5 4], cnt.fs);
    [cca.theta, avg.theta] = Power_spectrum(epoch.x, [4 8], cnt.fs);
    [cca.alpha, avg.alpha] = Power_spectrum(epoch.x, [8 13], cnt.fs);
    [cca.beta, avg.beta] = Power_spectrum(epoch.x, [13 30], cnt.fs);
    [cca.gamma, avg.gamma] = Power_spectrum(epoch.x, [30 40], cnt.fs);
    avg.delta = avg.delta'; avg.theta=avg.theta'; avg.alpha=avg.alpha'; avg.beta = avg.beta'; avg.gamma = avg.gamma';
    
    cca.PSD = [cca.delta cca.theta cca.alpha cca.beta cca.gamma];
    avg.PSD = [avg.delta avg.theta avg.alpha avg.beta avg.gamma];

    % RPL Feature Extraction
    cca.RPL.delta = cca.delta ./ (cca.delta + cca.theta + cca.alpha + cca.beta + cca.gamma) ;
    avg.RPL.delta = avg.delta ./ sum(avg.PSD,2) ;
    cca.RPL.theta = cca.theta ./ (cca.delta + cca.theta + cca.alpha + cca.beta + cca.gamma) ;
    avg.RPL.theta = avg.theta ./ sum(avg.PSD,2) ;
    cca.RPL.alpha = cca.alpha ./ (cca.delta + cca.theta + cca.alpha + cca.beta + cca.gamma) ;
    avg.RPL.alpha = avg.alpha ./ sum(avg.PSD,2) ;
    cca.RPL.beta = cca.beta ./ (cca.delta + cca.theta + cca.alpha + cca.beta + cca.gamma) ;
    avg.RPL.beta = avg.beta ./ sum(avg.PSD,2) ;
    cca.RPL.gamma = cca.gamma ./ (cca.delta + cca.theta + cca.alpha + cca.beta + cca.gamma) ;
    avg.RPL.gamma = avg.gamma ./ sum(avg.PSD,2) ;
    
    cca.RPL.PSD = [cca.RPL.delta, cca.RPL.theta, cca.RPL.alpha, cca.RPL.beta, cca.RPL.gamma];
    avg.RPL.PSD = [avg.RPL.delta, avg.RPL.theta, avg.RPL.alpha, avg.RPL.beta, avg.RPL.gamma];
    
    % Z-SCORE Feature Extraction
    cca.z.PSD = [zscore(cca.delta) zscore(cca.theta) zscore(cca.alpha) zscore(cca.beta) zscore(cca.gamma)];
    avg.z.PSD = [zscore(avg.delta) zscore(avg.theta) zscore(avg.alpha) zscore(avg.beta) zscore(avg.gamma)];
    
    
    % Regional Feature Extraction
    [cca.delta_f,avg.delta_f] = Power_spectrum(channel_f, [0.5 4], cnt.fs);
    [cca.theta_f,avg.theta_f] = Power_spectrum(channel_f, [4 8], cnt.fs);
    [cca.alpha_f,avg.alpha_f] = Power_spectrum(channel_f, [8 13], cnt.fs);
    [cca.beta_f,avg.beta_f] = Power_spectrum(channel_f, [13 30], cnt.fs);
    [cca.gamma_f,avg.gamma_f] = Power_spectrum(channel_f, [30 40], cnt.fs);
    avg.delta_f = avg.delta_f'; avg.theta_f=avg.theta_f'; avg.alpha_f=avg.alpha_f'; avg.beta_f = avg.beta_f'; avg.gamma_f = avg.gamma_f';
    
    cca.Regional_f = [cca.delta_f cca.theta_f cca.alpha_f cca.beta_f cca.gamma_f];
    avg.Regional_f = [avg.delta_f avg.theta_f avg.alpha_f avg.beta_f avg.gamma_f];
    
    cca.RPL.delta_f = cca.delta_f ./ (cca.delta_f + cca.theta_f + cca.alpha_f + cca.beta_f + cca.gamma_f) ;
    avg.RPL.delta_f = avg.delta_f ./ sum(avg.Regional_f,2) ;
    cca.RPL.theta_f = cca.theta_f ./ (cca.delta_f + cca.theta_f + cca.alpha_f + cca.beta_f + cca.gamma_f) ;
    avg.RPL.theta_f = avg.theta_f ./ sum(avg.Regional_f,2) ;
    cca.RPL.alpha_f = cca.alpha_f ./ (cca.delta_f + cca.theta_f + cca.alpha_f + cca.beta_f + cca.gamma_f) ;
    avg.RPL.alpha_f = avg.alpha_f ./ sum(avg.Regional_f,2) ;
    cca.RPL.beta_f = cca.beta_f ./ (cca.delta_f + cca.theta_f + cca.alpha_f + cca.beta_f + cca.gamma_f) ;
    avg.RPL.beta_f = avg.beta_f ./ sum(avg.Regional_f,2) ;
    cca.RPL.gamma_f = cca.gamma_f ./ (cca.delta_f + cca.theta_f + cca.alpha_f + cca.beta_f + cca.gamma_f) ;
    avg.RPL.gamma_f = avg.gamma_f ./ sum(avg.Regional_f,2) ;    
    
    cca.RPL.Regional_f = [cca.RPL.delta_f cca.RPL.theta_f cca.RPL.alpha_f cca.RPL.beta_f cca.RPL.gamma_f]; 
    avg.RPL.Regional_f = [avg.RPL.delta_f avg.RPL.theta_f avg.RPL.alpha_f avg.RPL.beta_f avg.RPL.gamma_f];
    
    cca.z.Regional_f = [zscore(cca.delta_f) zscore(cca.theta_f) zscore(cca.alpha_f) zscore(cca.beta_f) zscore(cca.gamma_f)];
    avg.z.Regional_f = [zscore(avg.delta_f) zscore(avg.theta_f) zscore(avg.alpha_f) zscore(avg.beta_f) zscore(avg.gamma_f)];   
    
    
    
    [cca.delta_c,avg.delta_c] = Power_spectrum(channel_c, [0.5 4], cnt.fs);
    [cca.theta_c,avg.theta_c] = Power_spectrum(channel_c, [4 8], cnt.fs);
    [cca.alpha_c,avg.alpha_c] = Power_spectrum(channel_c, [8 13], cnt.fs);
    [cca.beta_c,avg.beta_c] = Power_spectrum(channel_c, [13 30], cnt.fs);
    [cca.gamma_c,avg.gamma_c] = Power_spectrum(channel_c, [30 40], cnt.fs);
    avg.delta_c = avg.delta_c'; avg.theta_c=avg.theta_c'; avg.alpha_c=avg.alpha_c'; avg.beta_c = avg.beta_c'; avg.gamma_c = avg.gamma_c';
    
    cca.Regional_c = [cca.delta_c cca.theta_c cca.alpha_c cca.beta_c cca.gamma_c];
    avg.Regional_c = [avg.delta_c avg.theta_c avg.alpha_c avg.beta_c avg.gamma_c];
    
    cca.RPL.delta_c = cca.delta_c ./ (cca.delta_c + cca.theta_c + cca.alpha_c + cca.beta_c + cca.gamma_c) ;
    avg.RPL.delta_c = avg.delta_c ./ sum(avg.Regional_c,2) ;
    cca.RPL.theta_c = cca.theta_c ./ (cca.delta_c + cca.theta_c + cca.alpha_c + cca.beta_c + cca.gamma_c) ;
    avg.RPL.theta_c = avg.theta_c ./ sum(avg.Regional_c,2) ;
    cca.RPL.alpha_c = cca.alpha_c ./ (cca.delta_c + cca.theta_c + cca.alpha_c + cca.beta_c + cca.gamma_c) ;
    avg.RPL.alpha_c = avg.alpha_c ./ sum(avg.Regional_c,2) ;
    cca.RPL.beta_c = cca.beta_c ./ (cca.delta_c + cca.theta_c + cca.alpha_c + cca.beta_c + cca.gamma_c) ;
    avg.RPL.beta_c = avg.beta_c ./ sum(avg.Regional_c,2) ;
    cca.RPL.gamma_c = cca.gamma_c ./ (cca.delta_c + cca.theta_c + cca.alpha_c + cca.beta_c + cca.gamma_c) ;
    avg.RPL.gamma_c = avg.gamma_c ./ sum(avg.Regional_c,2) ;     
    
    cca.RPL.Regional_c = [cca.RPL.delta_c cca.RPL.theta_c cca.RPL.alpha_c cca.RPL.beta_c cca.RPL.gamma_c]; 
    avg.RPL.Regional_c = [avg.RPL.delta_c avg.RPL.theta_c avg.RPL.alpha_c avg.RPL.beta_c avg.RPL.gamma_c];
    
    cca.z.Regional_c = [zscore(cca.delta_c) zscore(cca.theta_c) zscore(cca.alpha_c) zscore(cca.beta_c) zscore(cca.gamma_c)];
    avg.z.Regional_c = [zscore(avg.delta_c) zscore(avg.theta_c) zscore(avg.alpha_c) zscore(avg.beta_c) zscore(avg.gamma_c)];   
    
    
    [cca.delta_p,avg.delta_p] = Power_spectrum(channel_p, [0.5 4], cnt.fs);
    [cca.theta_p,avg.theta_p] = Power_spectrum(channel_p, [4 8], cnt.fs);
    [cca.alpha_p,avg.alpha_p] = Power_spectrum(channel_p, [8 13], cnt.fs);
    [cca.beta_p,avg.beta_p] = Power_spectrum(channel_p, [13 30], cnt.fs);
    [cca.gamma_p,avg.gamma_p] = Power_spectrum(channel_p, [30 40], cnt.fs);
    avg.delta_p = avg.delta_p'; avg.theta_p=avg.theta_p'; avg.alpha_p=avg.alpha_p'; avg.beta_p = avg.beta_p'; avg.gamma_p = avg.gamma_p';
    
    cca.Regional_p = [cca.delta_p cca.theta_p cca.alpha_p cca.beta_p cca.gamma_p];
    avg.Regional_p = [avg.delta_p avg.theta_p avg.alpha_p avg.beta_p avg.gamma_p];

    cca.RPL.delta_p = cca.delta_p ./ (cca.delta_p + cca.theta_p + cca.alpha_p + cca.beta_p + cca.gamma_p) ;
    avg.RPL.delta_p = avg.delta_p ./ sum(avg.Regional_p,2) ;
    cca.RPL.theta_p = cca.theta_p ./ (cca.delta_p + cca.theta_p + cca.alpha_p + cca.beta_p + cca.gamma_p) ;
    avg.RPL.theta_p = avg.theta_p ./ sum(avg.Regional_p,2) ;
    cca.RPL.alpha_p = cca.alpha_p ./ (cca.delta_p + cca.theta_p + cca.alpha_p + cca.beta_p + cca.gamma_p) ;
    avg.RPL.alpha_p = avg.alpha_p ./ sum(avg.Regional_p,2) ;
    cca.RPL.beta_p = cca.beta_p ./ (cca.delta_p + cca.theta_p + cca.alpha_p + cca.beta_p + cca.gamma_p) ;
    avg.RPL.beta_p = avg.beta_p ./ sum(avg.Regional_p,2) ;
    cca.RPL.gamma_p = cca.gamma_p ./ (cca.delta_p + cca.theta_p + cca.alpha_p + cca.beta_p + cca.gamma_p) ;
    avg.RPL.gamma_p = avg.gamma_p ./ sum(avg.Regional_p,2) ;     
    
    cca.RPL.Regional_p = [cca.RPL.delta_p cca.RPL.theta_p cca.RPL.alpha_p cca.RPL.beta_p cca.RPL.gamma_p]; 
    avg.RPL.Regional_p = [avg.RPL.delta_p avg.RPL.theta_p avg.RPL.alpha_p avg.RPL.beta_p avg.RPL.gamma_p];
    
    cca.z.Regional_p = [zscore(cca.delta_p) zscore(cca.theta_p) zscore(cca.alpha_p) zscore(cca.beta_p) zscore(cca.gamma_p)];
    avg.z.Regional_p = [zscore(avg.delta_p) zscore(avg.theta_p) zscore(avg.alpha_p) zscore(avg.beta_p) zscore(avg.gamma_p)];   
    
    
    [cca.delta_t,avg.delta_t] = Power_spectrum(channel_t, [0.5 4], cnt.fs);
    [cca.theta_t,avg.theta_t] = Power_spectrum(channel_t, [4 8], cnt.fs);
    [cca.alpha_t,avg.alpha_t] = Power_spectrum(channel_t, [8 13], cnt.fs);
    [cca.beta_t,avg.beta_t] = Power_spectrum(channel_t, [13 30], cnt.fs);
    [cca.gamma_t,avg.gamma_t] = Power_spectrum(channel_t, [30 40], cnt.fs);
    avg.delta_t = avg.delta_t'; avg.theta_t=avg.theta_t'; avg.alpha_t=avg.alpha_t'; avg.beta_t = avg.beta_t'; avg.gamma_t = avg.gamma_t';
    
    cca.Regional_t = [cca.delta_t cca.theta_t cca.alpha_t cca.beta_t cca.gamma_t];
    avg.Regional_t = [avg.delta_t avg.theta_t avg.alpha_t avg.beta_t avg.gamma_t];

    cca.RPL.delta_t = cca.delta_t ./ (cca.delta_t + cca.theta_t + cca.alpha_t + cca.beta_t + cca.gamma_t) ;
    avg.RPL.delta_t = avg.delta_t ./ sum(avg.Regional_t,2) ;
    cca.RPL.theta_t = cca.theta_t ./ (cca.delta_t + cca.theta_t + cca.alpha_t + cca.beta_t + cca.gamma_t) ;
    avg.RPL.theta_t = avg.theta_t ./ sum(avg.Regional_t,2) ;
    cca.RPL.alpha_t = cca.alpha_t ./ (cca.delta_t + cca.theta_t + cca.alpha_t + cca.beta_t + cca.gamma_t) ;
    avg.RPL.alpha_t = avg.alpha_t ./ sum(avg.Regional_t,2) ;
    cca.RPL.beta_t = cca.beta_t ./ (cca.delta_t + cca.theta_t + cca.alpha_t + cca.beta_t + cca.gamma_t) ;
    avg.RPL.beta_t = avg.beta_t ./ sum(avg.Regional_t,2) ;
    cca.RPL.gamma_t = cca.gamma_t ./ (cca.delta_t + cca.theta_t + cca.alpha_t + cca.beta_t + cca.gamma_t) ;
    avg.RPL.gamma_t = avg.gamma_t ./ sum(avg.Regional_t,2) ;     
    
    cca.RPL.Regional_t = [cca.RPL.delta_t cca.RPL.theta_t cca.RPL.alpha_t cca.RPL.beta_t cca.RPL.gamma_t]; 
    avg.RPL.Regional_t = [avg.RPL.delta_t avg.RPL.theta_t avg.RPL.alpha_t avg.RPL.beta_t avg.RPL.gamma_t];
    
    cca.z.Regional_t = [zscore(cca.delta_t) zscore(cca.theta_t) zscore(cca.alpha_t) zscore(cca.beta_t) zscore(cca.gamma_t)];
    avg.z.Regional_t = [zscore(avg.delta_t) zscore(avg.theta_t) zscore(avg.alpha_t) zscore(avg.beta_t) zscore(avg.gamma_t)];   
    
    
    [cca.delta_o,avg.delta_o] = Power_spectrum(channel_o, [0.5 4], cnt.fs);
    [cca.theta_o,avg.theta_o] = Power_spectrum(channel_o, [4 8], cnt.fs);
    [cca.alpha_o,avg.alpha_o] = Power_spectrum(channel_o, [8 13], cnt.fs);
    [cca.beta_o,avg.beta_o] = Power_spectrum(channel_o, [13 30], cnt.fs);
    [cca.gamma_o,avg.gamma_o] = Power_spectrum(channel_o, [30 40], cnt.fs);
    avg.delta_o = avg.delta_o'; avg.theta_o=avg.theta_o'; avg.alpha_o=avg.alpha_o'; avg.beta_o = avg.beta_o'; avg.gamma_o = avg.gamma_o';
    
    cca.Regional_o = [cca.delta_o cca.theta_o cca.alpha_o cca.beta_o cca.gamma_o];
    avg.Regional_o = [avg.delta_o avg.theta_o avg.alpha_o avg.beta_o avg.gamma_o];
    
    cca.RPL.delta_o = cca.delta_o ./ (cca.delta_o + cca.theta_o + cca.alpha_o + cca.beta_o + cca.gamma_o) ;
    avg.RPL.delta_o = avg.delta_o ./ sum(avg.Regional_o,2) ;
    cca.RPL.theta_o = cca.theta_o ./ (cca.delta_o + cca.theta_o + cca.alpha_o + cca.beta_o + cca.gamma_o) ;
    avg.RPL.theta_o = avg.theta_o ./ sum(avg.Regional_o,2) ;
    cca.RPL.alpha_o = cca.alpha_o ./ (cca.delta_o + cca.theta_o + cca.alpha_o + cca.beta_o + cca.gamma_o) ;
    avg.RPL.alpha_o = avg.alpha_o ./ sum(avg.Regional_o,2) ;
    cca.RPL.beta_o = cca.beta_o ./ (cca.delta_o + cca.theta_o + cca.alpha_o + cca.beta_o + cca.gamma_o) ;
    avg.RPL.beta_o = avg.beta_o ./ sum(avg.Regional_o,2) ;
    cca.RPL.gamma_o = cca.gamma_o ./ (cca.delta_o + cca.theta_o + cca.alpha_o + cca.beta_o + cca.gamma_o) ;
    avg.RPL.gamma_o = avg.gamma_o ./ sum(avg.Regional_o,2) ;     
    
    cca.RPL.Regional_o = [cca.RPL.delta_o cca.RPL.theta_o cca.RPL.alpha_o cca.RPL.beta_o cca.RPL.gamma_o]; 
    avg.RPL.Regional_o = [avg.RPL.delta_o avg.RPL.theta_o avg.RPL.alpha_o avg.RPL.beta_o avg.RPL.gamma_o];
    
    cca.z.Regional_o = [zscore(cca.delta_o) zscore(cca.theta_o) zscore(cca.alpha_o) zscore(cca.beta_o) zscore(cca.gamma_o)];
    avg.z.Regional_o = [zscore(avg.delta_o) zscore(avg.theta_o) zscore(avg.alpha_o) zscore(avg.beta_o) zscore(avg.gamma_o)];   
    
    
    
    
    % Regional Vertical Feature Extraction
    [cca.delta_f1,avg.delta_f1] = Power_spectrum(channel_f1, [0.5 4], cnt.fs);
    [cca.theta_f1,avg.theta_f1] = Power_spectrum(channel_f1, [4 8], cnt.fs);
    [cca.alpha_f1,avg.alpha_f1] = Power_spectrum(channel_f1, [8 13], cnt.fs);
    [cca.beta_f1,avg.beta_f1] = Power_spectrum(channel_f1, [13 30], cnt.fs);
    [cca.gamma_f1,avg.gamma_f1] = Power_spectrum(channel_f1, [30 40], cnt.fs);
    avg.delta_f1 = avg.delta_f1'; avg.theta_f1=avg.theta_f1'; avg.alpha_f1=avg.alpha_f1'; avg.beta_f1 = avg.beta_f1'; avg.gamma_f1 = avg.gamma_f1';
    
    cca.Regional_f1 = [cca.delta_f1 cca.theta_f1 cca.alpha_f1 cca.beta_f1 cca.gamma_f1];
    avg.Regional_f1 = [avg.delta_f1 avg.theta_f1 avg.alpha_f1 avg.beta_f1 avg.gamma_f1];
    
    cca.RPL.delta_f1 = cca.delta_f1 ./ (cca.delta_f1 + cca.theta_f1 + cca.alpha_f1 + cca.beta_f1 + cca.gamma_f1) ;
    avg.RPL.delta_f1 = avg.delta_f1 ./ sum(avg.Regional_f1,2) ;
    cca.RPL.theta_f1 = cca.theta_f1 ./ (cca.delta_f1 + cca.theta_f1 + cca.alpha_f1 + cca.beta_f1 + cca.gamma_f1) ;
    avg.RPL.theta_f1 = avg.theta_f1 ./ sum(avg.Regional_f1,2) ;
    cca.RPL.alpha_f1 = cca.alpha_f1 ./ (cca.delta_f1 + cca.theta_f1 + cca.alpha_f1 + cca.beta_f1 + cca.gamma_f1) ;
    avg.RPL.alpha_f1 = avg.alpha_f1 ./ sum(avg.Regional_f1,2) ;
    cca.RPL.beta_f1 = cca.beta_f1 ./ (cca.delta_f1 + cca.theta_f1 + cca.alpha_f1 + cca.beta_f1 + cca.gamma_f1) ;
    avg.RPL.beta_f1 = avg.beta_f1 ./ sum(avg.Regional_f1,2) ;
    cca.RPL.gamma_f1 = cca.gamma_f1 ./ (cca.delta_f1 + cca.theta_f1 + cca.alpha_f1 + cca.beta_f1 + cca.gamma_f1) ;
    avg.RPL.gamma_f1 = avg.gamma_f1 ./ sum(avg.Regional_f1,2) ;    
    
    cca.RPL.Regional_f1 = [cca.RPL.delta_f1 cca.RPL.theta_f1 cca.RPL.alpha_f1 cca.RPL.beta_f1 cca.RPL.gamma_f1]; 
    avg.RPL.Regional_f1 = [avg.RPL.delta_f1 avg.RPL.theta_f1 avg.RPL.alpha_f1 avg.RPL.beta_f1 avg.RPL.gamma_f1];
    
    cca.z.Regional_f1 = [zscore(cca.delta_f1) zscore(cca.theta_f1) zscore(cca.alpha_f1) zscore(cca.beta_f1) zscore(cca.gamma_f1)];
    avg.z.Regional_f1 = [zscore(avg.delta_f1) zscore(avg.theta_f1) zscore(avg.alpha_f1) zscore(avg.beta_f1) zscore(avg.gamma_f1)];   
    
    
    [cca.delta_f2,avg.delta_f2] = Power_spectrum(channel_f2, [0.5 4], cnt.fs);
    [cca.theta_f2,avg.theta_f2] = Power_spectrum(channel_f2, [4 8], cnt.fs);
    [cca.alpha_f2,avg.alpha_f2] = Power_spectrum(channel_f2, [8 13], cnt.fs);
    [cca.beta_f2,avg.beta_f2] = Power_spectrum(channel_f2, [13 30], cnt.fs);
    [cca.gamma_f2,avg.gamma_f2] = Power_spectrum(channel_f2, [30 40], cnt.fs);
    avg.delta_f2 = avg.delta_f2'; avg.theta_f2=avg.theta_f2'; avg.alpha_f2=avg.alpha_f2'; avg.beta_f2 = avg.beta_f2'; avg.gamma_f2 = avg.gamma_f2';
    
    cca.Regional_f2 = [cca.delta_f2 cca.theta_f2 cca.alpha_f2 cca.beta_f2 cca.gamma_f2];
    avg.Regional_f2 = [avg.delta_f2 avg.theta_f2 avg.alpha_f2 avg.beta_f2 avg.gamma_f2];
    
    cca.RPL.delta_f2 = cca.delta_f2 ./ (cca.delta_f2 + cca.theta_f2 + cca.alpha_f2 + cca.beta_f2 + cca.gamma_f2) ;
    avg.RPL.delta_f2 = avg.delta_f2 ./ sum(avg.Regional_f2,2) ;
    cca.RPL.theta_f2 = cca.theta_f2 ./ (cca.delta_f2 + cca.theta_f2 + cca.alpha_f2 + cca.beta_f2 + cca.gamma_f2) ;
    avg.RPL.theta_f2 = avg.theta_f2 ./ sum(avg.Regional_f2,2) ;
    cca.RPL.alpha_f2 = cca.alpha_f2 ./ (cca.delta_f2 + cca.theta_f2 + cca.alpha_f2 + cca.beta_f2 + cca.gamma_f2) ;
    avg.RPL.alpha_f2 = avg.alpha_f2 ./ sum(avg.Regional_f2,2) ;
    cca.RPL.beta_f2 = cca.beta_f2 ./ (cca.delta_f2 + cca.theta_f2 + cca.alpha_f2 + cca.beta_f2 + cca.gamma_f2) ;
    avg.RPL.beta_f2 = avg.beta_f2 ./ sum(avg.Regional_f2,2) ;
    cca.RPL.gamma_f2 = cca.gamma_f2 ./ (cca.delta_f2 + cca.theta_f2 + cca.alpha_f2 + cca.beta_f2 + cca.gamma_f2) ;
    avg.RPL.gamma_f2 = avg.gamma_f2 ./ sum(avg.Regional_f2,2) ;    
    
    cca.RPL.Regional_f2 = [cca.RPL.delta_f2 cca.RPL.theta_f2 cca.RPL.alpha_f2 cca.RPL.beta_f2 cca.RPL.gamma_f2]; 
    avg.RPL.Regional_f2 = [avg.RPL.delta_f2 avg.RPL.theta_f2 avg.RPL.alpha_f2 avg.RPL.beta_f2 avg.RPL.gamma_f2];
    
    cca.z.Regional_f2 = [zscore(cca.delta_f2) zscore(cca.theta_f2) zscore(cca.alpha_f2) zscore(cca.beta_f2) zscore(cca.gamma_f2)];
    avg.z.Regional_f2 = [zscore(avg.delta_f2) zscore(avg.theta_f2) zscore(avg.alpha_f2) zscore(avg.beta_f2) zscore(avg.gamma_f2)];       
    
    
    [cca.delta_f3,avg.delta_f3] = Power_spectrum(channel_f3, [0.5 4], cnt.fs);
    [cca.theta_f3,avg.theta_f3] = Power_spectrum(channel_f3, [4 8], cnt.fs);
    [cca.alpha_f3,avg.alpha_f3] = Power_spectrum(channel_f3, [8 13], cnt.fs);
    [cca.beta_f3,avg.beta_f3] = Power_spectrum(channel_f3, [13 30], cnt.fs);
    [cca.gamma_f3,avg.gamma_f3] = Power_spectrum(channel_f3, [30 40], cnt.fs);
    avg.delta_f3 = avg.delta_f3'; avg.theta_f3=avg.theta_f3'; avg.alpha_f3=avg.alpha_f3'; avg.beta_f3 = avg.beta_f3'; avg.gamma_f3 = avg.gamma_f3';
    
    cca.Regional_f3 = [cca.delta_f3 cca.theta_f3 cca.alpha_f3 cca.beta_f3 cca.gamma_f3];
    avg.Regional_f3 = [avg.delta_f3 avg.theta_f3 avg.alpha_f3 avg.beta_f3 avg.gamma_f3];
    
    cca.RPL.delta_f3 = cca.delta_f3 ./ (cca.delta_f3 + cca.theta_f3 + cca.alpha_f3 + cca.beta_f3 + cca.gamma_f3) ;
    avg.RPL.delta_f3 = avg.delta_f3 ./ sum(avg.Regional_f3,2) ;
    cca.RPL.theta_f3 = cca.theta_f3 ./ (cca.delta_f3 + cca.theta_f3 + cca.alpha_f3 + cca.beta_f3 + cca.gamma_f3) ;
    avg.RPL.theta_f3 = avg.theta_f3 ./ sum(avg.Regional_f3,2) ;
    cca.RPL.alpha_f3 = cca.alpha_f3 ./ (cca.delta_f3 + cca.theta_f3 + cca.alpha_f3 + cca.beta_f3 + cca.gamma_f3) ;
    avg.RPL.alpha_f3 = avg.alpha_f3 ./ sum(avg.Regional_f3,2) ;
    cca.RPL.beta_f3 = cca.beta_f3 ./ (cca.delta_f3 + cca.theta_f3 + cca.alpha_f3 + cca.beta_f3 + cca.gamma_f3) ;
    avg.RPL.beta_f3 = avg.beta_f3 ./ sum(avg.Regional_f3,2) ;
    cca.RPL.gamma_f3 = cca.gamma_f3 ./ (cca.delta_f3 + cca.theta_f3 + cca.alpha_f3 + cca.beta_f3 + cca.gamma_f3) ;
    avg.RPL.gamma_f3 = avg.gamma_f3 ./ sum(avg.Regional_f3,2) ;    
    
    cca.RPL.Regional_f3 = [cca.RPL.delta_f3 cca.RPL.theta_f3 cca.RPL.alpha_f3 cca.RPL.beta_f3 cca.RPL.gamma_f3]; 
    avg.RPL.Regional_f3 = [avg.RPL.delta_f3 avg.RPL.theta_f3 avg.RPL.alpha_f3 avg.RPL.beta_f3 avg.RPL.gamma_f3];
    
    cca.z.Regional_f3 = [zscore(cca.delta_f3) zscore(cca.theta_f3) zscore(cca.alpha_f3) zscore(cca.beta_f3) zscore(cca.gamma_f3)];
    avg.z.Regional_f3 = [zscore(avg.delta_f3) zscore(avg.theta_f3) zscore(avg.alpha_f3) zscore(avg.beta_f3) zscore(avg.gamma_f3)];           
    
    
    [cca.delta_c1,avg.delta_c1] = Power_spectrum(channel_c1, [0.5 4], cnt.fs);
    [cca.theta_c1,avg.theta_c1] = Power_spectrum(channel_c1, [4 8], cnt.fs);
    [cca.alpha_c1,avg.alpha_c1] = Power_spectrum(channel_c1, [8 13], cnt.fs);
    [cca.beta_c1,avg.beta_c1] = Power_spectrum(channel_c1, [13 30], cnt.fs);
    [cca.gamma_c1,avg.gamma_c1] = Power_spectrum(channel_c1, [30 40], cnt.fs);
    avg.delta_c1 = avg.delta_c1'; avg.theta_c1=avg.theta_c1'; avg.alpha_c1=avg.alpha_c1'; avg.beta_c1 = avg.beta_c1'; avg.gamma_c1 = avg.gamma_c1';
    
    cca.Regional_c1 = [cca.delta_c1 cca.theta_c1 cca.alpha_c1 cca.beta_c1 cca.gamma_c1];
    avg.Regional_c1 = [avg.delta_c1 avg.theta_c1 avg.alpha_c1 avg.beta_c1 avg.gamma_c1];
    
    cca.RPL.delta_c1 = cca.delta_c1 ./ (cca.delta_c1 + cca.theta_c1 + cca.alpha_c1 + cca.beta_c1 + cca.gamma_c1) ;
    avg.RPL.delta_c1 = avg.delta_c1 ./ sum(avg.Regional_c1,2) ;
    cca.RPL.theta_c1 = cca.theta_c1 ./ (cca.delta_c1 + cca.theta_c1 + cca.alpha_c1 + cca.beta_c1 + cca.gamma_c1) ;
    avg.RPL.theta_c1 = avg.theta_c1 ./ sum(avg.Regional_c1,2) ;
    cca.RPL.alpha_c1 = cca.alpha_c1 ./ (cca.delta_c1 + cca.theta_c1 + cca.alpha_c1 + cca.beta_c1 + cca.gamma_c1) ;
    avg.RPL.alpha_c1 = avg.alpha_c1 ./ sum(avg.Regional_c1,2) ;
    cca.RPL.beta_c1 = cca.beta_c1 ./ (cca.delta_c1 + cca.theta_c1 + cca.alpha_c1 + cca.beta_c1 + cca.gamma_c1) ;
    avg.RPL.beta_c1 = avg.beta_c1 ./ sum(avg.Regional_c1,2) ;
    cca.RPL.gamma_c1 = cca.gamma_c1 ./ (cca.delta_c1 + cca.theta_c1 + cca.alpha_c1 + cca.beta_c1 + cca.gamma_c1) ;
    avg.RPL.gamma_c1 = avg.gamma_c1 ./ sum(avg.Regional_c1,2) ;     
    
    cca.RPL.Regional_c1 = [cca.RPL.delta_c1 cca.RPL.theta_c1 cca.RPL.alpha_c1 cca.RPL.beta_c1 cca.RPL.gamma_c1]; 
    avg.RPL.Regional_c1 = [avg.RPL.delta_c1 avg.RPL.theta_c1 avg.RPL.alpha_c1 avg.RPL.beta_c1 avg.RPL.gamma_c1];
    
    cca.z.Regional_c1 = [zscore(cca.delta_c1) zscore(cca.theta_c1) zscore(cca.alpha_c1) zscore(cca.beta_c1) zscore(cca.gamma_c1)];
    avg.z.Regional_c1 = [zscore(avg.delta_c1) zscore(avg.theta_c1) zscore(avg.alpha_c1) zscore(avg.beta_c1) zscore(avg.gamma_c1)];               
    
    [cca.delta_c2,avg.delta_c2] = Power_spectrum(channel_c2, [0.5 4], cnt.fs);
    [cca.theta_c2,avg.theta_c2] = Power_spectrum(channel_c2, [4 8], cnt.fs);
    [cca.alpha_c2,avg.alpha_c2] = Power_spectrum(channel_c2, [8 13], cnt.fs);
    [cca.beta_c2,avg.beta_c2] = Power_spectrum(channel_c2, [13 30], cnt.fs);
    [cca.gamma_c2,avg.gamma_c2] = Power_spectrum(channel_c2, [30 40], cnt.fs);
    avg.delta_c2 = avg.delta_c2'; avg.theta_c2=avg.theta_c2'; avg.alpha_c2=avg.alpha_c2'; avg.beta_c2 = avg.beta_c2'; avg.gamma_c2 = avg.gamma_c2';
    
    cca.Regional_c2 = [cca.delta_c2 cca.theta_c2 cca.alpha_c2 cca.beta_c2 cca.gamma_c2];
    avg.Regional_c2 = [avg.delta_c2 avg.theta_c2 avg.alpha_c2 avg.beta_c2 avg.gamma_c2];
    
    cca.RPL.delta_c2 = cca.delta_c2 ./ (cca.delta_c2 + cca.theta_c2 + cca.alpha_c2 + cca.beta_c2 + cca.gamma_c2) ;
    avg.RPL.delta_c2 = avg.delta_c2 ./ sum(avg.Regional_c2,2) ;
    cca.RPL.theta_c2 = cca.theta_c2 ./ (cca.delta_c2 + cca.theta_c2 + cca.alpha_c2 + cca.beta_c2 + cca.gamma_c2) ;
    avg.RPL.theta_c2 = avg.theta_c2 ./ sum(avg.Regional_c2,2) ;
    cca.RPL.alpha_c2 = cca.alpha_c2 ./ (cca.delta_c2 + cca.theta_c2 + cca.alpha_c2 + cca.beta_c2 + cca.gamma_c2) ;
    avg.RPL.alpha_c2 = avg.alpha_c2 ./ sum(avg.Regional_c2,2) ;
    cca.RPL.beta_c2 = cca.beta_c2 ./ (cca.delta_c2 + cca.theta_c2 + cca.alpha_c2 + cca.beta_c2 + cca.gamma_c2) ;
    avg.RPL.beta_c2 = avg.beta_c2 ./ sum(avg.Regional_c2,2) ;
    cca.RPL.gamma_c2 = cca.gamma_c2 ./ (cca.delta_c2 + cca.theta_c2 + cca.alpha_c2 + cca.beta_c2 + cca.gamma_c2) ;
    avg.RPL.gamma_c2 = avg.gamma_c2 ./ sum(avg.Regional_c2,2) ;     
    
    cca.RPL.Regional_c2 = [cca.RPL.delta_c2 cca.RPL.theta_c2 cca.RPL.alpha_c2 cca.RPL.beta_c2 cca.RPL.gamma_c2]; 
    avg.RPL.Regional_c2 = [avg.RPL.delta_c2 avg.RPL.theta_c2 avg.RPL.alpha_c2 avg.RPL.beta_c2 avg.RPL.gamma_c2];
    
    cca.z.Regional_c2 = [zscore(cca.delta_c2) zscore(cca.theta_c2) zscore(cca.alpha_c2) zscore(cca.beta_c2) zscore(cca.gamma_c2)];
    avg.z.Regional_c2 = [zscore(avg.delta_c2) zscore(avg.theta_c2) zscore(avg.alpha_c2) zscore(avg.beta_c2) zscore(avg.gamma_c2)];                   
    
    [cca.delta_c3,avg.delta_c3] = Power_spectrum(channel_c3, [0.5 4], cnt.fs);
    [cca.theta_c3,avg.theta_c3] = Power_spectrum(channel_c3, [4 8], cnt.fs);
    [cca.alpha_c3,avg.alpha_c3] = Power_spectrum(channel_c3, [8 13], cnt.fs);
    [cca.beta_c3,avg.beta_c3] = Power_spectrum(channel_c3, [13 30], cnt.fs);
    [cca.gamma_c3,avg.gamma_c3] = Power_spectrum(channel_c3, [30 40], cnt.fs);
    avg.delta_c3 = avg.delta_c3'; avg.theta_c3=avg.theta_c3'; avg.alpha_c3=avg.alpha_c3'; avg.beta_c3 = avg.beta_c3'; avg.gamma_c3 = avg.gamma_c3';
    
    cca.Regional_c3 = [cca.delta_c3 cca.theta_c3 cca.alpha_c3 cca.beta_c3 cca.gamma_c3];
    avg.Regional_c3 = [avg.delta_c3 avg.theta_c3 avg.alpha_c3 avg.beta_c3 avg.gamma_c3];
    
    cca.RPL.delta_c3 = cca.delta_c3 ./ (cca.delta_c3 + cca.theta_c3 + cca.alpha_c3 + cca.beta_c3 + cca.gamma_c3) ;
    avg.RPL.delta_c3 = avg.delta_c3 ./ sum(avg.Regional_c3,2) ;
    cca.RPL.theta_c3 = cca.theta_c3 ./ (cca.delta_c3 + cca.theta_c3 + cca.alpha_c3 + cca.beta_c3 + cca.gamma_c3) ;
    avg.RPL.theta_c3 = avg.theta_c3 ./ sum(avg.Regional_c3,2) ;
    cca.RPL.alpha_c3 = cca.alpha_c3 ./ (cca.delta_c3 + cca.theta_c3 + cca.alpha_c3 + cca.beta_c3 + cca.gamma_c3) ;
    avg.RPL.alpha_c3 = avg.alpha_c3 ./ sum(avg.Regional_c3,2) ;
    cca.RPL.beta_c3 = cca.beta_c3 ./ (cca.delta_c3 + cca.theta_c3 + cca.alpha_c3 + cca.beta_c3 + cca.gamma_c3) ;
    avg.RPL.beta_c3 = avg.beta_c3 ./ sum(avg.Regional_c3,2) ;
    cca.RPL.gamma_c3 = cca.gamma_c3 ./ (cca.delta_c3 + cca.theta_c3 + cca.alpha_c3 + cca.beta_c3 + cca.gamma_c3) ;
    avg.RPL.gamma_c3 = avg.gamma_c3 ./ sum(avg.Regional_c3,2) ;     
    
    cca.RPL.Regional_c3 = [cca.RPL.delta_c3 cca.RPL.theta_c3 cca.RPL.alpha_c3 cca.RPL.beta_c3 cca.RPL.gamma_c3]; 
    avg.RPL.Regional_c3 = [avg.RPL.delta_c3 avg.RPL.theta_c3 avg.RPL.alpha_c3 avg.RPL.beta_c3 avg.RPL.gamma_c3];
    
    cca.z.Regional_c3 = [zscore(cca.delta_c3) zscore(cca.theta_c3) zscore(cca.alpha_c3) zscore(cca.beta_c3) zscore(cca.gamma_c3)];
    avg.z.Regional_c3 = [zscore(avg.delta_c3) zscore(avg.theta_c3) zscore(avg.alpha_c3) zscore(avg.beta_c3) zscore(avg.gamma_c3)];                   
    
    [cca.delta_p1,avg.delta_p1] = Power_spectrum(channel_p1, [0.5 4], cnt.fs);
    [cca.theta_p1,avg.theta_p1] = Power_spectrum(channel_p1, [4 8], cnt.fs);
    [cca.alpha_p1,avg.alpha_p1] = Power_spectrum(channel_p1, [8 13], cnt.fs);
    [cca.beta_p1,avg.beta_p1] = Power_spectrum(channel_p1, [13 30], cnt.fs);
    [cca.gamma_p1,avg.gamma_p1] = Power_spectrum(channel_p1, [30 40], cnt.fs);
    avg.delta_p1 = avg.delta_p1'; avg.theta_p1=avg.theta_p1'; avg.alpha_p1=avg.alpha_p1'; avg.beta_p1 = avg.beta_p1'; avg.gamma_p1 = avg.gamma_p1';
    
    cca.Regional_p1 = [cca.delta_p1 cca.theta_p1 cca.alpha_p1 cca.beta_p1 cca.gamma_p1];
    avg.Regional_p1 = [avg.delta_p1 avg.theta_p1 avg.alpha_p1 avg.beta_p1 avg.gamma_p1];
    
    cca.RPL.delta_p1 = cca.delta_p1 ./ (cca.delta_p1 + cca.theta_p1 + cca.alpha_p1 + cca.beta_p1 + cca.gamma_p1) ;
    avg.RPL.delta_p1 = avg.delta_p1 ./ sum(avg.Regional_p1,2) ;
    cca.RPL.theta_p1 = cca.theta_p1 ./ (cca.delta_p1 + cca.theta_p1 + cca.alpha_p1 + cca.beta_p1 + cca.gamma_p1) ;
    avg.RPL.theta_p1 = avg.theta_p1 ./ sum(avg.Regional_p1,2) ;
    cca.RPL.alpha_p1 = cca.alpha_p1 ./ (cca.delta_p1 + cca.theta_p1 + cca.alpha_p1 + cca.beta_p1 + cca.gamma_p1) ;
    avg.RPL.alpha_p1 = avg.alpha_p1 ./ sum(avg.Regional_p1,2) ;
    cca.RPL.beta_p1 = cca.beta_p1 ./ (cca.delta_p1 + cca.theta_p1 + cca.alpha_p1 + cca.beta_p1 + cca.gamma_p1) ;
    avg.RPL.beta_p1 = avg.beta_p1 ./ sum(avg.Regional_p1,2) ;
    cca.RPL.gamma_p1 = cca.gamma_p1 ./ (cca.delta_p1 + cca.theta_p1 + cca.alpha_p1 + cca.beta_p1 + cca.gamma_p1) ;
    avg.RPL.gamma_p1 = avg.gamma_p1 ./ sum(avg.Regional_p1,2) ;     
    
    cca.RPL.Regional_p1 = [cca.RPL.delta_p1 cca.RPL.theta_p1 cca.RPL.alpha_p1 cca.RPL.beta_p1 cca.RPL.gamma_p1]; 
    avg.RPL.Regional_p1 = [avg.RPL.delta_p1 avg.RPL.theta_p1 avg.RPL.alpha_p1 avg.RPL.beta_p1 avg.RPL.gamma_p1];
    
    cca.z.Regional_p1 = [zscore(cca.delta_p1) zscore(cca.theta_p1) zscore(cca.alpha_p1) zscore(cca.beta_p1) zscore(cca.gamma_p1)];
    avg.z.Regional_p1 = [zscore(avg.delta_p1) zscore(avg.theta_p1) zscore(avg.alpha_p1) zscore(avg.beta_p1) zscore(avg.gamma_p1)];               
    
    [cca.delta_p2,avg.delta_p2] = Power_spectrum(channel_p2, [0.5 4], cnt.fs);
    [cca.theta_p2,avg.theta_p2] = Power_spectrum(channel_p2, [4 8], cnt.fs);
    [cca.alpha_p2,avg.alpha_p2] = Power_spectrum(channel_p2, [8 13], cnt.fs);
    [cca.beta_p2,avg.beta_p2] = Power_spectrum(channel_p2, [13 30], cnt.fs);
    [cca.gamma_p2,avg.gamma_p2] = Power_spectrum(channel_p2, [30 40], cnt.fs);
    avg.delta_p2 = avg.delta_p2'; avg.theta_p2=avg.theta_p2'; avg.alpha_p2=avg.alpha_p2'; avg.beta_p2 = avg.beta_p2'; avg.gamma_p2 = avg.gamma_p2';
    
    cca.Regional_p2 = [cca.delta_p2 cca.theta_p2 cca.alpha_p2 cca.beta_p2 cca.gamma_p2];
    avg.Regional_p2 = [avg.delta_p2 avg.theta_p2 avg.alpha_p2 avg.beta_p2 avg.gamma_p2];
    
    cca.RPL.delta_p2 = cca.delta_p2 ./ (cca.delta_p2 + cca.theta_p2 + cca.alpha_p2 + cca.beta_p2 + cca.gamma_p2) ;
    avg.RPL.delta_p2 = avg.delta_p2 ./ sum(avg.Regional_p2,2) ;
    cca.RPL.theta_p2 = cca.theta_p2 ./ (cca.delta_p2 + cca.theta_p2 + cca.alpha_p2 + cca.beta_p2 + cca.gamma_p2) ;
    avg.RPL.theta_p2 = avg.theta_p2 ./ sum(avg.Regional_p2,2) ;
    cca.RPL.alpha_p2 = cca.alpha_p2 ./ (cca.delta_p2 + cca.theta_p2 + cca.alpha_p2 + cca.beta_p2 + cca.gamma_p2) ;
    avg.RPL.alpha_p2 = avg.alpha_p2 ./ sum(avg.Regional_p2,2) ;
    cca.RPL.beta_p2 = cca.beta_p2 ./ (cca.delta_p2 + cca.theta_p2 + cca.alpha_p2 + cca.beta_p2 + cca.gamma_p2) ;
    avg.RPL.beta_p2 = avg.beta_p2 ./ sum(avg.Regional_p2,2) ;
    cca.RPL.gamma_p2 = cca.gamma_p2 ./ (cca.delta_p2 + cca.theta_p2 + cca.alpha_p2 + cca.beta_p2 + cca.gamma_p2) ;
    avg.RPL.gamma_p2 = avg.gamma_p2 ./ sum(avg.Regional_p2,2) ;     
    
    cca.RPL.Regional_p2 = [cca.RPL.delta_p2 cca.RPL.theta_p2 cca.RPL.alpha_p2 cca.RPL.beta_p2 cca.RPL.gamma_p2]; 
    avg.RPL.Regional_p2 = [avg.RPL.delta_p2 avg.RPL.theta_p2 avg.RPL.alpha_p2 avg.RPL.beta_p2 avg.RPL.gamma_p2];
    
    cca.z.Regional_p2 = [zscore(cca.delta_p2) zscore(cca.theta_p2) zscore(cca.alpha_p2) zscore(cca.beta_p2) zscore(cca.gamma_p2)];
    avg.z.Regional_p2 = [zscore(avg.delta_p2) zscore(avg.theta_p2) zscore(avg.alpha_p2) zscore(avg.beta_p2) zscore(avg.gamma_p2)];               
    
    
    [cca.delta_p3,avg.delta_p3] = Power_spectrum(channel_p3, [0.5 4], cnt.fs);
    [cca.theta_p3,avg.theta_p3] = Power_spectrum(channel_p3, [4 8], cnt.fs);
    [cca.alpha_p3,avg.alpha_p3] = Power_spectrum(channel_p3, [8 13], cnt.fs);
    [cca.beta_p3,avg.beta_p3] = Power_spectrum(channel_p3, [13 30], cnt.fs);
    [cca.gamma_p3,avg.gamma_p3] = Power_spectrum(channel_p3, [30 40], cnt.fs);
    avg.delta_p3 = avg.delta_p3'; avg.theta_p3=avg.theta_p3'; avg.alpha_p3=avg.alpha_p3'; avg.beta_p3 = avg.beta_p3'; avg.gamma_p3 = avg.gamma_p3';
    
    cca.Regional_p3 = [cca.delta_p3 cca.theta_p3 cca.alpha_p3 cca.beta_p3 cca.gamma_p3];
    avg.Regional_p3 = [avg.delta_p3 avg.theta_p3 avg.alpha_p3 avg.beta_p3 avg.gamma_p3];
    
    cca.RPL.delta_p3 = cca.delta_p3 ./ (cca.delta_p3 + cca.theta_p3 + cca.alpha_p3 + cca.beta_p3 + cca.gamma_p3) ;
    avg.RPL.delta_p3 = avg.delta_p3 ./ sum(avg.Regional_p3,2) ;
    cca.RPL.theta_p3 = cca.theta_p3 ./ (cca.delta_p3 + cca.theta_p3 + cca.alpha_p3 + cca.beta_p3 + cca.gamma_p3) ;
    avg.RPL.theta_p3 = avg.theta_p3 ./ sum(avg.Regional_p3,2) ;
    cca.RPL.alpha_p3 = cca.alpha_p3 ./ (cca.delta_p3 + cca.theta_p3 + cca.alpha_p3 + cca.beta_p3 + cca.gamma_p3) ;
    avg.RPL.alpha_p3 = avg.alpha_p3 ./ sum(avg.Regional_p3,2) ;
    cca.RPL.beta_p3 = cca.beta_p3 ./ (cca.delta_p3 + cca.theta_p3 + cca.alpha_p3 + cca.beta_p3 + cca.gamma_p3) ;
    avg.RPL.beta_p3 = avg.beta_p3 ./ sum(avg.Regional_p3,2) ;
    cca.RPL.gamma_p3 = cca.gamma_p3 ./ (cca.delta_p3 + cca.theta_p3 + cca.alpha_p3 + cca.beta_p3 + cca.gamma_p3) ;
    avg.RPL.gamma_p3 = avg.gamma_p3 ./ sum(avg.Regional_p3,2) ;     
    
    cca.RPL.Regional_p3 = [cca.RPL.delta_p3 cca.RPL.theta_p3 cca.RPL.alpha_p3 cca.RPL.beta_p3 cca.RPL.gamma_p3]; 
    avg.RPL.Regional_p3 = [avg.RPL.delta_p3 avg.RPL.theta_p3 avg.RPL.alpha_p3 avg.RPL.beta_p3 avg.RPL.gamma_p3];
    
    cca.z.Regional_p3 = [zscore(cca.delta_p3) zscore(cca.theta_p3) zscore(cca.alpha_p3) zscore(cca.beta_p3) zscore(cca.gamma_p3)];
    avg.z.Regional_p3 = [zscore(avg.delta_p3) zscore(avg.theta_p3) zscore(avg.alpha_p3) zscore(avg.beta_p3) zscore(avg.gamma_p3)];               
    
    
    [cca.delta_t1,avg.delta_t1] = Power_spectrum(channel_t1, [0.5 4], cnt.fs);
    [cca.theta_t1,avg.theta_t1] = Power_spectrum(channel_t1, [4 8], cnt.fs);
    [cca.alpha_t1,avg.alpha_t1] = Power_spectrum(channel_t1, [8 13], cnt.fs);
    [cca.beta_t1,avg.beta_t1] = Power_spectrum(channel_t1, [13 30], cnt.fs);
    [cca.gamma_t1,avg.gamma_t1] = Power_spectrum(channel_t1, [30 40], cnt.fs);
    avg.delta_t1 = avg.delta_t1'; avg.theta_t1=avg.theta_t1'; avg.alpha_t1=avg.alpha_t1'; avg.beta_t1 = avg.beta_t1'; avg.gamma_t1 = avg.gamma_t1';
    
    cca.Regional_t1 = [cca.delta_t1 cca.theta_t1 cca.alpha_t1 cca.beta_t1 cca.gamma_t1];
    avg.Regional_t1 = [avg.delta_t1 avg.theta_t1 avg.alpha_t1 avg.beta_t1 avg.gamma_t1];
    
    cca.RPL.delta_t1 = cca.delta_t1 ./ (cca.delta_t1 + cca.theta_t1 + cca.alpha_t1 + cca.beta_t1 + cca.gamma_t1) ;
    avg.RPL.delta_t1 = avg.delta_t1 ./ sum(avg.Regional_t1,2) ;
    cca.RPL.theta_t1 = cca.theta_t1 ./ (cca.delta_t1 + cca.theta_t1 + cca.alpha_t1 + cca.beta_t1 + cca.gamma_t1) ;
    avg.RPL.theta_t1 = avg.theta_t1 ./ sum(avg.Regional_t1,2) ;
    cca.RPL.alpha_t1 = cca.alpha_t1 ./ (cca.delta_t1 + cca.theta_t1 + cca.alpha_t1 + cca.beta_t1 + cca.gamma_t1) ;
    avg.RPL.alpha_t1 = avg.alpha_t1 ./ sum(avg.Regional_t1,2) ;
    cca.RPL.beta_t1 = cca.beta_t1 ./ (cca.delta_t1 + cca.theta_t1 + cca.alpha_t1 + cca.beta_t1 + cca.gamma_t1) ;
    avg.RPL.beta_t1 = avg.beta_t1 ./ sum(avg.Regional_t1,2) ;
    cca.RPL.gamma_t1 = cca.gamma_t1 ./ (cca.delta_t1 + cca.theta_t1 + cca.alpha_t1 + cca.beta_t1 + cca.gamma_t1) ;
    avg.RPL.gamma_t1 = avg.gamma_t1 ./ sum(avg.Regional_t1,2) ;     

    cca.RPL.Regional_t1 = [cca.RPL.delta_t1 cca.RPL.theta_t1 cca.RPL.alpha_t1 cca.RPL.beta_t1 cca.RPL.gamma_t1]; 
    avg.RPL.Regional_t1 = [avg.RPL.delta_t1 avg.RPL.theta_t1 avg.RPL.alpha_t1 avg.RPL.beta_t1 avg.RPL.gamma_t1];
    
    cca.z.Regional_t1 = [zscore(cca.delta_t1) zscore(cca.theta_t1) zscore(cca.alpha_t1) zscore(cca.beta_t1) zscore(cca.gamma_t1)];
    avg.z.Regional_t1 = [zscore(avg.delta_t1) zscore(avg.theta_t1) zscore(avg.alpha_t1) zscore(avg.beta_t1) zscore(avg.gamma_t1)];                   
    
    
    [cca.delta_t2,avg.delta_t2] = Power_spectrum(channel_t2, [0.5 4], cnt.fs);
    [cca.theta_t2,avg.theta_t2] = Power_spectrum(channel_t2, [4 8], cnt.fs);
    [cca.alpha_t2,avg.alpha_t2] = Power_spectrum(channel_t2, [8 13], cnt.fs);
    [cca.beta_t2,avg.beta_t2] = Power_spectrum(channel_t2, [13 30], cnt.fs);
    [cca.gamma_t2,avg.gamma_t2] = Power_spectrum(channel_t2, [30 40], cnt.fs);
    avg.delta_t2 = avg.delta_t2'; avg.theta_t2=avg.theta_t2'; avg.alpha_t2=avg.alpha_t2'; avg.beta_t2 = avg.beta_t2'; avg.gamma_t2 = avg.gamma_t2';
    
    cca.Regional_t2 = [cca.delta_t2 cca.theta_t2 cca.alpha_t2 cca.beta_t2 cca.gamma_t2];
    avg.Regional_t2 = [avg.delta_t2 avg.theta_t2 avg.alpha_t2 avg.beta_t2 avg.gamma_t2];
    
    cca.RPL.delta_t2 = cca.delta_t2 ./ (cca.delta_t2 + cca.theta_t2 + cca.alpha_t2 + cca.beta_t2 + cca.gamma_t2) ;
    avg.RPL.delta_t2 = avg.delta_t2 ./ sum(avg.Regional_t2,2) ;
    cca.RPL.theta_t2 = cca.theta_t2 ./ (cca.delta_t2 + cca.theta_t2 + cca.alpha_t2 + cca.beta_t2 + cca.gamma_t2) ;
    avg.RPL.theta_t2 = avg.theta_t2 ./ sum(avg.Regional_t2,2) ;
    cca.RPL.alpha_t2 = cca.alpha_t2 ./ (cca.delta_t2 + cca.theta_t2 + cca.alpha_t2 + cca.beta_t2 + cca.gamma_t2) ;
    avg.RPL.alpha_t2 = avg.alpha_t2 ./ sum(avg.Regional_t2,2) ;
    cca.RPL.beta_t2 = cca.beta_t2 ./ (cca.delta_t2 + cca.theta_t2 + cca.alpha_t2 + cca.beta_t2 + cca.gamma_t2) ;
    avg.RPL.beta_t2 = avg.beta_t2 ./ sum(avg.Regional_t2,2) ;
    cca.RPL.gamma_t2 = cca.gamma_t2 ./ (cca.delta_t2 + cca.theta_t2 + cca.alpha_t2 + cca.beta_t2 + cca.gamma_t2) ;
    avg.RPL.gamma_t2 = avg.gamma_t2 ./ sum(avg.Regional_t2,2) ;     
    
    cca.RPL.Regional_t2 = [cca.RPL.delta_t2 cca.RPL.theta_t2 cca.RPL.alpha_t2 cca.RPL.beta_t2 cca.RPL.gamma_t2]; 
    avg.RPL.Regional_t2 = [avg.RPL.delta_t2 avg.RPL.theta_t2 avg.RPL.alpha_t2 avg.RPL.beta_t2 avg.RPL.gamma_t2];
    
    cca.z.Regional_t2 = [zscore(cca.delta_t2) zscore(cca.theta_t2) zscore(cca.alpha_t2) zscore(cca.beta_t2) zscore(cca.gamma_t2)];
    avg.z.Regional_t2 = [zscore(avg.delta_t2) zscore(avg.theta_t2) zscore(avg.alpha_t2) zscore(avg.beta_t2) zscore(avg.gamma_t2)];                   
    
    
    
    
    
    [cca.delta_o1,avg.delta_o1] = Power_spectrum(channel_o1, [0.5 4], cnt.fs);
    [cca.theta_o1,avg.theta_o1] = Power_spectrum(channel_o1, [4 8], cnt.fs);
    [cca.alpha_o1,avg.alpha_o1] = Power_spectrum(channel_o1, [8 13], cnt.fs);
    [cca.beta_o1,avg.beta_o1] = Power_spectrum(channel_o1, [13 30], cnt.fs);
    [cca.gamma_o1,avg.gamma_o1] = Power_spectrum(channel_o1, [30 40], cnt.fs);
    avg.delta_o1 = avg.delta_o1'; avg.theta_o1=avg.theta_o1'; avg.alpha_o1=avg.alpha_o1'; avg.beta_o1 = avg.beta_o1'; avg.gamma_o1 = avg.gamma_o1';
    
    cca.Regional_o1 = [cca.delta_o1 cca.theta_o1 cca.alpha_o1 cca.beta_o1 cca.gamma_o1];
    avg.Regional_o1 = [avg.delta_o1 avg.theta_o1 avg.alpha_o1 avg.beta_o1 avg.gamma_o1];

    cca.RPL.delta_o1 = cca.delta_o1 ./ (cca.delta_o1 + cca.theta_o1 + cca.alpha_o1 + cca.beta_o1 + cca.gamma_o1) ;
    avg.RPL.delta_o1 = avg.delta_o1 ./ sum(avg.Regional_o1,2) ;
    cca.RPL.theta_o1 = cca.theta_o1 ./ (cca.delta_o1 + cca.theta_o1 + cca.alpha_o1 + cca.beta_o1 + cca.gamma_o1) ;
    avg.RPL.theta_o1 = avg.theta_o1 ./ sum(avg.Regional_o1,2) ;
    cca.RPL.alpha_o1 = cca.alpha_o1 ./ (cca.delta_o1 + cca.theta_o1 + cca.alpha_o1 + cca.beta_o1 + cca.gamma_o1) ;
    avg.RPL.alpha_o1 = avg.alpha_o1 ./ sum(avg.Regional_o1,2) ;
    cca.RPL.beta_o1 = cca.beta_o1 ./ (cca.delta_o1 + cca.theta_o1 + cca.alpha_o1 + cca.beta_o1 + cca.gamma_o1) ;
    avg.RPL.beta_o1 = avg.beta_o1 ./ sum(avg.Regional_o1,2) ;
    cca.RPL.gamma_o1 = cca.gamma_o1 ./ (cca.delta_o1 + cca.theta_o1 + cca.alpha_o1 + cca.beta_o1 + cca.gamma_o1) ;
    avg.RPL.gamma_o1 = avg.gamma_o1 ./ sum(avg.Regional_o1,2) ;     

    cca.RPL.Regional_o1 = [cca.RPL.delta_o1 cca.RPL.theta_o1 cca.RPL.alpha_o1 cca.RPL.beta_o1 cca.RPL.gamma_o1]; 
    avg.RPL.Regional_o1 = [avg.RPL.delta_o1 avg.RPL.theta_o1 avg.RPL.alpha_o1 avg.RPL.beta_o1 avg.RPL.gamma_o1];
    
    cca.z.Regional_o1 = [zscore(cca.delta_o1) zscore(cca.theta_o1) zscore(cca.alpha_o1) zscore(cca.beta_o1) zscore(cca.gamma_o1)];
    avg.z.Regional_o1 = [zscore(avg.delta_o1) zscore(avg.theta_o1) zscore(avg.alpha_o1) zscore(avg.beta_o1) zscore(avg.gamma_o1)];                   

    
    
    
    [cca.delta_o2,avg.delta_o2] = Power_spectrum(channel_o2, [0.5 4], cnt.fs);
    [cca.theta_o2,avg.theta_o2] = Power_spectrum(channel_o2, [4 8], cnt.fs);
    [cca.alpha_o2,avg.alpha_o2] = Power_spectrum(channel_o2, [8 13], cnt.fs);
    [cca.beta_o2,avg.beta_o2] = Power_spectrum(channel_o2, [13 30], cnt.fs);
    [cca.gamma_o2,avg.gamma_o2] = Power_spectrum(channel_o2, [30 40], cnt.fs);
    avg.delta_o2 = avg.delta_o2'; avg.theta_o2=avg.theta_o2'; avg.alpha_o2=avg.alpha_o2'; avg.beta_o2 = avg.beta_o2'; avg.gamma_o2 = avg.gamma_o2';
    
    cca.Regional_o2 = [cca.delta_o2 cca.theta_o2 cca.alpha_o2 cca.beta_o2 cca.gamma_o2];
    avg.Regional_o2 = [avg.delta_o2 avg.theta_o2 avg.alpha_o2 avg.beta_o2 avg.gamma_o2];
    
    cca.RPL.delta_o2 = cca.delta_o2 ./ (cca.delta_o2 + cca.theta_o2 + cca.alpha_o2 + cca.beta_o2 + cca.gamma_o2) ;
    avg.RPL.delta_o2 = avg.delta_o2 ./ sum(avg.Regional_o2,2) ;
    cca.RPL.theta_o2 = cca.theta_o2 ./ (cca.delta_o2 + cca.theta_o2 + cca.alpha_o2 + cca.beta_o2 + cca.gamma_o2) ;
    avg.RPL.theta_o2 = avg.theta_o2 ./ sum(avg.Regional_o2,2) ;
    cca.RPL.alpha_o2 = cca.alpha_o2 ./ (cca.delta_o2 + cca.theta_o2 + cca.alpha_o2 + cca.beta_o2 + cca.gamma_o2) ;
    avg.RPL.alpha_o2 = avg.alpha_o2 ./ sum(avg.Regional_o2,2) ;
    cca.RPL.beta_o2 = cca.beta_o2 ./ (cca.delta_o2 + cca.theta_o2 + cca.alpha_o2 + cca.beta_o2 + cca.gamma_o2) ;
    avg.RPL.beta_o2 = avg.beta_o2 ./ sum(avg.Regional_o2,2) ;
    cca.RPL.gamma_o2 = cca.gamma_o2 ./ (cca.delta_o2 + cca.theta_o2 + cca.alpha_o2 + cca.beta_o2 + cca.gamma_o2) ;
    avg.RPL.gamma_o2 = avg.gamma_o2 ./ sum(avg.Regional_o2,2) ;     

    cca.RPL.Regional_o2 = [cca.RPL.delta_o2 cca.RPL.theta_o2 cca.RPL.alpha_o2 cca.RPL.beta_o2 cca.RPL.gamma_o2]; 
    avg.RPL.Regional_o2 = [avg.RPL.delta_o2 avg.RPL.theta_o2 avg.RPL.alpha_o2 avg.RPL.beta_o2 avg.RPL.gamma_o2];
        
    cca.z.Regional_o2 = [zscore(cca.delta_o2) zscore(cca.theta_o2) zscore(cca.alpha_o2) zscore(cca.beta_o2) zscore(cca.gamma_o2)];
    avg.z.Regional_o2 = [zscore(avg.delta_o2) zscore(avg.theta_o2) zscore(avg.alpha_o2) zscore(avg.beta_o2) zscore(avg.gamma_o2)];                   
    
    
    
    [cca.delta_o3,avg.delta_o3] = Power_spectrum(channel_o3, [0.5 4], cnt.fs);
    [cca.theta_o3,avg.theta_o3] = Power_spectrum(channel_o3, [4 8], cnt.fs);
    [cca.alpha_o3,avg.alpha_o3] = Power_spectrum(channel_o3, [8 13], cnt.fs);
    [cca.beta_o3,avg.beta_o3] = Power_spectrum(channel_o3, [13 30], cnt.fs);
    [cca.gamma_o3,avg.gamma_o3] = Power_spectrum(channel_o3, [30 40], cnt.fs);
    avg.delta_o3 = avg.delta_o3'; avg.theta_o3=avg.theta_o3'; avg.alpha_o3=avg.alpha_o3'; avg.beta_o3 = avg.beta_o3'; avg.gamma_o3 = avg.gamma_o3';
    
    cca.Regional_o3 = [cca.delta_o3 cca.theta_o3 cca.alpha_o3 cca.beta_o3 cca.gamma_o3];
    avg.Regional_o3 = [avg.delta_o3 avg.theta_o3 avg.alpha_o3 avg.beta_o3 avg.gamma_o3];

    cca.RPL.delta_o3 = cca.delta_o3 ./ (cca.delta_o3 + cca.theta_o3 + cca.alpha_o3 + cca.beta_o3 + cca.gamma_o3) ;
    avg.RPL.delta_o3 = avg.delta_o3 ./ sum(avg.Regional_o3,2) ;
    cca.RPL.theta_o3 = cca.theta_o3 ./ (cca.delta_o3 + cca.theta_o3 + cca.alpha_o3 + cca.beta_o3 + cca.gamma_o3) ;
    avg.RPL.theta_o3 = avg.theta_o3 ./ sum(avg.Regional_o3,2) ;
    cca.RPL.alpha_o3 = cca.alpha_o3 ./ (cca.delta_o3 + cca.theta_o3 + cca.alpha_o3 + cca.beta_o3 + cca.gamma_o3) ;
    avg.RPL.alpha_o3 = avg.alpha_o3 ./ sum(avg.Regional_o3,2) ;
    cca.RPL.beta_o3 = cca.beta_o3 ./ (cca.delta_o3 + cca.theta_o3 + cca.alpha_o3 + cca.beta_o3 + cca.gamma_o3) ;
    avg.RPL.beta_o3 = avg.beta_o3 ./ sum(avg.Regional_o3,2) ;
    cca.RPL.gamma_o3 = cca.gamma_o3 ./ (cca.delta_o3 + cca.theta_o3 + cca.alpha_o3 + cca.beta_o3 + cca.gamma_o3) ;
    avg.RPL.gamma_o3 = avg.gamma_o3 ./ sum(avg.Regional_o3,2) ;     

    cca.RPL.Regional_o3 = [cca.RPL.delta_o3 cca.RPL.theta_o3 cca.RPL.alpha_o3 cca.RPL.beta_o3 cca.RPL.gamma_o3]; 
    avg.RPL.Regional_o3 = [avg.RPL.delta_o3 avg.RPL.theta_o3 avg.RPL.alpha_o3 avg.RPL.beta_o3 avg.RPL.gamma_o3];
        
    cca.z.Regional_o3 = [zscore(cca.delta_o3) zscore(cca.theta_o3) zscore(cca.alpha_o3) zscore(cca.beta_o3) zscore(cca.gamma_o3)];
    avg.z.Regional_o3 = [zscore(avg.delta_o3) zscore(avg.theta_o3) zscore(avg.alpha_o3) zscore(avg.beta_o3) zscore(avg.gamma_o3)];                   
    
    
    %% Statistical Analysis
%         lowIdx = find(kss==min(kss)):1:find(kss==min(kss))+200;
%         lowIdx = lowIdx';
%         highIdx = find(kss==max(kss))-200:1:find(kss==max(kss));
%         highIdx = highIdx';
    
    %         최소+1, 최대-1
    lowIdx = min(kss)<kss & min(kss)+1>kss;
    highIdx = max(kss)>kss & max(kss)-1<kss;
    
    % PSD T-TEST 분석
    [h, p, ci, stats] = ttest2(avg.delta(lowIdx), avg.delta(highIdx));
    p_delta(s) = p;
    [h, p, ci, stats] = ttest2(avg.theta(lowIdx), avg.theta(highIdx));
    p_theta(s) = p;
    [h, p, ci, stats] = ttest2(avg.alpha(lowIdx), avg.alpha(highIdx));
    p_alpha(s) = p;
    [h, p, ci, stats] = ttest2(avg.beta(lowIdx), avg.beta(highIdx));
    p_beta(s) = p;
    [h, p, ci, stats] = ttest2(avg.gamma(lowIdx), avg.gamma(highIdx));
    p_gamma(s) = p;
    
    p_analysis(s,1:5) = [p_delta(s), p_theta(s), p_alpha(s), p_beta(s), p_gamma(s)];
    p_analysis_min(s) = find(p_analysis(s,1:5)==min(p_analysis(s,1:5)));
    
    Result.ttest.total(s,:) = p_analysis(s,1:5);
    
    % RPL T-TEST 분석
    [h, p, ci, stats] = ttest2(avg.RPL.delta(lowIdx), avg.RPL.delta(highIdx));
    p_delta(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.theta(lowIdx), avg.RPL.theta(highIdx));
    p_theta(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.alpha(lowIdx), avg.RPL.alpha(highIdx));
    p_alpha(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.beta(lowIdx), avg.RPL.beta(highIdx));
    p_beta(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.gamma(lowIdx), avg.RPL.gamma(highIdx));
    p_gamma(s) = p;
    
    p_analysis_RPL(s,1:5) = [p_delta(s), p_theta(s), p_alpha(s), p_beta(s), p_gamma(s)];
    p_analysis_RPL_min(s) = find(p_analysis_RPL(s,1:5)==min(p_analysis_RPL(s,1:5)));
    
    Result.ttest.total_RPL(s,:) = p_analysis_RPL(s,1:5);    
    
    
    % z-score T-TEST 분석
    [h, p, ci, stats] = ttest2(avg.z.PSD(lowIdx,1), avg.z.PSD(highIdx,1));
    p_delta(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.PSD(lowIdx,2), avg.z.PSD(lowIdx,2));
    p_theta(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.PSD(lowIdx,3), avg.z.PSD(lowIdx,3));
    p_alpha(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.PSD(lowIdx,4), avg.z.PSD(lowIdx,4));
    p_beta(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.PSD(lowIdx,5), avg.z.PSD(lowIdx,5));
    p_gamma(s) = p;
    
    p_analysis_z(s,1:5) = [p_delta(s), p_theta(s), p_alpha(s), p_beta(s), p_gamma(s)];
    p_analysis_z_min(s) = find(p_analysis_z(s,1:5)==min(p_analysis_z(s,1:5)));
    
    Result.ttest.total_z(s,:) = p_analysis_z(s,1:5);    
    
    
    
    % Regional PSD T-TEST 분석
    [h, p, ci, stats] = ttest2(avg.delta_f(lowIdx), avg.delta_f(highIdx));
    p_delta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.theta_f(lowIdx), avg.theta_f(highIdx));
    p_theta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.alpha_f(lowIdx), avg.alpha_f(highIdx));
    p_alpha_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.beta_f(lowIdx), avg.beta_f(highIdx));
    p_beta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.gamma_f(lowIdx), avg.gamma_f(highIdx));
    p_gamma_f(s) = p;
    
    p_analysis_f(s,1:5) = [p_delta_f(s), p_theta_f(s), p_alpha_f(s), p_beta_f(s), p_gamma_f(s)];
    p_analysis_f_min(s) = find(p_analysis_f(s,1:5)==min(p_analysis_f(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.delta_c(lowIdx), avg.delta_c(highIdx));
    p_delta_c(s) = p;
    [h, p, ci, stats] = ttest2(avg.theta_c(lowIdx), avg.theta_c(highIdx));
    p_theta_c(s) = p;
    [h, p, ci, stats] = ttest2(avg.alpha_c(lowIdx), avg.alpha_c(highIdx));
    p_alpha_c(s) = p;
    [h, p, ci, stats] = ttest2(avg.beta_c(lowIdx), avg.beta_c(highIdx));
    p_beta_c(s) = p;
    [h, p, ci, stats] = ttest2(avg.gamma_c(lowIdx), avg.gamma_c(highIdx));
    p_gamma_c(s) = p;
    
    p_analysis_c(s,1:5) = [p_delta_c(s), p_theta_c(s), p_alpha_c(s), p_beta_c(s), p_gamma_c(s)];
    p_analysis_c_min(s) = find(p_analysis_c(s,1:5)==min(p_analysis_c(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.delta_p(lowIdx), avg.delta_p(highIdx));
    p_delta_p(s) = p;
    [h, p, ci, stats] = ttest2(avg.theta_p(lowIdx), avg.theta_p(highIdx));
    p_theta_p(s) = p;
    [h, p, ci, stats] = ttest2(avg.alpha_p(lowIdx), avg.alpha_p(highIdx));
    p_alpha_p(s) = p;
    [h, p, ci, stats] = ttest2(avg.beta_p(lowIdx), avg.beta_p(highIdx));
    p_beta_p(s) = p;
    [h, p, ci, stats] = ttest2(avg.gamma_p(lowIdx), avg.gamma_p(highIdx));
    p_gamma_p(s) = p;
    
    p_analysis_p(s,1:5) = [p_delta_p(s), p_theta_p(s), p_alpha_p(s), p_beta_p(s), p_gamma_p(s)];
    p_analysis_p_min(s) = find(p_analysis_p(s,1:5)==min(p_analysis_p(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.delta_t(lowIdx), avg.delta_t(highIdx));
    p_delta_t(s) = p;
    [h, p, ci, stats] = ttest2(avg.theta_t(lowIdx), avg.theta_t(highIdx));
    p_theta_t(s) = p;
    [h, p, ci, stats] = ttest2(avg.alpha_t(lowIdx), avg.alpha_t(highIdx));
    p_alpha_t(s) = p;
    [h, p, ci, stats] = ttest2(avg.beta_t(lowIdx), avg.beta_t(highIdx));
    p_beta_t(s) = p;
    [h, p, ci, stats] = ttest2(avg.gamma_t(lowIdx), avg.gamma_t(highIdx));
    p_gamma_t(s) = p;
    
    p_analysis_t(s,1:5) = [p_delta_t(s), p_theta_t(s), p_alpha_t(s), p_beta_t(s), p_gamma_t(s)];
    p_analysis_t_min(s) = find(p_analysis_t(s,1:5)==min(p_analysis_t(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.delta_o(lowIdx), avg.delta_o(highIdx));
    p_delta_o(s) = p;
    [h, p, ci, stats] = ttest2(avg.theta_o(lowIdx), avg.theta_o(highIdx));
    p_theta_o(s) = p;
    [h, p, ci, stats] = ttest2(avg.alpha_o(lowIdx), avg.alpha_o(highIdx));
    p_alpha_o(s) = p;
    [h, p, ci, stats] = ttest2(avg.beta_o(lowIdx), avg.beta_o(highIdx));
    p_beta_o(s) = p;
    [h, p, ci, stats] = ttest2(avg.gamma_o(lowIdx), avg.gamma_o(highIdx));
    p_gamma_o(s) = p;
    
    p_analysis_o(s,1:5) = [p_delta_o(s), p_theta_o(s), p_alpha_o(s), p_beta_o(s), p_gamma_o(s)];
    p_analysis_o_min(s) = find(p_analysis_o(s,1:5)==min(p_analysis_o(s,1:5)));
    
    Result.ttest.regional(s,:) = [p_analysis_f(s,:) p_analysis_c(s,:) p_analysis_p(s,:) p_analysis_t(s,:) p_analysis_o(s,:)];

    
    
    [h, p, ci, stats] = ttest2(avg.RPL.delta_f(lowIdx), avg.RPL.delta_f(highIdx));
    p_delta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.theta_f(lowIdx), avg.RPL.theta_f(highIdx));
    p_theta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.alpha_f(lowIdx), avg.RPL.alpha_f(highIdx));
    p_alpha_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.beta_f(lowIdx), avg.RPL.beta_f(highIdx));
    p_beta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.gamma_f(lowIdx), avg.RPL.gamma_f(highIdx));
    p_gamma_f(s) = p;
    
    p_analysis_RPL_f(s,1:5) = [p_delta_f(s), p_theta_f(s), p_alpha_f(s), p_beta_f(s), p_gamma_f(s)];
    p_analysis_RPL_f_min(s) = find(p_analysis_RPL_f(s,1:5)==min(p_analysis_RPL_f(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.RPL.delta_c(lowIdx), avg.RPL.delta_c(highIdx));
    p_delta_c(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.theta_c(lowIdx), avg.RPL.theta_c(highIdx));
    p_theta_c(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.alpha_c(lowIdx), avg.RPL.alpha_c(highIdx));
    p_alpha_c(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.beta_c(lowIdx), avg.RPL.beta_c(highIdx));
    p_beta_c(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.gamma_c(lowIdx), avg.RPL.gamma_c(highIdx));
    p_gamma_c(s) = p;
    
    p_analysis_RPL_c(s,1:5) = [p_delta_c(s), p_theta_c(s), p_alpha_c(s), p_beta_c(s), p_gamma_c(s)];
    p_analysis_RPL_c_min(s) = find(p_analysis_RPL_c(s,1:5)==min(p_analysis_RPL_c(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.RPL.delta_p(lowIdx), avg.RPL.delta_p(highIdx));
    p_delta_p(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.theta_p(lowIdx), avg.RPL.theta_p(highIdx));
    p_theta_p(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.alpha_p(lowIdx), avg.RPL.alpha_p(highIdx));
    p_alpha_p(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.beta_p(lowIdx), avg.RPL.beta_p(highIdx));
    p_beta_p(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.gamma_p(lowIdx), avg.RPL.gamma_p(highIdx));
    p_gamma_p(s) = p;
    
    p_analysis_RPL_p(s,1:5) = [p_delta_p(s), p_theta_p(s), p_alpha_p(s), p_beta_p(s), p_gamma_p(s)];
    p_analysis_RPL_p_min(s) = find(p_analysis_RPL_p(s,1:5)==min(p_analysis_RPL_p(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.RPL.delta_t(lowIdx), avg.RPL.delta_t(highIdx));
    p_delta_t(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.theta_t(lowIdx), avg.RPL.theta_t(highIdx));
    p_theta_t(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.alpha_t(lowIdx), avg.RPL.alpha_t(highIdx));
    p_alpha_t(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.beta_t(lowIdx), avg.RPL.beta_t(highIdx));
    p_beta_t(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.gamma_t(lowIdx), avg.RPL.gamma_t(highIdx));
    p_gamma_t(s) = p;
    
    p_analysis_RPL_t(s,1:5) = [p_delta_t(s), p_theta_t(s), p_alpha_t(s), p_beta_t(s), p_gamma_t(s)];
    p_analysis_RPL_t_min(s) = find(p_analysis_RPL_t(s,1:5)==min(p_analysis_RPL_t(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.RPL.delta_o(lowIdx), avg.RPL.delta_o(highIdx));
    p_delta_o(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.theta_o(lowIdx), avg.RPL.theta_o(highIdx));
    p_theta_o(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.alpha_o(lowIdx), avg.RPL.alpha_o(highIdx));
    p_alpha_o(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.beta_o(lowIdx), avg.RPL.beta_o(highIdx));
    p_beta_o(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.gamma_o(lowIdx), avg.RPL.gamma_o(highIdx));
    p_gamma_o(s) = p;
    
    p_analysis_RPL_o(s,1:5) = [p_delta_o(s), p_theta_o(s), p_alpha_o(s), p_beta_o(s), p_gamma_o(s)];
    p_analysis_RPL_o_min(s) = find(p_analysis_RPL_o(s,1:5)==min(p_analysis_RPL_o(s,1:5)));
    
    Result.ttest.regional_RPL(s,:) = [p_analysis_RPL_f(s,:) p_analysis_RPL_c(s,:) p_analysis_RPL_p(s,:) p_analysis_RPL_t(s,:) p_analysis_RPL_o(s,:)];

    
    
    [h, p, ci, stats] = ttest2(avg.z.Regional_f(lowIdx,1), avg.z.Regional_f(highIdx,1));
    p_delta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_f(lowIdx,2), avg.z.Regional_f(highIdx,2));
    p_theta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_f(lowIdx,3), avg.z.Regional_f(highIdx,3));
    p_alpha_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_f(lowIdx,4), avg.z.Regional_f(highIdx,4));
    p_beta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_f(lowIdx,5), avg.z.Regional_f(highIdx,5));
    p_gamma_f(s) = p;
    
    p_analysis_z_f(s,1:5) = [p_delta_f(s), p_theta_f(s), p_alpha_f(s), p_beta_f(s), p_gamma_f(s)];
    p_analysis_z_f_min(s) = find(p_analysis_z_f(s,1:5)==min(p_analysis_z_f(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.z.Regional_c(lowIdx,1), avg.z.Regional_c(highIdx,1));
    p_delta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_c(lowIdx,2), avg.z.Regional_c(highIdx,2));
    p_theta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_c(lowIdx,3), avg.z.Regional_c(highIdx,3));
    p_alpha_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_c(lowIdx,4), avg.z.Regional_c(highIdx,4));
    p_beta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_c(lowIdx,5), avg.z.Regional_c(highIdx,5));
    p_gamma_f(s) = p;
    
    p_analysis_z_c(s,1:5) = [p_delta_c(s), p_theta_c(s), p_alpha_c(s), p_beta_c(s), p_gamma_c(s)];
    p_analysis_z_c_min(s) = find(p_analysis_z_c(s,1:5)==min(p_analysis_z_c(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.z.Regional_p(lowIdx,1), avg.z.Regional_p(highIdx,1));
    p_delta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_p(lowIdx,2), avg.z.Regional_p(highIdx,2));
    p_theta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_p(lowIdx,3), avg.z.Regional_p(highIdx,3));
    p_alpha_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_p(lowIdx,4), avg.z.Regional_p(highIdx,4));
    p_beta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_p(lowIdx,5), avg.z.Regional_p(highIdx,5));
    p_gamma_f(s) = p;
    
    p_analysis_z_p(s,1:5) = [p_delta_p(s), p_theta_p(s), p_alpha_p(s), p_beta_p(s), p_gamma_p(s)];
    p_analysis_z_p_min(s) = find(p_analysis_z_p(s,1:5)==min(p_analysis_z_p(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.z.Regional_t(lowIdx,1), avg.z.Regional_t(highIdx,1));
    p_delta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_t(lowIdx,2), avg.z.Regional_t(highIdx,2));
    p_theta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_t(lowIdx,3), avg.z.Regional_t(highIdx,3));
    p_alpha_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_t(lowIdx,4), avg.z.Regional_t(highIdx,4));
    p_beta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_t(lowIdx,5), avg.z.Regional_t(highIdx,5));
    p_gamma_f(s) = p;
    
    p_analysis_z_t(s,1:5) = [p_delta_t(s), p_theta_t(s), p_alpha_t(s), p_beta_t(s), p_gamma_t(s)];
    p_analysis_z_t_min(s) = find(p_analysis_z_t(s,1:5)==min(p_analysis_z_t(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.z.Regional_o(lowIdx,1), avg.z.Regional_o(highIdx,1));
    p_delta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_o(lowIdx,2), avg.z.Regional_o(highIdx,2));
    p_theta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_o(lowIdx,3), avg.z.Regional_o(highIdx,3));
    p_alpha_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_o(lowIdx,4), avg.z.Regional_o(highIdx,4));
    p_beta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_o(lowIdx,5), avg.z.Regional_o(highIdx,5));
    p_gamma_f(s) = p;
    
    p_analysis_z_o(s,1:5) = [p_delta_o(s), p_theta_o(s), p_alpha_o(s), p_beta_o(s), p_gamma_o(s)];
    p_analysis_z_o_min(s) = find(p_analysis_z_o(s,1:5)==min(p_analysis_z_o(s,1:5)));
    
    Result.ttest.regional_z(s,:) = [p_analysis_f(s,:) p_analysis_c(s,:) p_analysis_p(s,:) p_analysis_t(s,:) p_analysis_o(s,:)];

    
    
    
    
    
    % Regional Regional T-TEST 분석
    [h, p, ci, stats] = ttest2(avg.delta_f1(lowIdx), avg.delta_f1(highIdx));
    p_delta_f1(s) = p;
    [h, p, ci, stats] = ttest2(avg.theta_f1(lowIdx), avg.theta_f1(highIdx));
    p_theta_f1(s) = p;
    [h, p, ci, stats] = ttest2(avg.alpha_f1(lowIdx), avg.alpha_f1(highIdx));
    p_alpha_f1(s) = p;
    [h, p, ci, stats] = ttest2(avg.beta_f1(lowIdx), avg.beta_f1(highIdx));
    p_beta_f1(s) = p;
    [h, p, ci, stats] = ttest2(avg.gamma_f1(lowIdx), avg.gamma_f1(highIdx));
    p_gamma_f1(s) = p;
    
    p_analysis_f1(s,1:5) = [p_delta_f1(s), p_theta_f1(s), p_alpha_f1(s), p_beta_f1(s), p_gamma_f1(s)];
    p_analysis_f1_min(s) = find(p_analysis_f1(s,1:5)==min(p_analysis_f1(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.delta_f2(lowIdx), avg.delta_f2(highIdx));
    p_delta_f2(s) = p;
    [h, p, ci, stats] = ttest2(avg.theta_f2(lowIdx), avg.theta_f2(highIdx));
    p_theta_f2(s) = p;
    [h, p, ci, stats] = ttest2(avg.alpha_f2(lowIdx), avg.alpha_f2(highIdx));
    p_alpha_f2(s) = p;
    [h, p, ci, stats] = ttest2(avg.beta_f2(lowIdx), avg.beta_f2(highIdx));
    p_beta_f2(s) = p;
    [h, p, ci, stats] = ttest2(avg.gamma_f2(lowIdx), avg.gamma_f2(highIdx));
    p_gamma_f2(s) = p;
    
    p_analysis_f2(s,1:5) = [p_delta_f2(s), p_theta_f2(s), p_alpha_f2(s), p_beta_f2(s), p_gamma_f2(s)];
    p_analysis_f2_min(s) = find(p_analysis_f2(s,1:5)==min(p_analysis_f2(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.delta_f3(lowIdx), avg.delta_f3(highIdx));
    p_delta_f3(s) = p;
    [h, p, ci, stats] = ttest2(avg.theta_f3(lowIdx), avg.theta_f3(highIdx));
    p_theta_f3(s) = p;
    [h, p, ci, stats] = ttest2(avg.alpha_f3(lowIdx), avg.alpha_f3(highIdx));
    p_alpha_f3(s) = p;
    [h, p, ci, stats] = ttest2(avg.beta_f3(lowIdx), avg.beta_f3(highIdx));
    p_beta_f3(s) = p;
    [h, p, ci, stats] = ttest2(avg.gamma_f3(lowIdx), avg.gamma_f3(highIdx));
    p_gamma_f3(s) = p;
    
    p_analysis_f3(s,1:5) = [p_delta_f3(s), p_theta_f3(s), p_alpha_f3(s), p_beta_f3(s), p_gamma_f3(s)];
    p_analysis_f3_min(s) = find(p_analysis_f3(s,1:5)==min(p_analysis_f3(s,1:5)));
    
    
    
    [h, p, ci, stats] = ttest2(avg.delta_c1(lowIdx), avg.delta_c1(highIdx));
    p_delta_c1(s) = p;
    [h, p, ci, stats] = ttest2(avg.theta_c1(lowIdx), avg.theta_c1(highIdx));
    p_theta_c1(s) = p;
    [h, p, ci, stats] = ttest2(avg.alpha_c1(lowIdx), avg.alpha_c1(highIdx));
    p_alpha_c1(s) = p;
    [h, p, ci, stats] = ttest2(avg.beta_c1(lowIdx), avg.beta_c1(highIdx));
    p_beta_c1(s) = p;
    [h, p, ci, stats] = ttest2(avg.gamma_c1(lowIdx), avg.gamma_c1(highIdx));
    p_gamma_c1(s) = p;
    
    p_analysis_c1(s,1:5) = [p_delta_c1(s), p_theta_c1(s), p_alpha_c1(s), p_beta_c1(s), p_gamma_c1(s)];
    p_analysis_c1_min(s) = find(p_analysis_c1(s,1:5)==min(p_analysis_c1(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.delta_c2(lowIdx), avg.delta_c2(highIdx));
    p_delta_c2(s) = p;
    [h, p, ci, stats] = ttest2(avg.theta_c2(lowIdx), avg.theta_c2(highIdx));
    p_theta_c2(s) = p;
    [h, p, ci, stats] = ttest2(avg.alpha_c2(lowIdx), avg.alpha_c2(highIdx));
    p_alpha_c2(s) = p;
    [h, p, ci, stats] = ttest2(avg.beta_c2(lowIdx), avg.beta_c2(highIdx));
    p_beta_c2(s) = p;
    [h, p, ci, stats] = ttest2(avg.gamma_c2(lowIdx), avg.gamma_c2(highIdx));
    p_gamma_c2(s) = p;
    
    p_analysis_c2(s,1:5) = [p_delta_c2(s), p_theta_c2(s), p_alpha_c2(s), p_beta_c2(s), p_gamma_c2(s)];
    p_analysis_c2_min(s) = find(p_analysis_c2(s,1:5)==min(p_analysis_c2(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.delta_c3(lowIdx), avg.delta_c3(highIdx));
    p_delta_c3(s) = p;
    [h, p, ci, stats] = ttest2(avg.theta_c3(lowIdx), avg.theta_c3(highIdx));
    p_theta_c3(s) = p;
    [h, p, ci, stats] = ttest2(avg.alpha_c3(lowIdx), avg.alpha_c3(highIdx));
    p_alpha_c3(s) = p;
    [h, p, ci, stats] = ttest2(avg.beta_c3(lowIdx), avg.beta_c3(highIdx));
    p_beta_c3(s) = p;
    [h, p, ci, stats] = ttest2(avg.gamma_c3(lowIdx), avg.gamma_c3(highIdx));
    p_gamma_c3(s) = p;
    
    p_analysis_c3(s,1:5) = [p_delta_c3(s), p_theta_c3(s), p_alpha_c3(s), p_beta_c3(s), p_gamma_c3(s)];
    p_analysis_c3_min(s) = find(p_analysis_c3(s,1:5)==min(p_analysis_c3(s,1:5)));
    
    

    
    
    [h, p, ci, stats] = ttest2(avg.delta_p1(lowIdx), avg.delta_p1(highIdx));
    p_delta_p1(s) = p;
    [h, p, ci, stats] = ttest2(avg.theta_p1(lowIdx), avg.theta_p1(highIdx));
    p_theta_p1(s) = p;
    [h, p, ci, stats] = ttest2(avg.alpha_p1(lowIdx), avg.alpha_p1(highIdx));
    p_alpha_p1(s) = p;
    [h, p, ci, stats] = ttest2(avg.beta_p1(lowIdx), avg.beta_p1(highIdx));
    p_beta_p1(s) = p;
    [h, p, ci, stats] = ttest2(avg.gamma_p1(lowIdx), avg.gamma_p1(highIdx));
    p_gamma_p1(s) = p;
    
    p_analysis_p1(s,1:5) = [p_delta_p1(s), p_theta_p1(s), p_alpha_p1(s), p_beta_p1(s), p_gamma_p1(s)];
    p_analysis_p1_min(s) = find(p_analysis_p1(s,1:5)==min(p_analysis_p1(s,1:5)));
    
    
    
    [h, p, ci, stats] = ttest2(avg.delta_p2(lowIdx), avg.delta_p2(highIdx));
    p_delta_p2(s) = p;
    [h, p, ci, stats] = ttest2(avg.theta_p2(lowIdx), avg.theta_p2(highIdx));
    p_theta_p2(s) = p;
    [h, p, ci, stats] = ttest2(avg.alpha_p2(lowIdx), avg.alpha_p2(highIdx));
    p_alpha_p2(s) = p;
    [h, p, ci, stats] = ttest2(avg.beta_p2(lowIdx), avg.beta_p2(highIdx));
    p_beta_p2(s) = p;
    [h, p, ci, stats] = ttest2(avg.gamma_p2(lowIdx), avg.gamma_p2(highIdx));
    p_gamma_p2(s) = p;
    
    p_analysis_p2(s,1:5) = [p_delta_p2(s), p_theta_p2(s), p_alpha_p2(s), p_beta_p2(s), p_gamma_p2(s)];
    p_analysis_p2_min(s) = find(p_analysis_p2(s,1:5)==min(p_analysis_p2(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.delta_p3(lowIdx), avg.delta_p3(highIdx));
    p_delta_p3(s) = p;
    [h, p, ci, stats] = ttest2(avg.theta_p3(lowIdx), avg.theta_p3(highIdx));
    p_theta_p3(s) = p;
    [h, p, ci, stats] = ttest2(avg.alpha_p3(lowIdx), avg.alpha_p3(highIdx));
    p_alpha_p3(s) = p;
    [h, p, ci, stats] = ttest2(avg.beta_p3(lowIdx), avg.beta_p3(highIdx));
    p_beta_p3(s) = p;
    [h, p, ci, stats] = ttest2(avg.gamma_p3(lowIdx), avg.gamma_p3(highIdx));
    p_gamma_p3(s) = p;
    
    p_analysis_p3(s,1:5) = [p_delta_p3(s), p_theta_p3(s), p_alpha_p3(s), p_beta_p3(s), p_gamma_p3(s)];
    p_analysis_p3_min(s) = find(p_analysis_p3(s,1:5)==min(p_analysis_p3(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.delta_t1(lowIdx), avg.delta_t1(highIdx));
    p_delta_t1(s) = p;
    [h, p, ci, stats] = ttest2(avg.theta_t1(lowIdx), avg.theta_t1(highIdx));
    p_theta_t1(s) = p;
    [h, p, ci, stats] = ttest2(avg.alpha_t1(lowIdx), avg.alpha_t1(highIdx));
    p_alpha_t1(s) = p;
    [h, p, ci, stats] = ttest2(avg.beta_t1(lowIdx), avg.beta_t1(highIdx));
    p_beta_t1(s) = p;
    [h, p, ci, stats] = ttest2(avg.gamma_t1(lowIdx), avg.gamma_t1(highIdx));
    p_gamma_t1(s) = p;
    
    p_analysis_t1(s,1:5) = [p_delta_t1(s), p_theta_t1(s), p_alpha_t1(s), p_beta_t1(s), p_gamma_t1(s)];
    p_analysis_t1_min(s) = find(p_analysis_t1(s,1:5)==min(p_analysis_t1(s,1:5)));
    
    [h, p, ci, stats] = ttest2(avg.delta_t2(lowIdx), avg.delta_t2(highIdx));
    p_delta_t2(s) = p;
    [h, p, ci, stats] = ttest2(avg.theta_t2(lowIdx), avg.theta_t2(highIdx));
    p_theta_t2(s) = p;
    [h, p, ci, stats] = ttest2(avg.alpha_t2(lowIdx), avg.alpha_t2(highIdx));
    p_alpha_t2(s) = p;
    [h, p, ci, stats] = ttest2(avg.beta_t2(lowIdx), avg.beta_t2(highIdx));
    p_beta_t2(s) = p;
    [h, p, ci, stats] = ttest2(avg.gamma_t2(lowIdx), avg.gamma_t2(highIdx));
    p_gamma_t2(s) = p;
    
    p_analysis_t2(s,1:5) = [p_delta_t2(s), p_theta_t2(s), p_alpha_t2(s), p_beta_t2(s), p_gamma_t2(s)];
    p_analysis_t2_min(s) = find(p_analysis_t2(s,1:5)==min(p_analysis_t2(s,1:5)));
    
    
    
    [h, p, ci, stats] = ttest2(avg.delta_o1(lowIdx), avg.delta_o1(highIdx));
    p_delta_o1(s) = p;
    [h, p, ci, stats] = ttest2(avg.theta_o1(lowIdx), avg.theta_o1(highIdx));
    p_theta_o1(s) = p;
    [h, p, ci, stats] = ttest2(avg.alpha_o1(lowIdx), avg.alpha_o1(highIdx));
    p_alpha_o1(s) = p;
    [h, p, ci, stats] = ttest2(avg.beta_o1(lowIdx), avg.beta_o1(highIdx));
    p_beta_o1(s) = p;
    [h, p, ci, stats] = ttest2(avg.gamma_o1(lowIdx), avg.gamma_o1(highIdx));
    p_gamma_o1(s) = p;
    
    p_analysis_o1(s,1:5) = [p_delta_o1(s), p_theta_o1(s), p_alpha_o1(s), p_beta_o1(s), p_gamma_o1(s)];
    p_analysis_o1_min(s) = find(p_analysis_o1(s,1:5)==min(p_analysis_o1(s,1:5)));
    
    [h, p, ci, stats] = ttest2(avg.delta_o2(lowIdx), avg.delta_o2(highIdx));
    p_delta_o2(s) = p;
    [h, p, ci, stats] = ttest2(avg.theta_o2(lowIdx), avg.theta_o2(highIdx));
    p_theta_o2(s) = p;
    [h, p, ci, stats] = ttest2(avg.alpha_o2(lowIdx), avg.alpha_o2(highIdx));
    p_alpha_o2(s) = p;
    [h, p, ci, stats] = ttest2(avg.beta_o2(lowIdx), avg.beta_o2(highIdx));
    p_beta_o2(s) = p;
    [h, p, ci, stats] = ttest2(avg.gamma_o2(lowIdx), avg.gamma_o2(highIdx));
    p_gamma_o2(s) = p;
    
    p_analysis_o2(s,1:5) = [p_delta_o2(s), p_theta_o2(s), p_alpha_o2(s), p_beta_o2(s), p_gamma_o2(s)];
    p_analysis_o2_min(s) = find(p_analysis_o2(s,1:5)==min(p_analysis_o2(s,1:5)));
    
    [h, p, ci, stats] = ttest2(avg.delta_o3(lowIdx), avg.delta_o3(highIdx));
    p_delta_o3(s) = p;
    [h, p, ci, stats] = ttest2(avg.theta_o3(lowIdx), avg.theta_o3(highIdx));
    p_theta_o3(s) = p;
    [h, p, ci, stats] = ttest2(avg.alpha_o3(lowIdx), avg.alpha_o3(highIdx));
    p_alpha_o3(s) = p;
    [h, p, ci, stats] = ttest2(avg.beta_o3(lowIdx), avg.beta_o3(highIdx));
    p_beta_o3(s) = p;
    [h, p, ci, stats] = ttest2(avg.gamma_o3(lowIdx), avg.gamma_o3(highIdx));
    p_gamma_o3(s) = p;
    
    p_analysis_o3(s,1:5) = [p_delta_o3(s), p_theta_o3(s), p_alpha_o3(s), p_beta_o3(s), p_gamma_o3(s)];
    p_analysis_o3_min(s) = find(p_analysis_o3(s,1:5)==min(p_analysis_o3(s,1:5)));
    
    Result.ttest.vertical(s,:) = [p_analysis_f1(s,:) p_analysis_f2(s,:) p_analysis_f3(s,:) p_analysis_c1(s,:) p_analysis_c2(s,:) p_analysis_c3(s,:) p_analysis_p1(s,:) p_analysis_p2(s,:) p_analysis_p3(s,:) p_analysis_t1(s,:) p_analysis_t2(s,:) p_analysis_o1(s,:) p_analysis_o2(s,:) p_analysis_o3(s,:)];
    
    
    
        % Regional Vertical T-TEST 분석
    [h, p, ci, stats] = ttest2(avg.RPL.delta_f1(lowIdx), avg.RPL.delta_f1(highIdx));
    p_delta_f1(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.theta_f1(lowIdx), avg.RPL.theta_f1(highIdx));
    p_theta_f1(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.alpha_f1(lowIdx), avg.RPL.alpha_f1(highIdx));
    p_alpha_f1(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.beta_f1(lowIdx), avg.RPL.beta_f1(highIdx));
    p_beta_f1(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.gamma_f1(lowIdx), avg.RPL.gamma_f1(highIdx));
    p_gamma_f1(s) = p;
    
    p_analysis_f1_rpl(s,1:5) = [p_delta_f1(s), p_theta_f1(s), p_alpha_f1(s), p_beta_f1(s), p_gamma_f1(s)];
    p_analysis_f1_rpl_min(s) = find(p_analysis_f1_rpl(s,1:5)==min(p_analysis_f1_rpl(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.RPL.delta_f2(lowIdx), avg.RPL.delta_f2(highIdx));
    p_delta_f2(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.theta_f2(lowIdx), avg.RPL.theta_f2(highIdx));
    p_theta_f2(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.alpha_f2(lowIdx), avg.RPL.alpha_f2(highIdx));
    p_alpha_f2(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.beta_f2(lowIdx), avg.RPL.beta_f2(highIdx));
    p_beta_f2(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.gamma_f2(lowIdx), avg.RPL.gamma_f2(highIdx));
    p_gamma_f2(s) = p;
    
    p_analysis_f2_rpl(s,1:5) = [p_delta_f2(s), p_theta_f2(s), p_alpha_f2(s), p_beta_f2(s), p_gamma_f2(s)];
    p_analysis_f2_rpl_min(s) = find(p_analysis_f2_rpl(s,1:5)==min(p_analysis_f2_rpl(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.RPL.delta_f3(lowIdx), avg.RPL.delta_f3(highIdx));
    p_delta_f3(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.theta_f3(lowIdx), avg.RPL.theta_f3(highIdx));
    p_theta_f3(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.alpha_f3(lowIdx), avg.RPL.alpha_f3(highIdx));
    p_alpha_f3(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.beta_f3(lowIdx), avg.RPL.beta_f3(highIdx));
    p_beta_f3(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.gamma_f3(lowIdx), avg.RPL.gamma_f3(highIdx));
    p_gamma_f3(s) = p;
    
    p_analysis_f3_rpl(s,1:5) = [p_delta_f3(s), p_theta_f3(s), p_alpha_f3(s), p_beta_f3(s), p_gamma_f3(s)];
    p_analysis_f3_rpl_min(s) = find(p_analysis_f3_rpl(s,1:5)==min(p_analysis_f3_rpl(s,1:5)));
    
    
    
    [h, p, ci, stats] = ttest2(avg.RPL.delta_c1(lowIdx), avg.RPL.delta_c1(highIdx));
    p_delta_c1(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.theta_c1(lowIdx), avg.RPL.theta_c1(highIdx));
    p_theta_c1(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.alpha_c1(lowIdx), avg.RPL.alpha_c1(highIdx));
    p_alpha_c1(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.beta_c1(lowIdx), avg.RPL.beta_c1(highIdx));
    p_beta_c1(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.gamma_c1(lowIdx), avg.RPL.gamma_c1(highIdx));
    p_gamma_c1(s) = p;
    
    p_analysis_c1_rpl(s,1:5) = [p_delta_c1(s), p_theta_c1(s), p_alpha_c1(s), p_beta_c1(s), p_gamma_c1(s)];
    p_analysis_c1_rpl_min(s) = find(p_analysis_c1_rpl(s,1:5)==min(p_analysis_c1_rpl(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.RPL.delta_c2(lowIdx), avg.RPL.delta_c2(highIdx));
    p_delta_c2(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.theta_c2(lowIdx), avg.RPL.theta_c2(highIdx));
    p_theta_c2(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.alpha_c2(lowIdx), avg.RPL.alpha_c2(highIdx));
    p_alpha_c2(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.beta_c2(lowIdx), avg.RPL.beta_c2(highIdx));
    p_beta_c2(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.gamma_c2(lowIdx), avg.RPL.gamma_c2(highIdx));
    p_gamma_c2(s) = p;
    
    p_analysis_c2_rpl(s,1:5) = [p_delta_c2(s), p_theta_c2(s), p_alpha_c2(s), p_beta_c2(s), p_gamma_c2(s)];
    p_analysis_c2_rpl_min(s) = find(p_analysis_c2_rpl(s,1:5)==min(p_analysis_c2_rpl(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.RPL.delta_c3(lowIdx), avg.RPL.delta_c3(highIdx));
    p_delta_c3(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.theta_c3(lowIdx), avg.RPL.theta_c3(highIdx));
    p_theta_c3(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.alpha_c3(lowIdx), avg.RPL.alpha_c3(highIdx));
    p_alpha_c3(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.beta_c3(lowIdx), avg.RPL.beta_c3(highIdx));
    p_beta_c3(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.gamma_c3(lowIdx), avg.RPL.gamma_c3(highIdx));
    p_gamma_c3(s) = p;
    
    p_analysis_c3_rpl(s,1:5) = [p_delta_c3(s), p_theta_c3(s), p_alpha_c3(s), p_beta_c3(s), p_gamma_c3(s)];
    p_analysis_c3_rpl_min(s) = find(p_analysis_c3_rpl(s,1:5)==min(p_analysis_c3_rpl(s,1:5)));
    
    

    
    
    [h, p, ci, stats] = ttest2(avg.RPL.delta_p1(lowIdx), avg.RPL.delta_p1(highIdx));
    p_delta_p1(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.theta_p1(lowIdx), avg.RPL.theta_p1(highIdx));
    p_theta_p1(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.alpha_p1(lowIdx), avg.RPL.alpha_p1(highIdx));
    p_alpha_p1(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.beta_p1(lowIdx), avg.RPL.beta_p1(highIdx));
    p_beta_p1(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.gamma_p1(lowIdx), avg.RPL.gamma_p1(highIdx));
    p_gamma_p1(s) = p;
    
    p_analysis_p1_rpl(s,1:5) = [p_delta_p1(s), p_theta_p1(s), p_alpha_p1(s), p_beta_p1(s), p_gamma_p1(s)];
    p_analysis_p1_rpl_min(s) = find(p_analysis_p1_rpl(s,1:5)==min(p_analysis_p1_rpl(s,1:5)));
    
    
    
    [h, p, ci, stats] = ttest2(avg.RPL.delta_p2(lowIdx), avg.RPL.delta_p2(highIdx));
    p_delta_p2(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.theta_p2(lowIdx), avg.RPL.theta_p2(highIdx));
    p_theta_p2(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.alpha_p2(lowIdx), avg.RPL.alpha_p2(highIdx));
    p_alpha_p2(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.beta_p2(lowIdx), avg.RPL.beta_p2(highIdx));
    p_beta_p2(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.gamma_p2(lowIdx), avg.RPL.gamma_p2(highIdx));
    p_gamma_p2(s) = p;
    
    p_analysis_p2_rpl(s,1:5) = [p_delta_p2(s), p_theta_p2(s), p_alpha_p2(s), p_beta_p2(s), p_gamma_p2(s)];
    p_analysis_p2_rpl_min(s) = find(p_analysis_p2_rpl(s,1:5)==min(p_analysis_p2_rpl(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.RPL.delta_p3(lowIdx), avg.RPL.delta_p3(highIdx));
    p_delta_p3(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.theta_p3(lowIdx), avg.RPL.theta_p3(highIdx));
    p_theta_p3(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.alpha_p3(lowIdx), avg.RPL.alpha_p3(highIdx));
    p_alpha_p3(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.beta_p3(lowIdx), avg.RPL.beta_p3(highIdx));
    p_beta_p3(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.gamma_p3(lowIdx), avg.RPL.gamma_p3(highIdx));
    p_gamma_p3(s) = p;
    
    p_analysis_p3_rpl(s,1:5) = [p_delta_p3(s), p_theta_p3(s), p_alpha_p3(s), p_beta_p3(s), p_gamma_p3(s)];
    p_analysis_p3_rpl_min(s) = find(p_analysis_p3_rpl(s,1:5)==min(p_analysis_p3_rpl(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.RPL.delta_t1(lowIdx), avg.RPL.delta_t1(highIdx));
    p_delta_t1(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.theta_t1(lowIdx), avg.RPL.theta_t1(highIdx));
    p_theta_t1(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.alpha_t1(lowIdx), avg.RPL.alpha_t1(highIdx));
    p_alpha_t1(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.beta_t1(lowIdx), avg.RPL.beta_t1(highIdx));
    p_beta_t1(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.gamma_t1(lowIdx), avg.RPL.gamma_t1(highIdx));
    p_gamma_t1(s) = p;
    
    p_analysis_t1_rpl(s,1:5) = [p_delta_t1(s), p_theta_t1(s), p_alpha_t1(s), p_beta_t1(s), p_gamma_t1(s)];
    p_analysis_t1_rpl_min(s) = find(p_analysis_t1_rpl(s,1:5)==min(p_analysis_t1_rpl(s,1:5)));
    
    [h, p, ci, stats] = ttest2(avg.RPL.delta_t2(lowIdx), avg.RPL.delta_t2(highIdx));
    p_delta_t2(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.theta_t2(lowIdx), avg.RPL.theta_t2(highIdx));
    p_theta_t2(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.alpha_t2(lowIdx), avg.RPL.alpha_t2(highIdx));
    p_alpha_t2(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.beta_t2(lowIdx), avg.RPL.beta_t2(highIdx));
    p_beta_t2(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.gamma_t2(lowIdx), avg.RPL.gamma_t2(highIdx));
    p_gamma_t2(s) = p;
    
    p_analysis_t2_rpl(s,1:5) = [p_delta_t2(s), p_theta_t2(s), p_alpha_t2(s), p_beta_t2(s), p_gamma_t2(s)];
    p_analysis_t2_rpl_min(s) = find(p_analysis_t2_rpl(s,1:5)==min(p_analysis_t2_rpl(s,1:5)));
    
    
    
    [h, p, ci, stats] = ttest2(avg.RPL.delta_o1(lowIdx), avg.RPL.delta_o1(highIdx));
    p_delta_o1(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.theta_o1(lowIdx), avg.RPL.theta_o1(highIdx));
    p_theta_o1(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.alpha_o1(lowIdx), avg.RPL.alpha_o1(highIdx));
    p_alpha_o1(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.beta_o1(lowIdx), avg.RPL.beta_o1(highIdx));
    p_beta_o1(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.gamma_o1(lowIdx), avg.RPL.gamma_o1(highIdx));
    p_gamma_o1(s) = p;
    
    p_analysis_o1_rpl(s,1:5) = [p_delta_o1(s), p_theta_o1(s), p_alpha_o1(s), p_beta_o1(s), p_gamma_o1(s)];
    p_analysis_o1_rpl_min(s) = find(p_analysis_o1_rpl(s,1:5)==min(p_analysis_o1_rpl(s,1:5)));
    
    [h, p, ci, stats] = ttest2(avg.RPL.delta_o2(lowIdx), avg.RPL.delta_o2(highIdx));
    p_delta_o2(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.theta_o2(lowIdx), avg.RPL.theta_o2(highIdx));
    p_theta_o2(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.alpha_o2(lowIdx), avg.RPL.alpha_o2(highIdx));
    p_alpha_o2(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.beta_o2(lowIdx), avg.RPL.beta_o2(highIdx));
    p_beta_o2(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.gamma_o2(lowIdx), avg.RPL.gamma_o2(highIdx));
    p_gamma_o2(s) = p;
    
    p_analysis_o2_rpl(s,1:5) = [p_delta_o2(s), p_theta_o2(s), p_alpha_o2(s), p_beta_o2(s), p_gamma_o2(s)];
    p_analysis_o2_rpl_min(s) = find(p_analysis_o2_rpl(s,1:5)==min(p_analysis_o2_rpl(s,1:5)));
    
    [h, p, ci, stats] = ttest2(avg.RPL.delta_o3(lowIdx), avg.RPL.delta_o3(highIdx));
    p_delta_o3(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.theta_o3(lowIdx), avg.RPL.theta_o3(highIdx));
    p_theta_o3(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.alpha_o3(lowIdx), avg.RPL.alpha_o3(highIdx));
    p_alpha_o3(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.beta_o3(lowIdx), avg.RPL.beta_o3(highIdx));
    p_beta_o3(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.gamma_o3(lowIdx), avg.RPL.gamma_o3(highIdx));
    p_gamma_o3(s) = p;
    
    p_analysis_o3_rpl(s,1:5) = [p_delta_o3(s), p_theta_o3(s), p_alpha_o3(s), p_beta_o3(s), p_gamma_o3(s)];
    p_analysis_o3_rpl_min(s) = find(p_analysis_o3_rpl(s,1:5)==min(p_analysis_o3_rpl(s,1:5)));
    
    Result.ttest.vertical_rpl(s,:) = [p_analysis_f1(s,:) p_analysis_f2(s,:) p_analysis_f3(s,:) p_analysis_c1(s,:) p_analysis_c2(s,:) p_analysis_c3(s,:) p_analysis_p1(s,:) p_analysis_p2(s,:) p_analysis_p3(s,:) p_analysis_t1(s,:) p_analysis_t2(s,:) p_analysis_o1(s,:) p_analysis_o2(s,:) p_analysis_o3(s,:)];
    
    
    
    
    
    
    
            % Regional Vertical T-TEST 분석
    [h, p, ci, stats] = ttest2(avg.z.Regional_f1(lowIdx,1), avg.z.Regional_f1(highIdx,1));
    p_delta_f1(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_f1(lowIdx,2), avg.z.Regional_f1(highIdx,2));
    p_theta_f1(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_f1(lowIdx,3), avg.z.Regional_f1(highIdx,3));
    p_alpha_f1(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_f1(lowIdx,4), avg.z.Regional_f1(highIdx,4));
    p_beta_f1(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_f1(lowIdx,5), avg.z.Regional_f1(highIdx,5));
    p_gamma_f1(s) = p;
    
    p_analysis_f1_z(s,1:5) = [p_delta_f1(s), p_theta_f1(s), p_alpha_f1(s), p_beta_f1(s), p_gamma_f1(s)];
    p_analysis_f1_z_min(s) = find(p_analysis_f1_z(s,1:5)==min(p_analysis_f1_z(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.z.Regional_f2(lowIdx,1), avg.z.Regional_f2(highIdx,1));
    p_delta_f2(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_f2(lowIdx,2), avg.z.Regional_f2(highIdx,2));
    p_theta_f2(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_f2(lowIdx,3), avg.z.Regional_f2(highIdx,3));
    p_alpha_f2(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_f2(lowIdx,4), avg.z.Regional_f2(highIdx,4));
    p_beta_f2(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_f2(lowIdx,5), avg.z.Regional_f2(highIdx,5));
    p_gamma_f2(s) = p;
    
    p_analysis_f2_z(s,1:5) = [p_delta_f2(s), p_theta_f2(s), p_alpha_f2(s), p_beta_f2(s), p_gamma_f2(s)];
    p_analysis_f2_z_min(s) = find(p_analysis_f2_z(s,1:5)==min(p_analysis_f2_z(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.z.Regional_f3(lowIdx,1), avg.z.Regional_f3(highIdx,1));
    p_delta_f3(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_f3(lowIdx,2), avg.z.Regional_f3(highIdx,2));
    p_theta_f3(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_f3(lowIdx,3), avg.z.Regional_f3(highIdx,3));
    p_alpha_f3(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_f3(lowIdx,4), avg.z.Regional_f3(highIdx,4));
    p_beta_f3(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_f3(lowIdx,5), avg.z.Regional_f3(highIdx,5));
    p_gamma_f3(s) = p;
    
    p_analysis_f3_z(s,1:5) = [p_delta_f3(s), p_theta_f3(s), p_alpha_f3(s), p_beta_f3(s), p_gamma_f3(s)];
    p_analysis_f3_z_min(s) = find(p_analysis_f3_z(s,1:5)==min(p_analysis_f3_z(s,1:5)));
    
    
    
    [h, p, ci, stats] = ttest2(avg.z.Regional_c1(lowIdx,1), avg.z.Regional_c1(highIdx,1));
    p_delta_c1(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_c1(lowIdx,2), avg.z.Regional_c1(highIdx,2));
    p_theta_c1(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_c1(lowIdx,3), avg.z.Regional_c1(highIdx,3));
    p_alpha_c1(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_c1(lowIdx,4), avg.z.Regional_c1(highIdx,4));
    p_beta_c1(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_c1(lowIdx,5), avg.z.Regional_c1(highIdx,5));
    p_gamma_c1(s) = p;
    
    p_analysis_c1_z(s,1:5) = [p_delta_c1(s), p_theta_c1(s), p_alpha_c1(s), p_beta_c1(s), p_gamma_c1(s)];
    p_analysis_c1_z_min(s) = find(p_analysis_c1_z(s,1:5)==min(p_analysis_c1_z(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.z.Regional_c2(lowIdx,1), avg.z.Regional_c2(highIdx,1));
    p_delta_c2(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_c2(lowIdx,2), avg.z.Regional_c2(highIdx,2));
    p_theta_c2(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_c2(lowIdx,3), avg.z.Regional_c2(highIdx,3));
    p_alpha_c2(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_c2(lowIdx,4), avg.z.Regional_c2(highIdx,4));
    p_beta_c2(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_c2(lowIdx,5), avg.z.Regional_c2(highIdx,5));
    p_gamma_c2(s) = p;
    
    p_analysis_c2_z(s,1:5) = [p_delta_c2(s), p_theta_c2(s), p_alpha_c2(s), p_beta_c2(s), p_gamma_c2(s)];
    p_analysis_c2_z_min(s) = find(p_analysis_c2_z(s,1:5)==min(p_analysis_c2_z(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.z.Regional_c3(lowIdx,1), avg.z.Regional_c3(highIdx,1));
    p_delta_c3(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_c3(lowIdx,2), avg.z.Regional_c3(highIdx,2));
    p_theta_c3(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_c3(lowIdx,3), avg.z.Regional_c3(highIdx,3));
    p_alpha_c3(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_c3(lowIdx,4), avg.z.Regional_c3(highIdx,4));
    p_beta_c3(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_c3(lowIdx,5), avg.z.Regional_c3(highIdx,5));
    p_gamma_c3(s) = p;
    
    p_analysis_c3_z(s,1:5) = [p_delta_c3(s), p_theta_c3(s), p_alpha_c3(s), p_beta_c3(s), p_gamma_c3(s)];
    p_analysis_c3_z_min(s) = find(p_analysis_c3_z(s,1:5)==min(p_analysis_c3_z(s,1:5)));
    
    

    
    
    [h, p, ci, stats] = ttest2(avg.z.Regional_p1(lowIdx,1), avg.z.Regional_p1(highIdx,1));
    p_delta_p1(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_p1(lowIdx,2), avg.z.Regional_p1(highIdx,2));
    p_theta_p1(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_p1(lowIdx,3), avg.z.Regional_p1(highIdx,3));
    p_alpha_p1(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_p1(lowIdx,4), avg.z.Regional_p1(highIdx,4));
    p_beta_p1(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_p1(lowIdx,5), avg.z.Regional_p1(highIdx,5));
    p_gamma_p1(s) = p;
    
    p_analysis_p1_z(s,1:5) = [p_delta_p1(s), p_theta_p1(s), p_alpha_p1(s), p_beta_p1(s), p_gamma_p1(s)];
    p_analysis_p1_z_min(s) = find(p_analysis_p1_z(s,1:5)==min(p_analysis_p1_z(s,1:5)));
    
    
    
    [h, p, ci, stats] = ttest2(avg.z.Regional_p2(lowIdx,1), avg.z.Regional_p2(highIdx,1));
    p_delta_p2(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_p2(lowIdx,2), avg.z.Regional_p2(highIdx,2));
    p_theta_p2(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_p2(lowIdx,3), avg.z.Regional_p2(highIdx,3));
    p_alpha_p2(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_p2(lowIdx,4), avg.z.Regional_p2(highIdx,4));
    p_beta_p2(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_p2(lowIdx,5), avg.z.Regional_p2(highIdx,5));
    p_gamma_p2(s) = p;
    
    p_analysis_p2_z(s,1:5) = [p_delta_p2(s), p_theta_p2(s), p_alpha_p2(s), p_beta_p2(s), p_gamma_p2(s)];
    p_analysis_p2_z_min(s) = find(p_analysis_p2_z(s,1:5)==min(p_analysis_p2_z(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.z.Regional_p3(lowIdx,1), avg.z.Regional_p3(highIdx,1));
    p_delta_p3(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_p3(lowIdx,2), avg.z.Regional_p3(highIdx,2));
    p_theta_p3(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_p3(lowIdx,3), avg.z.Regional_p3(highIdx,3));
    p_alpha_p3(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_p3(lowIdx,4), avg.z.Regional_p3(highIdx,4));
    p_beta_p3(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_p3(lowIdx,5), avg.z.Regional_p3(highIdx,5));
    p_gamma_p3(s) = p;
    
    p_analysis_p3_z(s,1:5) = [p_delta_p3(s), p_theta_p3(s), p_alpha_p3(s), p_beta_p3(s), p_gamma_p3(s)];
    p_analysis_p3_z_min(s) = find(p_analysis_p3_z(s,1:5)==min(p_analysis_p3_z(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.z.Regional_t1(lowIdx,1), avg.z.Regional_t1(highIdx,1));
    p_delta_t1(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_t1(lowIdx,2), avg.z.Regional_t1(highIdx,2));
    p_theta_t1(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_t1(lowIdx,3), avg.z.Regional_t1(highIdx,3));
    p_alpha_t1(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_t1(lowIdx,4), avg.z.Regional_t1(highIdx,4));
    p_beta_t1(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_t1(lowIdx,5), avg.z.Regional_t1(highIdx,5));
    p_gamma_t1(s) = p;
    
    p_analysis_t1_z(s,1:5) = [p_delta_t1(s), p_theta_t1(s), p_alpha_t1(s), p_beta_t1(s), p_gamma_t1(s)];
    p_analysis_t1_z_min(s) = find(p_analysis_t1_z(s,1:5)==min(p_analysis_t1_z(s,1:5)));
    
    [h, p, ci, stats] = ttest2(avg.z.Regional_t2(lowIdx,1), avg.z.Regional_t2(highIdx,1));
    p_delta_t2(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_t2(lowIdx,2), avg.z.Regional_t2(highIdx,2));
    p_theta_t2(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_t2(lowIdx,3), avg.z.Regional_t2(highIdx,3));
    p_alpha_t2(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_t2(lowIdx,4), avg.z.Regional_t2(highIdx,4));
    p_beta_t2(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_t2(lowIdx,5), avg.z.Regional_t2(highIdx,5));
    p_gamma_t2(s) = p;
    
    p_analysis_t2_z(s,1:5) = [p_delta_t2(s), p_theta_t2(s), p_alpha_t2(s), p_beta_t2(s), p_gamma_t2(s)];
    p_analysis_t2_z_min(s) = find(p_analysis_t2_z(s,1:5)==min(p_analysis_t2_z(s,1:5)));
    
    
    
    [h, p, ci, stats] = ttest2(avg.z.Regional_o1(lowIdx,1), avg.z.Regional_o1(highIdx,1));
    p_delta_o1(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_o1(lowIdx,2), avg.z.Regional_o1(highIdx,2));
    p_theta_o1(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_o1(lowIdx,3), avg.z.Regional_o1(highIdx,3));
    p_alpha_o1(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_o1(lowIdx,4), avg.z.Regional_o1(highIdx,4));
    p_beta_o1(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_o1(lowIdx,5), avg.z.Regional_o1(highIdx,5));
    p_gamma_o1(s) = p;
    
    p_analysis_o1_z(s,1:5) = [p_delta_o1(s), p_theta_o1(s), p_alpha_o1(s), p_beta_o1(s), p_gamma_o1(s)];
    p_analysis_o1_z_min(s) = find(p_analysis_o1_z(s,1:5)==min(p_analysis_o1_z(s,1:5)));
    
    [h, p, ci, stats] = ttest2(avg.z.Regional_o2(lowIdx,1), avg.z.Regional_o2(highIdx,1));
    p_delta_o2(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_o2(lowIdx,2), avg.z.Regional_o2(highIdx,2));
    p_theta_o2(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_o2(lowIdx,3), avg.z.Regional_o2(highIdx,3));
    p_alpha_o2(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_o2(lowIdx,4), avg.z.Regional_o2(highIdx,4));
    p_beta_o2(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_o2(lowIdx,5), avg.z.Regional_o2(highIdx,5));
    p_gamma_o2(s) = p;
    
    p_analysis_o2_z(s,1:5) = [p_delta_o2(s), p_theta_o2(s), p_alpha_o2(s), p_beta_o2(s), p_gamma_o2(s)];
    p_analysis_o2_z_min(s) = find(p_analysis_o2_z(s,1:5)==min(p_analysis_o2_z(s,1:5)));
    
    [h, p, ci, stats] = ttest2(avg.z.Regional_o3(lowIdx,1), avg.z.Regional_o3(highIdx,1));
    p_delta_o3(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_o3(lowIdx,2), avg.z.Regional_o3(highIdx,2));
    p_theta_o3(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_o3(lowIdx,3), avg.z.Regional_o3(highIdx,3));
    p_alpha_o3(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_o3(lowIdx,4), avg.z.Regional_o3(highIdx,4));
    p_beta_o3(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_o3(lowIdx,5), avg.z.Regional_o3(highIdx,5));
    p_gamma_o3(s) = p;
    
    p_analysis_o3_z(s,1:5) = [p_delta_o3(s), p_theta_o3(s), p_alpha_o3(s), p_beta_o3(s), p_gamma_o3(s)];
    p_analysis_o3_z_min(s) = find(p_analysis_o3_z(s,1:5)==min(p_analysis_o3_z(s,1:5)));
    
    Result.ttest.vertical_z(s,:) = [p_analysis_f1(s,:) p_analysis_f2(s,:) p_analysis_f3(s,:) p_analysis_c1(s,:) p_analysis_c2(s,:) p_analysis_c3(s,:) p_analysis_p1(s,:) p_analysis_p2(s,:) p_analysis_p3(s,:) p_analysis_t1(s,:) p_analysis_t2(s,:) p_analysis_o1(s,:) p_analysis_o2(s,:) p_analysis_o3(s,:)];
    
    
    
    
    
    
    
    
    
    
    %% Feature Selection
    feature_ALL = avg.PSD;
    feature_PSD = avg.PSD(:,find(p_analysis(s,:)<0.05));
    feature_Regional = [avg.Regional_f(:,find(p_analysis_f(s,:)<0.05)),avg.Regional_c(:,find(p_analysis_c(s,:)<0.05)),avg.Regional_p(:,find(p_analysis_p(s,:)<0.05)),avg.Regional_t(:,find(p_analysis_t(s,:)<0.05)),avg.Regional_o(:,find(p_analysis_o(s,:)<0.05))];
    feature_Vertical = [avg.Regional_f1(:,find(p_analysis_f1(s,:)<0.05)),avg.Regional_f2(:,find(p_analysis_f2(s,:)<0.05)),avg.Regional_f3(:,find(p_analysis_f3(s,:)<0.05)),avg.Regional_c1(:,find(p_analysis_c1(s,:)<0.05)),avg.Regional_c2(:,find(p_analysis_c2(s,:)<0.05)),avg.Regional_c3(:,find(p_analysis_c3(s,:)<0.05)),avg.Regional_p1(:,find(p_analysis_p1(s,:)<0.05)),avg.Regional_p2(:,find(p_analysis_p2(s,:)<0.05)),avg.Regional_p3(:,find(p_analysis_p3(s,:)<0.05)),avg.Regional_t1(:,find(p_analysis_t1(s,:)<0.05)),avg.Regional_t2(:,find(p_analysis_t2(s,:)<0.05)),avg.Regional_o1(:,find(p_analysis_o1(s,:)<0.05)),avg.Regional_o2(:,find(p_analysis_o2(s,:)<0.05)),avg.Regional_o3(:,find(p_analysis_o3(s,:)<0.05))];
    
    feature_ALL_RPL = avg.RPL.PSD;
    feature_PSD_RPL = avg.RPL.PSD(:,find(p_analysis_RPL(s,:)<0.05));
    feature_Regional_RPL = [avg.RPL.Regional_f(:,find(p_analysis_RPL_f(s,:)<0.05)),avg.RPL.Regional_c(:,find(p_analysis_RPL_c(s,:)<0.05)),avg.RPL.Regional_p(:,find(p_analysis_RPL_p(s,:)<0.05)),avg.RPL.Regional_t(:,find(p_analysis_RPL_t(s,:)<0.05)),avg.RPL.Regional_o(:,find(p_analysis_RPL_o(s,:)<0.05))];
    feature_Vertical_RPL = [avg.RPL.Regional_f1(:,find(p_analysis_f1_rpl(s,:)<0.05)),avg.RPL.Regional_f2(:,find(p_analysis_f2_rpl(s,:)<0.05)),avg.RPL.Regional_f3(:,find(p_analysis_f3_rpl(s,:)<0.05)),avg.RPL.Regional_c1(:,find(p_analysis_c1_rpl(s,:)<0.05)),avg.RPL.Regional_c2(:,find(p_analysis_c2_rpl(s,:)<0.05)),avg.RPL.Regional_c3(:,find(p_analysis_c3_rpl(s,:)<0.05)),avg.RPL.Regional_p1(:,find(p_analysis_p1_rpl(s,:)<0.05)),avg.RPL.Regional_p2(:,find(p_analysis_p2_rpl(s,:)<0.05)),avg.RPL.Regional_p3(:,find(p_analysis_p3_rpl(s,:)<0.05)),avg.RPL.Regional_t1(:,find(p_analysis_t1_rpl(s,:)<0.05)),avg.RPL.Regional_t2(:,find(p_analysis_t2_rpl(s,:)<0.05)),avg.RPL.Regional_o1(:,find(p_analysis_o1_rpl(s,:)<0.05)),avg.RPL.Regional_o2(:,find(p_analysis_o2_rpl(s,:)<0.05)),avg.RPL.Regional_o3(:,find(p_analysis_o3_rpl(s,:)<0.05))];

    feature_ALL_z = avg.z.PSD;
    feature_PSD_z = avg.z.PSD(:,find(p_analysis_z(s,:)<0.05));
    feature_Regional_z = [avg.z.Regional_f(:,find(p_analysis_z_f(s,:)<0.05)),avg.z.Regional_c(:,find(p_analysis_z_c(s,:)<0.05)),avg.z.Regional_p(:,find(p_analysis_z_p(s,:)<0.05)),avg.z.Regional_t(:,find(p_analysis_z_t(s,:)<0.05)),avg.z.Regional_o(:,find(p_analysis_z_o(s,:)<0.05))];
    feature_Vertical_z = [avg.z.Regional_f1(:,find(p_analysis_f1_z(s,:)<0.05)),avg.z.Regional_f2(:,find(p_analysis_f2_z(s,:)<0.05)),avg.z.Regional_f3(:,find(p_analysis_f3_z(s,:)<0.05)),avg.z.Regional_c1(:,find(p_analysis_c1_z(s,:)<0.05)),avg.z.Regional_c2(:,find(p_analysis_c2_z(s,:)<0.05)),avg.z.Regional_c3(:,find(p_analysis_c3_z(s,:)<0.05)),avg.z.Regional_p1(:,find(p_analysis_p1_z(s,:)<0.05)),avg.z.Regional_p2(:,find(p_analysis_p2_z(s,:)<0.05)),avg.z.Regional_p3(:,find(p_analysis_p3_z(s,:)<0.05)),avg.z.Regional_t1(:,find(p_analysis_t1_z(s,:)<0.05)),avg.z.Regional_t2(:,find(p_analysis_t2_z(s,:)<0.05)),avg.z.Regional_o1(:,find(p_analysis_o1_z(s,:)<0.05)),avg.z.Regional_o2(:,find(p_analysis_o2_z(s,:)<0.05)),avg.z.Regional_o3(:,find(p_analysis_o3_z(s,:)<0.05))];
    
    %% Classification
    [Accuracy_ALL AUC_ALL] = LDA_4CV(feature_ALL, highIdx, lowIdx, s);
    [Accuracy_PSD AUC_PSD] = LDA_4CV(feature_PSD, highIdx, lowIdx, s);
    [Accuracy_Regional AUC_Regional] = LDA_4CV(feature_Regional, highIdx, lowIdx, s);
    [Accuracy_Vertical AUC_Vertical] = LDA_4CV(feature_Vertical, highIdx, lowIdx, s);
    
    [Accuracy_ALL_RPL AUC_ALL_RPL] = LDA_4CV(feature_ALL_RPL, highIdx, lowIdx, s);
    [Accuracy_PSD_RPL AUC_PSD_RPL] = LDA_4CV(feature_PSD_RPL, highIdx, lowIdx, s);
    [Accuracy_Regional_RPL AUC_Regional_RPL] = LDA_4CV(feature_Regional_RPL, highIdx, lowIdx, s);
    [Accuracy_Vertical_RPL AUC_Vertical_RPL] = LDA_4CV(feature_Vertical_RPL, highIdx, lowIdx, s);
    
    [Accuracy_ALL_z AUC_ALL_z] = LDA_4CV(feature_ALL_z, highIdx, lowIdx, s);
    [Accuracy_PSD_z AUC_PSD_z] = LDA_4CV(feature_PSD_z, highIdx, lowIdx, s);
    [Accuracy_Regional_z AUC_Regional_z] = LDA_4CV(feature_Regional_z, highIdx, lowIdx, s);
    [Accuracy_Vertical_z AUC_Vertical_z] = LDA_4CV(feature_Vertical_z, highIdx, lowIdx, s);
    
    Result.Fatigue_Accuracy(s,:) = [Accuracy_ALL Accuracy_PSD Accuracy_Regional Accuracy_Vertical];
    Result.Fatigue_AUC(s,:) = [AUC_ALL AUC_PSD AUC_Regional AUC_Vertical];
    
    Result.Fatigue_Accuracy_RPL(s,:) = [Accuracy_ALL_RPL Accuracy_PSD_RPL Accuracy_Regional_RPL Accuracy_Vertical_RPL];
    Result.Fatigue_AUC_RPL(s,:) = [AUC_ALL_RPL AUC_PSD_RPL AUC_Regional_RPL AUC_Vertical_RPL];
    
    Result.Fatigue_Accuracy_z(s,:) = [Accuracy_ALL_z Accuracy_PSD_z Accuracy_Regional_z Accuracy_Vertical_z];
    Result.Fatigue_AUC_z(s,:) = [AUC_ALL_z AUC_PSD_z AUC_Regional_z AUC_Vertical_z];
    
%     %% Regression
%     [Prediction_ALL, RMSE_ALL, Error_Rate_ALL] = MLR_4CV(feature_ALL, kss, s);
%     [Prediction_PSD, RMSE_PSD, Error_Rate_PSD] = MLR_4CV(feature_PSD, kss, s);
%     [Prediction_Regional, RMSE_Regional, Error_Rate_Regional] = MLR_4CV(feature_Regional, kss, s);
%     [Prediction_Vertical, RMSE_Vertical, Error_Rate_Vertical] = MLR_4CV(feature_Vertical, kss, s);
%     
%     [Prediction_ALL_RPL, RMSE_ALL_RPL, Error_Rate_ALL_RPL] = MLR_4CV(feature_ALL_RPL, kss, s);
%     [Prediction_PSD_RPL, RMSE_PSD_RPL, Error_Rate_PSD_RPL] = MLR_4CV(feature_PSD_RPL, kss, s);
%     [Prediction_Regional_RPL, RMSE_Regional_RPL, Error_Rate_Regional_RPL] = MLR_4CV(feature_Regional_RPL, kss, s);
%     [Prediction_Vertical_RPL, RMSE_Vertical_RPL, Error_Rate_Vertical_RPL] = MLR_4CV(feature_Vertical_RPL, kss, s);
%     
%     [Prediction_ALL_z, RMSE_ALL_z, Error_Rate_ALL_z] = MLR_4CV(feature_ALL_z, kss, s);
%     [Prediction_PSD_z, RMSE_PSD_z, Error_Rate_PSD_z] = MLR_4CV(feature_PSD_z, kss, s);
%     [Prediction_Regional_z, RMSE_Regional_z, Error_Rate_Regional_z] = MLR_4CV(feature_Regional_z, kss, s);
%     [Prediction_Vertical_z, RMSE_Vertical_z, Error_Rate_Vertical_z] = MLR_4CV(feature_Vertical_z, kss, s);
%  
%     Result.Fatigue_Prediction(s,:) = [Prediction_ALL Prediction_PSD Prediction_Regional Prediction_Vertical];
%     Result.Fatigue_RMSE(s,:) = [RMSE_ALL RMSE_PSD RMSE_Regional RMSE_Vertical];   
%     Result.Fatigue_Error(s,:) = [Error_Rate_ALL Error_Rate_PSD Error_Rate_Regional Error_Rate_Vertical];   
%     
%     Result.Fatigue_Prediction_RPL(s,:) = [Prediction_ALL_RPL Prediction_PSD_RPL Prediction_Regional_RPL Prediction_Vertical_RPL];
%     Result.Fatigue_RMSE_RPL(s,:) = [RMSE_ALL_RPL RMSE_PSD_RPL RMSE_Regional_RPL RMSE_Vertical_RPL];   
%     Result.Fatigue_Error_RPL(s,:) = [Error_Rate_ALL_RPL Error_Rate_PSD_RPL Error_Rate_Regional_RPL Error_Rate_Vertical_RPL];   
%     
%     Result.Fatigue_Prediction_z(s,:) = [Prediction_ALL_z Prediction_PSD_z Prediction_Regional_z Prediction_Vertical_z];
%     Result.Fatigue_RMSE_z(s,:) = [RMSE_ALL_z RMSE_PSD_z RMSE_Regional_z RMSE_Vertical_z];   
%     Result.Fatigue_Error_z(s,:) = [Error_Rate_ALL_z Error_Rate_PSD_z Error_Rate_Regional_z Error_Rate_Vertical_z];   


    %% Save
    Result.Fatigue_Accuracy_30(s,:) = Result.Fatigue_Accuracy(s,:);
    Result.Fatigue_AUC_30(s,:) = Result.Fatigue_AUC(s,:);  
%     Result.Fatigue_Prediction_30(s,:) = Result.Fatigue_Prediction(s,:);
%     Result.Fatigue_RMSE_30(s,:) = Result.Fatigue_RMSE(s,:);
%     Result.Fatigue_Error_30(s,:) = Result.Fatigue_Error(s,:);
    
    Result.Fatigue_Accuracy_30_RPL(s,:) = Result.Fatigue_Accuracy_RPL(s,:);
    Result.Fatigue_AUC_30_RPL(s,:) = Result.Fatigue_AUC_RPL(s,:);  
%     Result.Fatigue_Prediction_30_RPL(s,:) = Result.Fatigue_Prediction_RPL(s,:);
%     Result.Fatigue_RMSE_30_RPL(s,:) = Result.Fatigue_RMSE_RPL(s,:);
%     Result.Fatigue_Error_30_RPL(s,:) = Result.Fatigue_Error_RPL(s,:);
    
    Result.Fatigue_Accuracy_30_z(s,:) = Result.Fatigue_Accuracy_z(s,:);
    Result.Fatigue_AUC_30_z(s,:) = Result.Fatigue_AUC_z(s,:);  
%     Result.Fatigue_Prediction_30_z(s,:) = Result.Fatigue_Prediction_z(s,:);
%     Result.Fatigue_RMSE_30_z(s,:) = Result.Fatigue_RMSE_z(s,:);
%     Result.Fatigue_Error_30_z(s,:) = Result.Fatigue_Error_z(s,:);

    
    
        %% Segmentation of bio-signals with 1 seconds length of epoch
    epoch = segmentationFatigue_30_2(cnt, mrk, interKSS);
    kssIdx = find(strcmp(epoch.mClab, 'KSS'));
    kss = squeeze(mean(epoch.misc(:, kssIdx, :), 1));
    
    %% Regional Channel Selection
    channel_f=epoch.x(:,[1:5,31:36],:);
    channel_c=epoch.x(:,[6:9,11:13,16:19,39:40,43:46,48:50],:);
    channel_p=epoch.x(:,[21:25,52:55],:);
    channel_t=epoch.x(:,[10,14:15,20,37:38,41:42,47,51],:);
    channel_o=epoch.x(:,[26:30,56:60],:);
    
    channel_f1=epoch.x(:,[1,3,31,33],:);
    channel_f2=epoch.x(:,[2,5,32,36],:);
    channel_f3=epoch.x(:,[4,34:35],:);
    channel_c1=epoch.x(:,[6:7,11,16,17,39,43,44,48],:);
    channel_c2=epoch.x(:,[8:9,13,18,19,40,45,46,50],:);
    channel_c3=epoch.x(:,[12,49],:);
    channel_p1=epoch.x(:,[21:22,52,53],:);
    channel_p2=epoch.x(:,[24:25,54,55],:);
    channel_p3=epoch.x(:,[23],:);
    channel_t1=epoch.x(:,[10,15,37,38,47],:);
    channel_t2=epoch.x(:,[14,20,41,42,51],:);
    channel_o1=epoch.x(:,[26,27,56,57],:);
    channel_o2=epoch.x(:,[29,30,59,60],:);
    channel_o3=epoch.x(:,[28,58],:);
    
    %% Feature Extraction - Power spectrum analysis for EEG signals
    
    % PSD Feature Extraction
    [cca.delta, avg.delta] = Power_spectrum(epoch.x, [0.5 4], cnt.fs);
    [cca.theta, avg.theta] = Power_spectrum(epoch.x, [4 8], cnt.fs);
    [cca.alpha, avg.alpha] = Power_spectrum(epoch.x, [8 13], cnt.fs);
    [cca.beta, avg.beta] = Power_spectrum(epoch.x, [13 30], cnt.fs);
    [cca.gamma, avg.gamma] = Power_spectrum(epoch.x, [30 40], cnt.fs);
    avg.delta = avg.delta'; avg.theta=avg.theta'; avg.alpha=avg.alpha'; avg.beta = avg.beta'; avg.gamma = avg.gamma';
    
    cca.PSD = [cca.delta cca.theta cca.alpha cca.beta cca.gamma];
    avg.PSD = [avg.delta avg.theta avg.alpha avg.beta avg.gamma];

    % RPL Feature Extraction
    cca.RPL.delta = cca.delta ./ (cca.delta + cca.theta + cca.alpha + cca.beta + cca.gamma) ;
    avg.RPL.delta = avg.delta ./ sum(avg.PSD,2) ;
    cca.RPL.theta = cca.theta ./ (cca.delta + cca.theta + cca.alpha + cca.beta + cca.gamma) ;
    avg.RPL.theta = avg.theta ./ sum(avg.PSD,2) ;
    cca.RPL.alpha = cca.alpha ./ (cca.delta + cca.theta + cca.alpha + cca.beta + cca.gamma) ;
    avg.RPL.alpha = avg.alpha ./ sum(avg.PSD,2) ;
    cca.RPL.beta = cca.beta ./ (cca.delta + cca.theta + cca.alpha + cca.beta + cca.gamma) ;
    avg.RPL.beta = avg.beta ./ sum(avg.PSD,2) ;
    cca.RPL.gamma = cca.gamma ./ (cca.delta + cca.theta + cca.alpha + cca.beta + cca.gamma) ;
    avg.RPL.gamma = avg.gamma ./ sum(avg.PSD,2) ;
    
    cca.RPL.PSD = [cca.RPL.delta, cca.RPL.theta, cca.RPL.alpha, cca.RPL.beta, cca.RPL.gamma];
    avg.RPL.PSD = [avg.RPL.delta, avg.RPL.theta, avg.RPL.alpha, avg.RPL.beta, avg.RPL.gamma];
    
    % Z-SCORE Feature Extraction
    cca.z.PSD = [zscore(cca.delta) zscore(cca.theta) zscore(cca.alpha) zscore(cca.beta) zscore(cca.gamma)];
    avg.z.PSD = [zscore(avg.delta) zscore(avg.theta) zscore(avg.alpha) zscore(avg.beta) zscore(avg.gamma)];
    
    
    % Regional Feature Extraction
    [cca.delta_f,avg.delta_f] = Power_spectrum(channel_f, [0.5 4], cnt.fs);
    [cca.theta_f,avg.theta_f] = Power_spectrum(channel_f, [4 8], cnt.fs);
    [cca.alpha_f,avg.alpha_f] = Power_spectrum(channel_f, [8 13], cnt.fs);
    [cca.beta_f,avg.beta_f] = Power_spectrum(channel_f, [13 30], cnt.fs);
    [cca.gamma_f,avg.gamma_f] = Power_spectrum(channel_f, [30 40], cnt.fs);
    avg.delta_f = avg.delta_f'; avg.theta_f=avg.theta_f'; avg.alpha_f=avg.alpha_f'; avg.beta_f = avg.beta_f'; avg.gamma_f = avg.gamma_f';
    
    cca.Regional_f = [cca.delta_f cca.theta_f cca.alpha_f cca.beta_f cca.gamma_f];
    avg.Regional_f = [avg.delta_f avg.theta_f avg.alpha_f avg.beta_f avg.gamma_f];
    
    cca.RPL.delta_f = cca.delta_f ./ (cca.delta_f + cca.theta_f + cca.alpha_f + cca.beta_f + cca.gamma_f) ;
    avg.RPL.delta_f = avg.delta_f ./ sum(avg.Regional_f,2) ;
    cca.RPL.theta_f = cca.theta_f ./ (cca.delta_f + cca.theta_f + cca.alpha_f + cca.beta_f + cca.gamma_f) ;
    avg.RPL.theta_f = avg.theta_f ./ sum(avg.Regional_f,2) ;
    cca.RPL.alpha_f = cca.alpha_f ./ (cca.delta_f + cca.theta_f + cca.alpha_f + cca.beta_f + cca.gamma_f) ;
    avg.RPL.alpha_f = avg.alpha_f ./ sum(avg.Regional_f,2) ;
    cca.RPL.beta_f = cca.beta_f ./ (cca.delta_f + cca.theta_f + cca.alpha_f + cca.beta_f + cca.gamma_f) ;
    avg.RPL.beta_f = avg.beta_f ./ sum(avg.Regional_f,2) ;
    cca.RPL.gamma_f = cca.gamma_f ./ (cca.delta_f + cca.theta_f + cca.alpha_f + cca.beta_f + cca.gamma_f) ;
    avg.RPL.gamma_f = avg.gamma_f ./ sum(avg.Regional_f,2) ;    
    
    cca.RPL.Regional_f = [cca.RPL.delta_f cca.RPL.theta_f cca.RPL.alpha_f cca.RPL.beta_f cca.RPL.gamma_f]; 
    avg.RPL.Regional_f = [avg.RPL.delta_f avg.RPL.theta_f avg.RPL.alpha_f avg.RPL.beta_f avg.RPL.gamma_f];
    
    cca.z.Regional_f = [zscore(cca.delta_f) zscore(cca.theta_f) zscore(cca.alpha_f) zscore(cca.beta_f) zscore(cca.gamma_f)];
    avg.z.Regional_f = [zscore(avg.delta_f) zscore(avg.theta_f) zscore(avg.alpha_f) zscore(avg.beta_f) zscore(avg.gamma_f)];   
    
    
    
    [cca.delta_c,avg.delta_c] = Power_spectrum(channel_c, [0.5 4], cnt.fs);
    [cca.theta_c,avg.theta_c] = Power_spectrum(channel_c, [4 8], cnt.fs);
    [cca.alpha_c,avg.alpha_c] = Power_spectrum(channel_c, [8 13], cnt.fs);
    [cca.beta_c,avg.beta_c] = Power_spectrum(channel_c, [13 30], cnt.fs);
    [cca.gamma_c,avg.gamma_c] = Power_spectrum(channel_c, [30 40], cnt.fs);
    avg.delta_c = avg.delta_c'; avg.theta_c=avg.theta_c'; avg.alpha_c=avg.alpha_c'; avg.beta_c = avg.beta_c'; avg.gamma_c = avg.gamma_c';
    
    cca.Regional_c = [cca.delta_c cca.theta_c cca.alpha_c cca.beta_c cca.gamma_c];
    avg.Regional_c = [avg.delta_c avg.theta_c avg.alpha_c avg.beta_c avg.gamma_c];
    
    cca.RPL.delta_c = cca.delta_c ./ (cca.delta_c + cca.theta_c + cca.alpha_c + cca.beta_c + cca.gamma_c) ;
    avg.RPL.delta_c = avg.delta_c ./ sum(avg.Regional_c,2) ;
    cca.RPL.theta_c = cca.theta_c ./ (cca.delta_c + cca.theta_c + cca.alpha_c + cca.beta_c + cca.gamma_c) ;
    avg.RPL.theta_c = avg.theta_c ./ sum(avg.Regional_c,2) ;
    cca.RPL.alpha_c = cca.alpha_c ./ (cca.delta_c + cca.theta_c + cca.alpha_c + cca.beta_c + cca.gamma_c) ;
    avg.RPL.alpha_c = avg.alpha_c ./ sum(avg.Regional_c,2) ;
    cca.RPL.beta_c = cca.beta_c ./ (cca.delta_c + cca.theta_c + cca.alpha_c + cca.beta_c + cca.gamma_c) ;
    avg.RPL.beta_c = avg.beta_c ./ sum(avg.Regional_c,2) ;
    cca.RPL.gamma_c = cca.gamma_c ./ (cca.delta_c + cca.theta_c + cca.alpha_c + cca.beta_c + cca.gamma_c) ;
    avg.RPL.gamma_c = avg.gamma_c ./ sum(avg.Regional_c,2) ;     
    
    cca.RPL.Regional_c = [cca.RPL.delta_c cca.RPL.theta_c cca.RPL.alpha_c cca.RPL.beta_c cca.RPL.gamma_c]; 
    avg.RPL.Regional_c = [avg.RPL.delta_c avg.RPL.theta_c avg.RPL.alpha_c avg.RPL.beta_c avg.RPL.gamma_c];
    
    cca.z.Regional_c = [zscore(cca.delta_c) zscore(cca.theta_c) zscore(cca.alpha_c) zscore(cca.beta_c) zscore(cca.gamma_c)];
    avg.z.Regional_c = [zscore(avg.delta_c) zscore(avg.theta_c) zscore(avg.alpha_c) zscore(avg.beta_c) zscore(avg.gamma_c)];   
    
    
    [cca.delta_p,avg.delta_p] = Power_spectrum(channel_p, [0.5 4], cnt.fs);
    [cca.theta_p,avg.theta_p] = Power_spectrum(channel_p, [4 8], cnt.fs);
    [cca.alpha_p,avg.alpha_p] = Power_spectrum(channel_p, [8 13], cnt.fs);
    [cca.beta_p,avg.beta_p] = Power_spectrum(channel_p, [13 30], cnt.fs);
    [cca.gamma_p,avg.gamma_p] = Power_spectrum(channel_p, [30 40], cnt.fs);
    avg.delta_p = avg.delta_p'; avg.theta_p=avg.theta_p'; avg.alpha_p=avg.alpha_p'; avg.beta_p = avg.beta_p'; avg.gamma_p = avg.gamma_p';
    
    cca.Regional_p = [cca.delta_p cca.theta_p cca.alpha_p cca.beta_p cca.gamma_p];
    avg.Regional_p = [avg.delta_p avg.theta_p avg.alpha_p avg.beta_p avg.gamma_p];

    cca.RPL.delta_p = cca.delta_p ./ (cca.delta_p + cca.theta_p + cca.alpha_p + cca.beta_p + cca.gamma_p) ;
    avg.RPL.delta_p = avg.delta_p ./ sum(avg.Regional_p,2) ;
    cca.RPL.theta_p = cca.theta_p ./ (cca.delta_p + cca.theta_p + cca.alpha_p + cca.beta_p + cca.gamma_p) ;
    avg.RPL.theta_p = avg.theta_p ./ sum(avg.Regional_p,2) ;
    cca.RPL.alpha_p = cca.alpha_p ./ (cca.delta_p + cca.theta_p + cca.alpha_p + cca.beta_p + cca.gamma_p) ;
    avg.RPL.alpha_p = avg.alpha_p ./ sum(avg.Regional_p,2) ;
    cca.RPL.beta_p = cca.beta_p ./ (cca.delta_p + cca.theta_p + cca.alpha_p + cca.beta_p + cca.gamma_p) ;
    avg.RPL.beta_p = avg.beta_p ./ sum(avg.Regional_p,2) ;
    cca.RPL.gamma_p = cca.gamma_p ./ (cca.delta_p + cca.theta_p + cca.alpha_p + cca.beta_p + cca.gamma_p) ;
    avg.RPL.gamma_p = avg.gamma_p ./ sum(avg.Regional_p,2) ;     
    
    cca.RPL.Regional_p = [cca.RPL.delta_p cca.RPL.theta_p cca.RPL.alpha_p cca.RPL.beta_p cca.RPL.gamma_p]; 
    avg.RPL.Regional_p = [avg.RPL.delta_p avg.RPL.theta_p avg.RPL.alpha_p avg.RPL.beta_p avg.RPL.gamma_p];
    
    cca.z.Regional_p = [zscore(cca.delta_p) zscore(cca.theta_p) zscore(cca.alpha_p) zscore(cca.beta_p) zscore(cca.gamma_p)];
    avg.z.Regional_p = [zscore(avg.delta_p) zscore(avg.theta_p) zscore(avg.alpha_p) zscore(avg.beta_p) zscore(avg.gamma_p)];   
    
    
    [cca.delta_t,avg.delta_t] = Power_spectrum(channel_t, [0.5 4], cnt.fs);
    [cca.theta_t,avg.theta_t] = Power_spectrum(channel_t, [4 8], cnt.fs);
    [cca.alpha_t,avg.alpha_t] = Power_spectrum(channel_t, [8 13], cnt.fs);
    [cca.beta_t,avg.beta_t] = Power_spectrum(channel_t, [13 30], cnt.fs);
    [cca.gamma_t,avg.gamma_t] = Power_spectrum(channel_t, [30 40], cnt.fs);
    avg.delta_t = avg.delta_t'; avg.theta_t=avg.theta_t'; avg.alpha_t=avg.alpha_t'; avg.beta_t = avg.beta_t'; avg.gamma_t = avg.gamma_t';
    
    cca.Regional_t = [cca.delta_t cca.theta_t cca.alpha_t cca.beta_t cca.gamma_t];
    avg.Regional_t = [avg.delta_t avg.theta_t avg.alpha_t avg.beta_t avg.gamma_t];

    cca.RPL.delta_t = cca.delta_t ./ (cca.delta_t + cca.theta_t + cca.alpha_t + cca.beta_t + cca.gamma_t) ;
    avg.RPL.delta_t = avg.delta_t ./ sum(avg.Regional_t,2) ;
    cca.RPL.theta_t = cca.theta_t ./ (cca.delta_t + cca.theta_t + cca.alpha_t + cca.beta_t + cca.gamma_t) ;
    avg.RPL.theta_t = avg.theta_t ./ sum(avg.Regional_t,2) ;
    cca.RPL.alpha_t = cca.alpha_t ./ (cca.delta_t + cca.theta_t + cca.alpha_t + cca.beta_t + cca.gamma_t) ;
    avg.RPL.alpha_t = avg.alpha_t ./ sum(avg.Regional_t,2) ;
    cca.RPL.beta_t = cca.beta_t ./ (cca.delta_t + cca.theta_t + cca.alpha_t + cca.beta_t + cca.gamma_t) ;
    avg.RPL.beta_t = avg.beta_t ./ sum(avg.Regional_t,2) ;
    cca.RPL.gamma_t = cca.gamma_t ./ (cca.delta_t + cca.theta_t + cca.alpha_t + cca.beta_t + cca.gamma_t) ;
    avg.RPL.gamma_t = avg.gamma_t ./ sum(avg.Regional_t,2) ;     
    
    cca.RPL.Regional_t = [cca.RPL.delta_t cca.RPL.theta_t cca.RPL.alpha_t cca.RPL.beta_t cca.RPL.gamma_t]; 
    avg.RPL.Regional_t = [avg.RPL.delta_t avg.RPL.theta_t avg.RPL.alpha_t avg.RPL.beta_t avg.RPL.gamma_t];
    
    cca.z.Regional_t = [zscore(cca.delta_t) zscore(cca.theta_t) zscore(cca.alpha_t) zscore(cca.beta_t) zscore(cca.gamma_t)];
    avg.z.Regional_t = [zscore(avg.delta_t) zscore(avg.theta_t) zscore(avg.alpha_t) zscore(avg.beta_t) zscore(avg.gamma_t)];   
    
    
    [cca.delta_o,avg.delta_o] = Power_spectrum(channel_o, [0.5 4], cnt.fs);
    [cca.theta_o,avg.theta_o] = Power_spectrum(channel_o, [4 8], cnt.fs);
    [cca.alpha_o,avg.alpha_o] = Power_spectrum(channel_o, [8 13], cnt.fs);
    [cca.beta_o,avg.beta_o] = Power_spectrum(channel_o, [13 30], cnt.fs);
    [cca.gamma_o,avg.gamma_o] = Power_spectrum(channel_o, [30 40], cnt.fs);
    avg.delta_o = avg.delta_o'; avg.theta_o=avg.theta_o'; avg.alpha_o=avg.alpha_o'; avg.beta_o = avg.beta_o'; avg.gamma_o = avg.gamma_o';
    
    cca.Regional_o = [cca.delta_o cca.theta_o cca.alpha_o cca.beta_o cca.gamma_o];
    avg.Regional_o = [avg.delta_o avg.theta_o avg.alpha_o avg.beta_o avg.gamma_o];
    
    cca.RPL.delta_o = cca.delta_o ./ (cca.delta_o + cca.theta_o + cca.alpha_o + cca.beta_o + cca.gamma_o) ;
    avg.RPL.delta_o = avg.delta_o ./ sum(avg.Regional_o,2) ;
    cca.RPL.theta_o = cca.theta_o ./ (cca.delta_o + cca.theta_o + cca.alpha_o + cca.beta_o + cca.gamma_o) ;
    avg.RPL.theta_o = avg.theta_o ./ sum(avg.Regional_o,2) ;
    cca.RPL.alpha_o = cca.alpha_o ./ (cca.delta_o + cca.theta_o + cca.alpha_o + cca.beta_o + cca.gamma_o) ;
    avg.RPL.alpha_o = avg.alpha_o ./ sum(avg.Regional_o,2) ;
    cca.RPL.beta_o = cca.beta_o ./ (cca.delta_o + cca.theta_o + cca.alpha_o + cca.beta_o + cca.gamma_o) ;
    avg.RPL.beta_o = avg.beta_o ./ sum(avg.Regional_o,2) ;
    cca.RPL.gamma_o = cca.gamma_o ./ (cca.delta_o + cca.theta_o + cca.alpha_o + cca.beta_o + cca.gamma_o) ;
    avg.RPL.gamma_o = avg.gamma_o ./ sum(avg.Regional_o,2) ;     
    
    cca.RPL.Regional_o = [cca.RPL.delta_o cca.RPL.theta_o cca.RPL.alpha_o cca.RPL.beta_o cca.RPL.gamma_o]; 
    avg.RPL.Regional_o = [avg.RPL.delta_o avg.RPL.theta_o avg.RPL.alpha_o avg.RPL.beta_o avg.RPL.gamma_o];
    
    cca.z.Regional_o = [zscore(cca.delta_o) zscore(cca.theta_o) zscore(cca.alpha_o) zscore(cca.beta_o) zscore(cca.gamma_o)];
    avg.z.Regional_o = [zscore(avg.delta_o) zscore(avg.theta_o) zscore(avg.alpha_o) zscore(avg.beta_o) zscore(avg.gamma_o)];   
    
    
    
    
    % Regional Vertical Feature Extraction
    [cca.delta_f1,avg.delta_f1] = Power_spectrum(channel_f1, [0.5 4], cnt.fs);
    [cca.theta_f1,avg.theta_f1] = Power_spectrum(channel_f1, [4 8], cnt.fs);
    [cca.alpha_f1,avg.alpha_f1] = Power_spectrum(channel_f1, [8 13], cnt.fs);
    [cca.beta_f1,avg.beta_f1] = Power_spectrum(channel_f1, [13 30], cnt.fs);
    [cca.gamma_f1,avg.gamma_f1] = Power_spectrum(channel_f1, [30 40], cnt.fs);
    avg.delta_f1 = avg.delta_f1'; avg.theta_f1=avg.theta_f1'; avg.alpha_f1=avg.alpha_f1'; avg.beta_f1 = avg.beta_f1'; avg.gamma_f1 = avg.gamma_f1';
    
    cca.Regional_f1 = [cca.delta_f1 cca.theta_f1 cca.alpha_f1 cca.beta_f1 cca.gamma_f1];
    avg.Regional_f1 = [avg.delta_f1 avg.theta_f1 avg.alpha_f1 avg.beta_f1 avg.gamma_f1];
    
    cca.RPL.delta_f1 = cca.delta_f1 ./ (cca.delta_f1 + cca.theta_f1 + cca.alpha_f1 + cca.beta_f1 + cca.gamma_f1) ;
    avg.RPL.delta_f1 = avg.delta_f1 ./ sum(avg.Regional_f1,2) ;
    cca.RPL.theta_f1 = cca.theta_f1 ./ (cca.delta_f1 + cca.theta_f1 + cca.alpha_f1 + cca.beta_f1 + cca.gamma_f1) ;
    avg.RPL.theta_f1 = avg.theta_f1 ./ sum(avg.Regional_f1,2) ;
    cca.RPL.alpha_f1 = cca.alpha_f1 ./ (cca.delta_f1 + cca.theta_f1 + cca.alpha_f1 + cca.beta_f1 + cca.gamma_f1) ;
    avg.RPL.alpha_f1 = avg.alpha_f1 ./ sum(avg.Regional_f1,2) ;
    cca.RPL.beta_f1 = cca.beta_f1 ./ (cca.delta_f1 + cca.theta_f1 + cca.alpha_f1 + cca.beta_f1 + cca.gamma_f1) ;
    avg.RPL.beta_f1 = avg.beta_f1 ./ sum(avg.Regional_f1,2) ;
    cca.RPL.gamma_f1 = cca.gamma_f1 ./ (cca.delta_f1 + cca.theta_f1 + cca.alpha_f1 + cca.beta_f1 + cca.gamma_f1) ;
    avg.RPL.gamma_f1 = avg.gamma_f1 ./ sum(avg.Regional_f1,2) ;    
    
    cca.RPL.Regional_f1 = [cca.RPL.delta_f1 cca.RPL.theta_f1 cca.RPL.alpha_f1 cca.RPL.beta_f1 cca.RPL.gamma_f1]; 
    avg.RPL.Regional_f1 = [avg.RPL.delta_f1 avg.RPL.theta_f1 avg.RPL.alpha_f1 avg.RPL.beta_f1 avg.RPL.gamma_f1];
    
    cca.z.Regional_f1 = [zscore(cca.delta_f1) zscore(cca.theta_f1) zscore(cca.alpha_f1) zscore(cca.beta_f1) zscore(cca.gamma_f1)];
    avg.z.Regional_f1 = [zscore(avg.delta_f1) zscore(avg.theta_f1) zscore(avg.alpha_f1) zscore(avg.beta_f1) zscore(avg.gamma_f1)];   
    
    
    [cca.delta_f2,avg.delta_f2] = Power_spectrum(channel_f2, [0.5 4], cnt.fs);
    [cca.theta_f2,avg.theta_f2] = Power_spectrum(channel_f2, [4 8], cnt.fs);
    [cca.alpha_f2,avg.alpha_f2] = Power_spectrum(channel_f2, [8 13], cnt.fs);
    [cca.beta_f2,avg.beta_f2] = Power_spectrum(channel_f2, [13 30], cnt.fs);
    [cca.gamma_f2,avg.gamma_f2] = Power_spectrum(channel_f2, [30 40], cnt.fs);
    avg.delta_f2 = avg.delta_f2'; avg.theta_f2=avg.theta_f2'; avg.alpha_f2=avg.alpha_f2'; avg.beta_f2 = avg.beta_f2'; avg.gamma_f2 = avg.gamma_f2';
    
    cca.Regional_f2 = [cca.delta_f2 cca.theta_f2 cca.alpha_f2 cca.beta_f2 cca.gamma_f2];
    avg.Regional_f2 = [avg.delta_f2 avg.theta_f2 avg.alpha_f2 avg.beta_f2 avg.gamma_f2];
    
    cca.RPL.delta_f2 = cca.delta_f2 ./ (cca.delta_f2 + cca.theta_f2 + cca.alpha_f2 + cca.beta_f2 + cca.gamma_f2) ;
    avg.RPL.delta_f2 = avg.delta_f2 ./ sum(avg.Regional_f2,2) ;
    cca.RPL.theta_f2 = cca.theta_f2 ./ (cca.delta_f2 + cca.theta_f2 + cca.alpha_f2 + cca.beta_f2 + cca.gamma_f2) ;
    avg.RPL.theta_f2 = avg.theta_f2 ./ sum(avg.Regional_f2,2) ;
    cca.RPL.alpha_f2 = cca.alpha_f2 ./ (cca.delta_f2 + cca.theta_f2 + cca.alpha_f2 + cca.beta_f2 + cca.gamma_f2) ;
    avg.RPL.alpha_f2 = avg.alpha_f2 ./ sum(avg.Regional_f2,2) ;
    cca.RPL.beta_f2 = cca.beta_f2 ./ (cca.delta_f2 + cca.theta_f2 + cca.alpha_f2 + cca.beta_f2 + cca.gamma_f2) ;
    avg.RPL.beta_f2 = avg.beta_f2 ./ sum(avg.Regional_f2,2) ;
    cca.RPL.gamma_f2 = cca.gamma_f2 ./ (cca.delta_f2 + cca.theta_f2 + cca.alpha_f2 + cca.beta_f2 + cca.gamma_f2) ;
    avg.RPL.gamma_f2 = avg.gamma_f2 ./ sum(avg.Regional_f2,2) ;    
    
    cca.RPL.Regional_f2 = [cca.RPL.delta_f2 cca.RPL.theta_f2 cca.RPL.alpha_f2 cca.RPL.beta_f2 cca.RPL.gamma_f2]; 
    avg.RPL.Regional_f2 = [avg.RPL.delta_f2 avg.RPL.theta_f2 avg.RPL.alpha_f2 avg.RPL.beta_f2 avg.RPL.gamma_f2];
    
    cca.z.Regional_f2 = [zscore(cca.delta_f2) zscore(cca.theta_f2) zscore(cca.alpha_f2) zscore(cca.beta_f2) zscore(cca.gamma_f2)];
    avg.z.Regional_f2 = [zscore(avg.delta_f2) zscore(avg.theta_f2) zscore(avg.alpha_f2) zscore(avg.beta_f2) zscore(avg.gamma_f2)];       
    
    
    [cca.delta_f3,avg.delta_f3] = Power_spectrum(channel_f3, [0.5 4], cnt.fs);
    [cca.theta_f3,avg.theta_f3] = Power_spectrum(channel_f3, [4 8], cnt.fs);
    [cca.alpha_f3,avg.alpha_f3] = Power_spectrum(channel_f3, [8 13], cnt.fs);
    [cca.beta_f3,avg.beta_f3] = Power_spectrum(channel_f3, [13 30], cnt.fs);
    [cca.gamma_f3,avg.gamma_f3] = Power_spectrum(channel_f3, [30 40], cnt.fs);
    avg.delta_f3 = avg.delta_f3'; avg.theta_f3=avg.theta_f3'; avg.alpha_f3=avg.alpha_f3'; avg.beta_f3 = avg.beta_f3'; avg.gamma_f3 = avg.gamma_f3';
    
    cca.Regional_f3 = [cca.delta_f3 cca.theta_f3 cca.alpha_f3 cca.beta_f3 cca.gamma_f3];
    avg.Regional_f3 = [avg.delta_f3 avg.theta_f3 avg.alpha_f3 avg.beta_f3 avg.gamma_f3];
    
    cca.RPL.delta_f3 = cca.delta_f3 ./ (cca.delta_f3 + cca.theta_f3 + cca.alpha_f3 + cca.beta_f3 + cca.gamma_f3) ;
    avg.RPL.delta_f3 = avg.delta_f3 ./ sum(avg.Regional_f3,2) ;
    cca.RPL.theta_f3 = cca.theta_f3 ./ (cca.delta_f3 + cca.theta_f3 + cca.alpha_f3 + cca.beta_f3 + cca.gamma_f3) ;
    avg.RPL.theta_f3 = avg.theta_f3 ./ sum(avg.Regional_f3,2) ;
    cca.RPL.alpha_f3 = cca.alpha_f3 ./ (cca.delta_f3 + cca.theta_f3 + cca.alpha_f3 + cca.beta_f3 + cca.gamma_f3) ;
    avg.RPL.alpha_f3 = avg.alpha_f3 ./ sum(avg.Regional_f3,2) ;
    cca.RPL.beta_f3 = cca.beta_f3 ./ (cca.delta_f3 + cca.theta_f3 + cca.alpha_f3 + cca.beta_f3 + cca.gamma_f3) ;
    avg.RPL.beta_f3 = avg.beta_f3 ./ sum(avg.Regional_f3,2) ;
    cca.RPL.gamma_f3 = cca.gamma_f3 ./ (cca.delta_f3 + cca.theta_f3 + cca.alpha_f3 + cca.beta_f3 + cca.gamma_f3) ;
    avg.RPL.gamma_f3 = avg.gamma_f3 ./ sum(avg.Regional_f3,2) ;    
    
    cca.RPL.Regional_f3 = [cca.RPL.delta_f3 cca.RPL.theta_f3 cca.RPL.alpha_f3 cca.RPL.beta_f3 cca.RPL.gamma_f3]; 
    avg.RPL.Regional_f3 = [avg.RPL.delta_f3 avg.RPL.theta_f3 avg.RPL.alpha_f3 avg.RPL.beta_f3 avg.RPL.gamma_f3];
    
    cca.z.Regional_f3 = [zscore(cca.delta_f3) zscore(cca.theta_f3) zscore(cca.alpha_f3) zscore(cca.beta_f3) zscore(cca.gamma_f3)];
    avg.z.Regional_f3 = [zscore(avg.delta_f3) zscore(avg.theta_f3) zscore(avg.alpha_f3) zscore(avg.beta_f3) zscore(avg.gamma_f3)];           
    
    
    [cca.delta_c1,avg.delta_c1] = Power_spectrum(channel_c1, [0.5 4], cnt.fs);
    [cca.theta_c1,avg.theta_c1] = Power_spectrum(channel_c1, [4 8], cnt.fs);
    [cca.alpha_c1,avg.alpha_c1] = Power_spectrum(channel_c1, [8 13], cnt.fs);
    [cca.beta_c1,avg.beta_c1] = Power_spectrum(channel_c1, [13 30], cnt.fs);
    [cca.gamma_c1,avg.gamma_c1] = Power_spectrum(channel_c1, [30 40], cnt.fs);
    avg.delta_c1 = avg.delta_c1'; avg.theta_c1=avg.theta_c1'; avg.alpha_c1=avg.alpha_c1'; avg.beta_c1 = avg.beta_c1'; avg.gamma_c1 = avg.gamma_c1';
    
    cca.Regional_c1 = [cca.delta_c1 cca.theta_c1 cca.alpha_c1 cca.beta_c1 cca.gamma_c1];
    avg.Regional_c1 = [avg.delta_c1 avg.theta_c1 avg.alpha_c1 avg.beta_c1 avg.gamma_c1];
    
    cca.RPL.delta_c1 = cca.delta_c1 ./ (cca.delta_c1 + cca.theta_c1 + cca.alpha_c1 + cca.beta_c1 + cca.gamma_c1) ;
    avg.RPL.delta_c1 = avg.delta_c1 ./ sum(avg.Regional_c1,2) ;
    cca.RPL.theta_c1 = cca.theta_c1 ./ (cca.delta_c1 + cca.theta_c1 + cca.alpha_c1 + cca.beta_c1 + cca.gamma_c1) ;
    avg.RPL.theta_c1 = avg.theta_c1 ./ sum(avg.Regional_c1,2) ;
    cca.RPL.alpha_c1 = cca.alpha_c1 ./ (cca.delta_c1 + cca.theta_c1 + cca.alpha_c1 + cca.beta_c1 + cca.gamma_c1) ;
    avg.RPL.alpha_c1 = avg.alpha_c1 ./ sum(avg.Regional_c1,2) ;
    cca.RPL.beta_c1 = cca.beta_c1 ./ (cca.delta_c1 + cca.theta_c1 + cca.alpha_c1 + cca.beta_c1 + cca.gamma_c1) ;
    avg.RPL.beta_c1 = avg.beta_c1 ./ sum(avg.Regional_c1,2) ;
    cca.RPL.gamma_c1 = cca.gamma_c1 ./ (cca.delta_c1 + cca.theta_c1 + cca.alpha_c1 + cca.beta_c1 + cca.gamma_c1) ;
    avg.RPL.gamma_c1 = avg.gamma_c1 ./ sum(avg.Regional_c1,2) ;     
    
    cca.RPL.Regional_c1 = [cca.RPL.delta_c1 cca.RPL.theta_c1 cca.RPL.alpha_c1 cca.RPL.beta_c1 cca.RPL.gamma_c1]; 
    avg.RPL.Regional_c1 = [avg.RPL.delta_c1 avg.RPL.theta_c1 avg.RPL.alpha_c1 avg.RPL.beta_c1 avg.RPL.gamma_c1];
    
    cca.z.Regional_c1 = [zscore(cca.delta_c1) zscore(cca.theta_c1) zscore(cca.alpha_c1) zscore(cca.beta_c1) zscore(cca.gamma_c1)];
    avg.z.Regional_c1 = [zscore(avg.delta_c1) zscore(avg.theta_c1) zscore(avg.alpha_c1) zscore(avg.beta_c1) zscore(avg.gamma_c1)];               
    
    [cca.delta_c2,avg.delta_c2] = Power_spectrum(channel_c2, [0.5 4], cnt.fs);
    [cca.theta_c2,avg.theta_c2] = Power_spectrum(channel_c2, [4 8], cnt.fs);
    [cca.alpha_c2,avg.alpha_c2] = Power_spectrum(channel_c2, [8 13], cnt.fs);
    [cca.beta_c2,avg.beta_c2] = Power_spectrum(channel_c2, [13 30], cnt.fs);
    [cca.gamma_c2,avg.gamma_c2] = Power_spectrum(channel_c2, [30 40], cnt.fs);
    avg.delta_c2 = avg.delta_c2'; avg.theta_c2=avg.theta_c2'; avg.alpha_c2=avg.alpha_c2'; avg.beta_c2 = avg.beta_c2'; avg.gamma_c2 = avg.gamma_c2';
    
    cca.Regional_c2 = [cca.delta_c2 cca.theta_c2 cca.alpha_c2 cca.beta_c2 cca.gamma_c2];
    avg.Regional_c2 = [avg.delta_c2 avg.theta_c2 avg.alpha_c2 avg.beta_c2 avg.gamma_c2];
    
    cca.RPL.delta_c2 = cca.delta_c2 ./ (cca.delta_c2 + cca.theta_c2 + cca.alpha_c2 + cca.beta_c2 + cca.gamma_c2) ;
    avg.RPL.delta_c2 = avg.delta_c2 ./ sum(avg.Regional_c2,2) ;
    cca.RPL.theta_c2 = cca.theta_c2 ./ (cca.delta_c2 + cca.theta_c2 + cca.alpha_c2 + cca.beta_c2 + cca.gamma_c2) ;
    avg.RPL.theta_c2 = avg.theta_c2 ./ sum(avg.Regional_c2,2) ;
    cca.RPL.alpha_c2 = cca.alpha_c2 ./ (cca.delta_c2 + cca.theta_c2 + cca.alpha_c2 + cca.beta_c2 + cca.gamma_c2) ;
    avg.RPL.alpha_c2 = avg.alpha_c2 ./ sum(avg.Regional_c2,2) ;
    cca.RPL.beta_c2 = cca.beta_c2 ./ (cca.delta_c2 + cca.theta_c2 + cca.alpha_c2 + cca.beta_c2 + cca.gamma_c2) ;
    avg.RPL.beta_c2 = avg.beta_c2 ./ sum(avg.Regional_c2,2) ;
    cca.RPL.gamma_c2 = cca.gamma_c2 ./ (cca.delta_c2 + cca.theta_c2 + cca.alpha_c2 + cca.beta_c2 + cca.gamma_c2) ;
    avg.RPL.gamma_c2 = avg.gamma_c2 ./ sum(avg.Regional_c2,2) ;     
    
    cca.RPL.Regional_c2 = [cca.RPL.delta_c2 cca.RPL.theta_c2 cca.RPL.alpha_c2 cca.RPL.beta_c2 cca.RPL.gamma_c2]; 
    avg.RPL.Regional_c2 = [avg.RPL.delta_c2 avg.RPL.theta_c2 avg.RPL.alpha_c2 avg.RPL.beta_c2 avg.RPL.gamma_c2];
    
    cca.z.Regional_c2 = [zscore(cca.delta_c2) zscore(cca.theta_c2) zscore(cca.alpha_c2) zscore(cca.beta_c2) zscore(cca.gamma_c2)];
    avg.z.Regional_c2 = [zscore(avg.delta_c2) zscore(avg.theta_c2) zscore(avg.alpha_c2) zscore(avg.beta_c2) zscore(avg.gamma_c2)];                   
    
    [cca.delta_c3,avg.delta_c3] = Power_spectrum(channel_c3, [0.5 4], cnt.fs);
    [cca.theta_c3,avg.theta_c3] = Power_spectrum(channel_c3, [4 8], cnt.fs);
    [cca.alpha_c3,avg.alpha_c3] = Power_spectrum(channel_c3, [8 13], cnt.fs);
    [cca.beta_c3,avg.beta_c3] = Power_spectrum(channel_c3, [13 30], cnt.fs);
    [cca.gamma_c3,avg.gamma_c3] = Power_spectrum(channel_c3, [30 40], cnt.fs);
    avg.delta_c3 = avg.delta_c3'; avg.theta_c3=avg.theta_c3'; avg.alpha_c3=avg.alpha_c3'; avg.beta_c3 = avg.beta_c3'; avg.gamma_c3 = avg.gamma_c3';
    
    cca.Regional_c3 = [cca.delta_c3 cca.theta_c3 cca.alpha_c3 cca.beta_c3 cca.gamma_c3];
    avg.Regional_c3 = [avg.delta_c3 avg.theta_c3 avg.alpha_c3 avg.beta_c3 avg.gamma_c3];
    
    cca.RPL.delta_c3 = cca.delta_c3 ./ (cca.delta_c3 + cca.theta_c3 + cca.alpha_c3 + cca.beta_c3 + cca.gamma_c3) ;
    avg.RPL.delta_c3 = avg.delta_c3 ./ sum(avg.Regional_c3,2) ;
    cca.RPL.theta_c3 = cca.theta_c3 ./ (cca.delta_c3 + cca.theta_c3 + cca.alpha_c3 + cca.beta_c3 + cca.gamma_c3) ;
    avg.RPL.theta_c3 = avg.theta_c3 ./ sum(avg.Regional_c3,2) ;
    cca.RPL.alpha_c3 = cca.alpha_c3 ./ (cca.delta_c3 + cca.theta_c3 + cca.alpha_c3 + cca.beta_c3 + cca.gamma_c3) ;
    avg.RPL.alpha_c3 = avg.alpha_c3 ./ sum(avg.Regional_c3,2) ;
    cca.RPL.beta_c3 = cca.beta_c3 ./ (cca.delta_c3 + cca.theta_c3 + cca.alpha_c3 + cca.beta_c3 + cca.gamma_c3) ;
    avg.RPL.beta_c3 = avg.beta_c3 ./ sum(avg.Regional_c3,2) ;
    cca.RPL.gamma_c3 = cca.gamma_c3 ./ (cca.delta_c3 + cca.theta_c3 + cca.alpha_c3 + cca.beta_c3 + cca.gamma_c3) ;
    avg.RPL.gamma_c3 = avg.gamma_c3 ./ sum(avg.Regional_c3,2) ;     
    
    cca.RPL.Regional_c3 = [cca.RPL.delta_c3 cca.RPL.theta_c3 cca.RPL.alpha_c3 cca.RPL.beta_c3 cca.RPL.gamma_c3]; 
    avg.RPL.Regional_c3 = [avg.RPL.delta_c3 avg.RPL.theta_c3 avg.RPL.alpha_c3 avg.RPL.beta_c3 avg.RPL.gamma_c3];
    
    cca.z.Regional_c3 = [zscore(cca.delta_c3) zscore(cca.theta_c3) zscore(cca.alpha_c3) zscore(cca.beta_c3) zscore(cca.gamma_c3)];
    avg.z.Regional_c3 = [zscore(avg.delta_c3) zscore(avg.theta_c3) zscore(avg.alpha_c3) zscore(avg.beta_c3) zscore(avg.gamma_c3)];                   
    
    [cca.delta_p1,avg.delta_p1] = Power_spectrum(channel_p1, [0.5 4], cnt.fs);
    [cca.theta_p1,avg.theta_p1] = Power_spectrum(channel_p1, [4 8], cnt.fs);
    [cca.alpha_p1,avg.alpha_p1] = Power_spectrum(channel_p1, [8 13], cnt.fs);
    [cca.beta_p1,avg.beta_p1] = Power_spectrum(channel_p1, [13 30], cnt.fs);
    [cca.gamma_p1,avg.gamma_p1] = Power_spectrum(channel_p1, [30 40], cnt.fs);
    avg.delta_p1 = avg.delta_p1'; avg.theta_p1=avg.theta_p1'; avg.alpha_p1=avg.alpha_p1'; avg.beta_p1 = avg.beta_p1'; avg.gamma_p1 = avg.gamma_p1';
    
    cca.Regional_p1 = [cca.delta_p1 cca.theta_p1 cca.alpha_p1 cca.beta_p1 cca.gamma_p1];
    avg.Regional_p1 = [avg.delta_p1 avg.theta_p1 avg.alpha_p1 avg.beta_p1 avg.gamma_p1];
    
    cca.RPL.delta_p1 = cca.delta_p1 ./ (cca.delta_p1 + cca.theta_p1 + cca.alpha_p1 + cca.beta_p1 + cca.gamma_p1) ;
    avg.RPL.delta_p1 = avg.delta_p1 ./ sum(avg.Regional_p1,2) ;
    cca.RPL.theta_p1 = cca.theta_p1 ./ (cca.delta_p1 + cca.theta_p1 + cca.alpha_p1 + cca.beta_p1 + cca.gamma_p1) ;
    avg.RPL.theta_p1 = avg.theta_p1 ./ sum(avg.Regional_p1,2) ;
    cca.RPL.alpha_p1 = cca.alpha_p1 ./ (cca.delta_p1 + cca.theta_p1 + cca.alpha_p1 + cca.beta_p1 + cca.gamma_p1) ;
    avg.RPL.alpha_p1 = avg.alpha_p1 ./ sum(avg.Regional_p1,2) ;
    cca.RPL.beta_p1 = cca.beta_p1 ./ (cca.delta_p1 + cca.theta_p1 + cca.alpha_p1 + cca.beta_p1 + cca.gamma_p1) ;
    avg.RPL.beta_p1 = avg.beta_p1 ./ sum(avg.Regional_p1,2) ;
    cca.RPL.gamma_p1 = cca.gamma_p1 ./ (cca.delta_p1 + cca.theta_p1 + cca.alpha_p1 + cca.beta_p1 + cca.gamma_p1) ;
    avg.RPL.gamma_p1 = avg.gamma_p1 ./ sum(avg.Regional_p1,2) ;     
    
    cca.RPL.Regional_p1 = [cca.RPL.delta_p1 cca.RPL.theta_p1 cca.RPL.alpha_p1 cca.RPL.beta_p1 cca.RPL.gamma_p1]; 
    avg.RPL.Regional_p1 = [avg.RPL.delta_p1 avg.RPL.theta_p1 avg.RPL.alpha_p1 avg.RPL.beta_p1 avg.RPL.gamma_p1];
    
    cca.z.Regional_p1 = [zscore(cca.delta_p1) zscore(cca.theta_p1) zscore(cca.alpha_p1) zscore(cca.beta_p1) zscore(cca.gamma_p1)];
    avg.z.Regional_p1 = [zscore(avg.delta_p1) zscore(avg.theta_p1) zscore(avg.alpha_p1) zscore(avg.beta_p1) zscore(avg.gamma_p1)];               
    
    [cca.delta_p2,avg.delta_p2] = Power_spectrum(channel_p2, [0.5 4], cnt.fs);
    [cca.theta_p2,avg.theta_p2] = Power_spectrum(channel_p2, [4 8], cnt.fs);
    [cca.alpha_p2,avg.alpha_p2] = Power_spectrum(channel_p2, [8 13], cnt.fs);
    [cca.beta_p2,avg.beta_p2] = Power_spectrum(channel_p2, [13 30], cnt.fs);
    [cca.gamma_p2,avg.gamma_p2] = Power_spectrum(channel_p2, [30 40], cnt.fs);
    avg.delta_p2 = avg.delta_p2'; avg.theta_p2=avg.theta_p2'; avg.alpha_p2=avg.alpha_p2'; avg.beta_p2 = avg.beta_p2'; avg.gamma_p2 = avg.gamma_p2';
    
    cca.Regional_p2 = [cca.delta_p2 cca.theta_p2 cca.alpha_p2 cca.beta_p2 cca.gamma_p2];
    avg.Regional_p2 = [avg.delta_p2 avg.theta_p2 avg.alpha_p2 avg.beta_p2 avg.gamma_p2];
    
    cca.RPL.delta_p2 = cca.delta_p2 ./ (cca.delta_p2 + cca.theta_p2 + cca.alpha_p2 + cca.beta_p2 + cca.gamma_p2) ;
    avg.RPL.delta_p2 = avg.delta_p2 ./ sum(avg.Regional_p2,2) ;
    cca.RPL.theta_p2 = cca.theta_p2 ./ (cca.delta_p2 + cca.theta_p2 + cca.alpha_p2 + cca.beta_p2 + cca.gamma_p2) ;
    avg.RPL.theta_p2 = avg.theta_p2 ./ sum(avg.Regional_p2,2) ;
    cca.RPL.alpha_p2 = cca.alpha_p2 ./ (cca.delta_p2 + cca.theta_p2 + cca.alpha_p2 + cca.beta_p2 + cca.gamma_p2) ;
    avg.RPL.alpha_p2 = avg.alpha_p2 ./ sum(avg.Regional_p2,2) ;
    cca.RPL.beta_p2 = cca.beta_p2 ./ (cca.delta_p2 + cca.theta_p2 + cca.alpha_p2 + cca.beta_p2 + cca.gamma_p2) ;
    avg.RPL.beta_p2 = avg.beta_p2 ./ sum(avg.Regional_p2,2) ;
    cca.RPL.gamma_p2 = cca.gamma_p2 ./ (cca.delta_p2 + cca.theta_p2 + cca.alpha_p2 + cca.beta_p2 + cca.gamma_p2) ;
    avg.RPL.gamma_p2 = avg.gamma_p2 ./ sum(avg.Regional_p2,2) ;     
    
    cca.RPL.Regional_p2 = [cca.RPL.delta_p2 cca.RPL.theta_p2 cca.RPL.alpha_p2 cca.RPL.beta_p2 cca.RPL.gamma_p2]; 
    avg.RPL.Regional_p2 = [avg.RPL.delta_p2 avg.RPL.theta_p2 avg.RPL.alpha_p2 avg.RPL.beta_p2 avg.RPL.gamma_p2];
    
    cca.z.Regional_p2 = [zscore(cca.delta_p2) zscore(cca.theta_p2) zscore(cca.alpha_p2) zscore(cca.beta_p2) zscore(cca.gamma_p2)];
    avg.z.Regional_p2 = [zscore(avg.delta_p2) zscore(avg.theta_p2) zscore(avg.alpha_p2) zscore(avg.beta_p2) zscore(avg.gamma_p2)];               
    
    
    [cca.delta_p3,avg.delta_p3] = Power_spectrum(channel_p3, [0.5 4], cnt.fs);
    [cca.theta_p3,avg.theta_p3] = Power_spectrum(channel_p3, [4 8], cnt.fs);
    [cca.alpha_p3,avg.alpha_p3] = Power_spectrum(channel_p3, [8 13], cnt.fs);
    [cca.beta_p3,avg.beta_p3] = Power_spectrum(channel_p3, [13 30], cnt.fs);
    [cca.gamma_p3,avg.gamma_p3] = Power_spectrum(channel_p3, [30 40], cnt.fs);
    avg.delta_p3 = avg.delta_p3'; avg.theta_p3=avg.theta_p3'; avg.alpha_p3=avg.alpha_p3'; avg.beta_p3 = avg.beta_p3'; avg.gamma_p3 = avg.gamma_p3';
    
    cca.Regional_p3 = [cca.delta_p3 cca.theta_p3 cca.alpha_p3 cca.beta_p3 cca.gamma_p3];
    avg.Regional_p3 = [avg.delta_p3 avg.theta_p3 avg.alpha_p3 avg.beta_p3 avg.gamma_p3];
    
    cca.RPL.delta_p3 = cca.delta_p3 ./ (cca.delta_p3 + cca.theta_p3 + cca.alpha_p3 + cca.beta_p3 + cca.gamma_p3) ;
    avg.RPL.delta_p3 = avg.delta_p3 ./ sum(avg.Regional_p3,2) ;
    cca.RPL.theta_p3 = cca.theta_p3 ./ (cca.delta_p3 + cca.theta_p3 + cca.alpha_p3 + cca.beta_p3 + cca.gamma_p3) ;
    avg.RPL.theta_p3 = avg.theta_p3 ./ sum(avg.Regional_p3,2) ;
    cca.RPL.alpha_p3 = cca.alpha_p3 ./ (cca.delta_p3 + cca.theta_p3 + cca.alpha_p3 + cca.beta_p3 + cca.gamma_p3) ;
    avg.RPL.alpha_p3 = avg.alpha_p3 ./ sum(avg.Regional_p3,2) ;
    cca.RPL.beta_p3 = cca.beta_p3 ./ (cca.delta_p3 + cca.theta_p3 + cca.alpha_p3 + cca.beta_p3 + cca.gamma_p3) ;
    avg.RPL.beta_p3 = avg.beta_p3 ./ sum(avg.Regional_p3,2) ;
    cca.RPL.gamma_p3 = cca.gamma_p3 ./ (cca.delta_p3 + cca.theta_p3 + cca.alpha_p3 + cca.beta_p3 + cca.gamma_p3) ;
    avg.RPL.gamma_p3 = avg.gamma_p3 ./ sum(avg.Regional_p3,2) ;     
    
    cca.RPL.Regional_p3 = [cca.RPL.delta_p3 cca.RPL.theta_p3 cca.RPL.alpha_p3 cca.RPL.beta_p3 cca.RPL.gamma_p3]; 
    avg.RPL.Regional_p3 = [avg.RPL.delta_p3 avg.RPL.theta_p3 avg.RPL.alpha_p3 avg.RPL.beta_p3 avg.RPL.gamma_p3];
    
    cca.z.Regional_p3 = [zscore(cca.delta_p3) zscore(cca.theta_p3) zscore(cca.alpha_p3) zscore(cca.beta_p3) zscore(cca.gamma_p3)];
    avg.z.Regional_p3 = [zscore(avg.delta_p3) zscore(avg.theta_p3) zscore(avg.alpha_p3) zscore(avg.beta_p3) zscore(avg.gamma_p3)];               
    
    
    [cca.delta_t1,avg.delta_t1] = Power_spectrum(channel_t1, [0.5 4], cnt.fs);
    [cca.theta_t1,avg.theta_t1] = Power_spectrum(channel_t1, [4 8], cnt.fs);
    [cca.alpha_t1,avg.alpha_t1] = Power_spectrum(channel_t1, [8 13], cnt.fs);
    [cca.beta_t1,avg.beta_t1] = Power_spectrum(channel_t1, [13 30], cnt.fs);
    [cca.gamma_t1,avg.gamma_t1] = Power_spectrum(channel_t1, [30 40], cnt.fs);
    avg.delta_t1 = avg.delta_t1'; avg.theta_t1=avg.theta_t1'; avg.alpha_t1=avg.alpha_t1'; avg.beta_t1 = avg.beta_t1'; avg.gamma_t1 = avg.gamma_t1';
    
    cca.Regional_t1 = [cca.delta_t1 cca.theta_t1 cca.alpha_t1 cca.beta_t1 cca.gamma_t1];
    avg.Regional_t1 = [avg.delta_t1 avg.theta_t1 avg.alpha_t1 avg.beta_t1 avg.gamma_t1];
    
    cca.RPL.delta_t1 = cca.delta_t1 ./ (cca.delta_t1 + cca.theta_t1 + cca.alpha_t1 + cca.beta_t1 + cca.gamma_t1) ;
    avg.RPL.delta_t1 = avg.delta_t1 ./ sum(avg.Regional_t1,2) ;
    cca.RPL.theta_t1 = cca.theta_t1 ./ (cca.delta_t1 + cca.theta_t1 + cca.alpha_t1 + cca.beta_t1 + cca.gamma_t1) ;
    avg.RPL.theta_t1 = avg.theta_t1 ./ sum(avg.Regional_t1,2) ;
    cca.RPL.alpha_t1 = cca.alpha_t1 ./ (cca.delta_t1 + cca.theta_t1 + cca.alpha_t1 + cca.beta_t1 + cca.gamma_t1) ;
    avg.RPL.alpha_t1 = avg.alpha_t1 ./ sum(avg.Regional_t1,2) ;
    cca.RPL.beta_t1 = cca.beta_t1 ./ (cca.delta_t1 + cca.theta_t1 + cca.alpha_t1 + cca.beta_t1 + cca.gamma_t1) ;
    avg.RPL.beta_t1 = avg.beta_t1 ./ sum(avg.Regional_t1,2) ;
    cca.RPL.gamma_t1 = cca.gamma_t1 ./ (cca.delta_t1 + cca.theta_t1 + cca.alpha_t1 + cca.beta_t1 + cca.gamma_t1) ;
    avg.RPL.gamma_t1 = avg.gamma_t1 ./ sum(avg.Regional_t1,2) ;     

    cca.RPL.Regional_t1 = [cca.RPL.delta_t1 cca.RPL.theta_t1 cca.RPL.alpha_t1 cca.RPL.beta_t1 cca.RPL.gamma_t1]; 
    avg.RPL.Regional_t1 = [avg.RPL.delta_t1 avg.RPL.theta_t1 avg.RPL.alpha_t1 avg.RPL.beta_t1 avg.RPL.gamma_t1];
    
    cca.z.Regional_t1 = [zscore(cca.delta_t1) zscore(cca.theta_t1) zscore(cca.alpha_t1) zscore(cca.beta_t1) zscore(cca.gamma_t1)];
    avg.z.Regional_t1 = [zscore(avg.delta_t1) zscore(avg.theta_t1) zscore(avg.alpha_t1) zscore(avg.beta_t1) zscore(avg.gamma_t1)];                   
    
    
    [cca.delta_t2,avg.delta_t2] = Power_spectrum(channel_t2, [0.5 4], cnt.fs);
    [cca.theta_t2,avg.theta_t2] = Power_spectrum(channel_t2, [4 8], cnt.fs);
    [cca.alpha_t2,avg.alpha_t2] = Power_spectrum(channel_t2, [8 13], cnt.fs);
    [cca.beta_t2,avg.beta_t2] = Power_spectrum(channel_t2, [13 30], cnt.fs);
    [cca.gamma_t2,avg.gamma_t2] = Power_spectrum(channel_t2, [30 40], cnt.fs);
    avg.delta_t2 = avg.delta_t2'; avg.theta_t2=avg.theta_t2'; avg.alpha_t2=avg.alpha_t2'; avg.beta_t2 = avg.beta_t2'; avg.gamma_t2 = avg.gamma_t2';
    
    cca.Regional_t2 = [cca.delta_t2 cca.theta_t2 cca.alpha_t2 cca.beta_t2 cca.gamma_t2];
    avg.Regional_t2 = [avg.delta_t2 avg.theta_t2 avg.alpha_t2 avg.beta_t2 avg.gamma_t2];
    
    cca.RPL.delta_t2 = cca.delta_t2 ./ (cca.delta_t2 + cca.theta_t2 + cca.alpha_t2 + cca.beta_t2 + cca.gamma_t2) ;
    avg.RPL.delta_t2 = avg.delta_t2 ./ sum(avg.Regional_t2,2) ;
    cca.RPL.theta_t2 = cca.theta_t2 ./ (cca.delta_t2 + cca.theta_t2 + cca.alpha_t2 + cca.beta_t2 + cca.gamma_t2) ;
    avg.RPL.theta_t2 = avg.theta_t2 ./ sum(avg.Regional_t2,2) ;
    cca.RPL.alpha_t2 = cca.alpha_t2 ./ (cca.delta_t2 + cca.theta_t2 + cca.alpha_t2 + cca.beta_t2 + cca.gamma_t2) ;
    avg.RPL.alpha_t2 = avg.alpha_t2 ./ sum(avg.Regional_t2,2) ;
    cca.RPL.beta_t2 = cca.beta_t2 ./ (cca.delta_t2 + cca.theta_t2 + cca.alpha_t2 + cca.beta_t2 + cca.gamma_t2) ;
    avg.RPL.beta_t2 = avg.beta_t2 ./ sum(avg.Regional_t2,2) ;
    cca.RPL.gamma_t2 = cca.gamma_t2 ./ (cca.delta_t2 + cca.theta_t2 + cca.alpha_t2 + cca.beta_t2 + cca.gamma_t2) ;
    avg.RPL.gamma_t2 = avg.gamma_t2 ./ sum(avg.Regional_t2,2) ;     
    
    cca.RPL.Regional_t2 = [cca.RPL.delta_t2 cca.RPL.theta_t2 cca.RPL.alpha_t2 cca.RPL.beta_t2 cca.RPL.gamma_t2]; 
    avg.RPL.Regional_t2 = [avg.RPL.delta_t2 avg.RPL.theta_t2 avg.RPL.alpha_t2 avg.RPL.beta_t2 avg.RPL.gamma_t2];
    
    cca.z.Regional_t2 = [zscore(cca.delta_t2) zscore(cca.theta_t2) zscore(cca.alpha_t2) zscore(cca.beta_t2) zscore(cca.gamma_t2)];
    avg.z.Regional_t2 = [zscore(avg.delta_t2) zscore(avg.theta_t2) zscore(avg.alpha_t2) zscore(avg.beta_t2) zscore(avg.gamma_t2)];                   
    
    
    
    
    
    [cca.delta_o1,avg.delta_o1] = Power_spectrum(channel_o1, [0.5 4], cnt.fs);
    [cca.theta_o1,avg.theta_o1] = Power_spectrum(channel_o1, [4 8], cnt.fs);
    [cca.alpha_o1,avg.alpha_o1] = Power_spectrum(channel_o1, [8 13], cnt.fs);
    [cca.beta_o1,avg.beta_o1] = Power_spectrum(channel_o1, [13 30], cnt.fs);
    [cca.gamma_o1,avg.gamma_o1] = Power_spectrum(channel_o1, [30 40], cnt.fs);
    avg.delta_o1 = avg.delta_o1'; avg.theta_o1=avg.theta_o1'; avg.alpha_o1=avg.alpha_o1'; avg.beta_o1 = avg.beta_o1'; avg.gamma_o1 = avg.gamma_o1';
    
    cca.Regional_o1 = [cca.delta_o1 cca.theta_o1 cca.alpha_o1 cca.beta_o1 cca.gamma_o1];
    avg.Regional_o1 = [avg.delta_o1 avg.theta_o1 avg.alpha_o1 avg.beta_o1 avg.gamma_o1];

    cca.RPL.delta_o1 = cca.delta_o1 ./ (cca.delta_o1 + cca.theta_o1 + cca.alpha_o1 + cca.beta_o1 + cca.gamma_o1) ;
    avg.RPL.delta_o1 = avg.delta_o1 ./ sum(avg.Regional_o1,2) ;
    cca.RPL.theta_o1 = cca.theta_o1 ./ (cca.delta_o1 + cca.theta_o1 + cca.alpha_o1 + cca.beta_o1 + cca.gamma_o1) ;
    avg.RPL.theta_o1 = avg.theta_o1 ./ sum(avg.Regional_o1,2) ;
    cca.RPL.alpha_o1 = cca.alpha_o1 ./ (cca.delta_o1 + cca.theta_o1 + cca.alpha_o1 + cca.beta_o1 + cca.gamma_o1) ;
    avg.RPL.alpha_o1 = avg.alpha_o1 ./ sum(avg.Regional_o1,2) ;
    cca.RPL.beta_o1 = cca.beta_o1 ./ (cca.delta_o1 + cca.theta_o1 + cca.alpha_o1 + cca.beta_o1 + cca.gamma_o1) ;
    avg.RPL.beta_o1 = avg.beta_o1 ./ sum(avg.Regional_o1,2) ;
    cca.RPL.gamma_o1 = cca.gamma_o1 ./ (cca.delta_o1 + cca.theta_o1 + cca.alpha_o1 + cca.beta_o1 + cca.gamma_o1) ;
    avg.RPL.gamma_o1 = avg.gamma_o1 ./ sum(avg.Regional_o1,2) ;     

    cca.RPL.Regional_o1 = [cca.RPL.delta_o1 cca.RPL.theta_o1 cca.RPL.alpha_o1 cca.RPL.beta_o1 cca.RPL.gamma_o1]; 
    avg.RPL.Regional_o1 = [avg.RPL.delta_o1 avg.RPL.theta_o1 avg.RPL.alpha_o1 avg.RPL.beta_o1 avg.RPL.gamma_o1];
    
    cca.z.Regional_o1 = [zscore(cca.delta_o1) zscore(cca.theta_o1) zscore(cca.alpha_o1) zscore(cca.beta_o1) zscore(cca.gamma_o1)];
    avg.z.Regional_o1 = [zscore(avg.delta_o1) zscore(avg.theta_o1) zscore(avg.alpha_o1) zscore(avg.beta_o1) zscore(avg.gamma_o1)];                   

    
    
    
    [cca.delta_o2,avg.delta_o2] = Power_spectrum(channel_o2, [0.5 4], cnt.fs);
    [cca.theta_o2,avg.theta_o2] = Power_spectrum(channel_o2, [4 8], cnt.fs);
    [cca.alpha_o2,avg.alpha_o2] = Power_spectrum(channel_o2, [8 13], cnt.fs);
    [cca.beta_o2,avg.beta_o2] = Power_spectrum(channel_o2, [13 30], cnt.fs);
    [cca.gamma_o2,avg.gamma_o2] = Power_spectrum(channel_o2, [30 40], cnt.fs);
    avg.delta_o2 = avg.delta_o2'; avg.theta_o2=avg.theta_o2'; avg.alpha_o2=avg.alpha_o2'; avg.beta_o2 = avg.beta_o2'; avg.gamma_o2 = avg.gamma_o2';
    
    cca.Regional_o2 = [cca.delta_o2 cca.theta_o2 cca.alpha_o2 cca.beta_o2 cca.gamma_o2];
    avg.Regional_o2 = [avg.delta_o2 avg.theta_o2 avg.alpha_o2 avg.beta_o2 avg.gamma_o2];
    
    cca.RPL.delta_o2 = cca.delta_o2 ./ (cca.delta_o2 + cca.theta_o2 + cca.alpha_o2 + cca.beta_o2 + cca.gamma_o2) ;
    avg.RPL.delta_o2 = avg.delta_o2 ./ sum(avg.Regional_o2,2) ;
    cca.RPL.theta_o2 = cca.theta_o2 ./ (cca.delta_o2 + cca.theta_o2 + cca.alpha_o2 + cca.beta_o2 + cca.gamma_o2) ;
    avg.RPL.theta_o2 = avg.theta_o2 ./ sum(avg.Regional_o2,2) ;
    cca.RPL.alpha_o2 = cca.alpha_o2 ./ (cca.delta_o2 + cca.theta_o2 + cca.alpha_o2 + cca.beta_o2 + cca.gamma_o2) ;
    avg.RPL.alpha_o2 = avg.alpha_o2 ./ sum(avg.Regional_o2,2) ;
    cca.RPL.beta_o2 = cca.beta_o2 ./ (cca.delta_o2 + cca.theta_o2 + cca.alpha_o2 + cca.beta_o2 + cca.gamma_o2) ;
    avg.RPL.beta_o2 = avg.beta_o2 ./ sum(avg.Regional_o2,2) ;
    cca.RPL.gamma_o2 = cca.gamma_o2 ./ (cca.delta_o2 + cca.theta_o2 + cca.alpha_o2 + cca.beta_o2 + cca.gamma_o2) ;
    avg.RPL.gamma_o2 = avg.gamma_o2 ./ sum(avg.Regional_o2,2) ;     

    cca.RPL.Regional_o2 = [cca.RPL.delta_o2 cca.RPL.theta_o2 cca.RPL.alpha_o2 cca.RPL.beta_o2 cca.RPL.gamma_o2]; 
    avg.RPL.Regional_o2 = [avg.RPL.delta_o2 avg.RPL.theta_o2 avg.RPL.alpha_o2 avg.RPL.beta_o2 avg.RPL.gamma_o2];
        
    cca.z.Regional_o2 = [zscore(cca.delta_o2) zscore(cca.theta_o2) zscore(cca.alpha_o2) zscore(cca.beta_o2) zscore(cca.gamma_o2)];
    avg.z.Regional_o2 = [zscore(avg.delta_o2) zscore(avg.theta_o2) zscore(avg.alpha_o2) zscore(avg.beta_o2) zscore(avg.gamma_o2)];                   
    
    
    
    [cca.delta_o3,avg.delta_o3] = Power_spectrum(channel_o3, [0.5 4], cnt.fs);
    [cca.theta_o3,avg.theta_o3] = Power_spectrum(channel_o3, [4 8], cnt.fs);
    [cca.alpha_o3,avg.alpha_o3] = Power_spectrum(channel_o3, [8 13], cnt.fs);
    [cca.beta_o3,avg.beta_o3] = Power_spectrum(channel_o3, [13 30], cnt.fs);
    [cca.gamma_o3,avg.gamma_o3] = Power_spectrum(channel_o3, [30 40], cnt.fs);
    avg.delta_o3 = avg.delta_o3'; avg.theta_o3=avg.theta_o3'; avg.alpha_o3=avg.alpha_o3'; avg.beta_o3 = avg.beta_o3'; avg.gamma_o3 = avg.gamma_o3';
    
    cca.Regional_o3 = [cca.delta_o3 cca.theta_o3 cca.alpha_o3 cca.beta_o3 cca.gamma_o3];
    avg.Regional_o3 = [avg.delta_o3 avg.theta_o3 avg.alpha_o3 avg.beta_o3 avg.gamma_o3];

    cca.RPL.delta_o3 = cca.delta_o3 ./ (cca.delta_o3 + cca.theta_o3 + cca.alpha_o3 + cca.beta_o3 + cca.gamma_o3) ;
    avg.RPL.delta_o3 = avg.delta_o3 ./ sum(avg.Regional_o3,2) ;
    cca.RPL.theta_o3 = cca.theta_o3 ./ (cca.delta_o3 + cca.theta_o3 + cca.alpha_o3 + cca.beta_o3 + cca.gamma_o3) ;
    avg.RPL.theta_o3 = avg.theta_o3 ./ sum(avg.Regional_o3,2) ;
    cca.RPL.alpha_o3 = cca.alpha_o3 ./ (cca.delta_o3 + cca.theta_o3 + cca.alpha_o3 + cca.beta_o3 + cca.gamma_o3) ;
    avg.RPL.alpha_o3 = avg.alpha_o3 ./ sum(avg.Regional_o3,2) ;
    cca.RPL.beta_o3 = cca.beta_o3 ./ (cca.delta_o3 + cca.theta_o3 + cca.alpha_o3 + cca.beta_o3 + cca.gamma_o3) ;
    avg.RPL.beta_o3 = avg.beta_o3 ./ sum(avg.Regional_o3,2) ;
    cca.RPL.gamma_o3 = cca.gamma_o3 ./ (cca.delta_o3 + cca.theta_o3 + cca.alpha_o3 + cca.beta_o3 + cca.gamma_o3) ;
    avg.RPL.gamma_o3 = avg.gamma_o3 ./ sum(avg.Regional_o3,2) ;     

    cca.RPL.Regional_o3 = [cca.RPL.delta_o3 cca.RPL.theta_o3 cca.RPL.alpha_o3 cca.RPL.beta_o3 cca.RPL.gamma_o3]; 
    avg.RPL.Regional_o3 = [avg.RPL.delta_o3 avg.RPL.theta_o3 avg.RPL.alpha_o3 avg.RPL.beta_o3 avg.RPL.gamma_o3];
        
    cca.z.Regional_o3 = [zscore(cca.delta_o3) zscore(cca.theta_o3) zscore(cca.alpha_o3) zscore(cca.beta_o3) zscore(cca.gamma_o3)];
    avg.z.Regional_o3 = [zscore(avg.delta_o3) zscore(avg.theta_o3) zscore(avg.alpha_o3) zscore(avg.beta_o3) zscore(avg.gamma_o3)];                   
    
    
    %% Statistical Analysis
%         lowIdx = find(kss==min(kss)):1:find(kss==min(kss))+200;
%         lowIdx = lowIdx';
%         highIdx = find(kss==max(kss))-200:1:find(kss==max(kss));
%         highIdx = highIdx';
    
    %         최소+1, 최대-1
    lowIdx = min(kss)<kss & min(kss)+1>kss;
    highIdx = max(kss)>kss & max(kss)-1<kss;
    
    % PSD T-TEST 분석
    [h, p, ci, stats] = ttest2(avg.delta(lowIdx), avg.delta(highIdx));
    p_delta(s) = p;
    [h, p, ci, stats] = ttest2(avg.theta(lowIdx), avg.theta(highIdx));
    p_theta(s) = p;
    [h, p, ci, stats] = ttest2(avg.alpha(lowIdx), avg.alpha(highIdx));
    p_alpha(s) = p;
    [h, p, ci, stats] = ttest2(avg.beta(lowIdx), avg.beta(highIdx));
    p_beta(s) = p;
    [h, p, ci, stats] = ttest2(avg.gamma(lowIdx), avg.gamma(highIdx));
    p_gamma(s) = p;
    
    p_analysis(s,1:5) = [p_delta(s), p_theta(s), p_alpha(s), p_beta(s), p_gamma(s)];
    p_analysis_min(s) = find(p_analysis(s,1:5)==min(p_analysis(s,1:5)));
    
    Result.ttest.total(s,:) = p_analysis(s,1:5);
    
    % RPL T-TEST 분석
    [h, p, ci, stats] = ttest2(avg.RPL.delta(lowIdx), avg.RPL.delta(highIdx));
    p_delta(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.theta(lowIdx), avg.RPL.theta(highIdx));
    p_theta(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.alpha(lowIdx), avg.RPL.alpha(highIdx));
    p_alpha(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.beta(lowIdx), avg.RPL.beta(highIdx));
    p_beta(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.gamma(lowIdx), avg.RPL.gamma(highIdx));
    p_gamma(s) = p;
    
    p_analysis_RPL(s,1:5) = [p_delta(s), p_theta(s), p_alpha(s), p_beta(s), p_gamma(s)];
    p_analysis_RPL_min(s) = find(p_analysis_RPL(s,1:5)==min(p_analysis_RPL(s,1:5)));
    
    Result.ttest.total_RPL(s,:) = p_analysis_RPL(s,1:5);    
    
    
    % z-score T-TEST 분석
    [h, p, ci, stats] = ttest2(avg.z.PSD(lowIdx,1), avg.z.PSD(highIdx,1));
    p_delta(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.PSD(lowIdx,2), avg.z.PSD(lowIdx,2));
    p_theta(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.PSD(lowIdx,3), avg.z.PSD(lowIdx,3));
    p_alpha(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.PSD(lowIdx,4), avg.z.PSD(lowIdx,4));
    p_beta(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.PSD(lowIdx,5), avg.z.PSD(lowIdx,5));
    p_gamma(s) = p;
    
    p_analysis_z(s,1:5) = [p_delta(s), p_theta(s), p_alpha(s), p_beta(s), p_gamma(s)];
    p_analysis_z_min(s) = find(p_analysis_z(s,1:5)==min(p_analysis_z(s,1:5)));
    
    Result.ttest.total_z(s,:) = p_analysis_z(s,1:5);    
    
    
    
    % Regional PSD T-TEST 분석
    [h, p, ci, stats] = ttest2(avg.delta_f(lowIdx), avg.delta_f(highIdx));
    p_delta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.theta_f(lowIdx), avg.theta_f(highIdx));
    p_theta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.alpha_f(lowIdx), avg.alpha_f(highIdx));
    p_alpha_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.beta_f(lowIdx), avg.beta_f(highIdx));
    p_beta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.gamma_f(lowIdx), avg.gamma_f(highIdx));
    p_gamma_f(s) = p;
    
    p_analysis_f(s,1:5) = [p_delta_f(s), p_theta_f(s), p_alpha_f(s), p_beta_f(s), p_gamma_f(s)];
    p_analysis_f_min(s) = find(p_analysis_f(s,1:5)==min(p_analysis_f(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.delta_c(lowIdx), avg.delta_c(highIdx));
    p_delta_c(s) = p;
    [h, p, ci, stats] = ttest2(avg.theta_c(lowIdx), avg.theta_c(highIdx));
    p_theta_c(s) = p;
    [h, p, ci, stats] = ttest2(avg.alpha_c(lowIdx), avg.alpha_c(highIdx));
    p_alpha_c(s) = p;
    [h, p, ci, stats] = ttest2(avg.beta_c(lowIdx), avg.beta_c(highIdx));
    p_beta_c(s) = p;
    [h, p, ci, stats] = ttest2(avg.gamma_c(lowIdx), avg.gamma_c(highIdx));
    p_gamma_c(s) = p;
    
    p_analysis_c(s,1:5) = [p_delta_c(s), p_theta_c(s), p_alpha_c(s), p_beta_c(s), p_gamma_c(s)];
    p_analysis_c_min(s) = find(p_analysis_c(s,1:5)==min(p_analysis_c(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.delta_p(lowIdx), avg.delta_p(highIdx));
    p_delta_p(s) = p;
    [h, p, ci, stats] = ttest2(avg.theta_p(lowIdx), avg.theta_p(highIdx));
    p_theta_p(s) = p;
    [h, p, ci, stats] = ttest2(avg.alpha_p(lowIdx), avg.alpha_p(highIdx));
    p_alpha_p(s) = p;
    [h, p, ci, stats] = ttest2(avg.beta_p(lowIdx), avg.beta_p(highIdx));
    p_beta_p(s) = p;
    [h, p, ci, stats] = ttest2(avg.gamma_p(lowIdx), avg.gamma_p(highIdx));
    p_gamma_p(s) = p;
    
    p_analysis_p(s,1:5) = [p_delta_p(s), p_theta_p(s), p_alpha_p(s), p_beta_p(s), p_gamma_p(s)];
    p_analysis_p_min(s) = find(p_analysis_p(s,1:5)==min(p_analysis_p(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.delta_t(lowIdx), avg.delta_t(highIdx));
    p_delta_t(s) = p;
    [h, p, ci, stats] = ttest2(avg.theta_t(lowIdx), avg.theta_t(highIdx));
    p_theta_t(s) = p;
    [h, p, ci, stats] = ttest2(avg.alpha_t(lowIdx), avg.alpha_t(highIdx));
    p_alpha_t(s) = p;
    [h, p, ci, stats] = ttest2(avg.beta_t(lowIdx), avg.beta_t(highIdx));
    p_beta_t(s) = p;
    [h, p, ci, stats] = ttest2(avg.gamma_t(lowIdx), avg.gamma_t(highIdx));
    p_gamma_t(s) = p;
    
    p_analysis_t(s,1:5) = [p_delta_t(s), p_theta_t(s), p_alpha_t(s), p_beta_t(s), p_gamma_t(s)];
    p_analysis_t_min(s) = find(p_analysis_t(s,1:5)==min(p_analysis_t(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.delta_o(lowIdx), avg.delta_o(highIdx));
    p_delta_o(s) = p;
    [h, p, ci, stats] = ttest2(avg.theta_o(lowIdx), avg.theta_o(highIdx));
    p_theta_o(s) = p;
    [h, p, ci, stats] = ttest2(avg.alpha_o(lowIdx), avg.alpha_o(highIdx));
    p_alpha_o(s) = p;
    [h, p, ci, stats] = ttest2(avg.beta_o(lowIdx), avg.beta_o(highIdx));
    p_beta_o(s) = p;
    [h, p, ci, stats] = ttest2(avg.gamma_o(lowIdx), avg.gamma_o(highIdx));
    p_gamma_o(s) = p;
    
    p_analysis_o(s,1:5) = [p_delta_o(s), p_theta_o(s), p_alpha_o(s), p_beta_o(s), p_gamma_o(s)];
    p_analysis_o_min(s) = find(p_analysis_o(s,1:5)==min(p_analysis_o(s,1:5)));
    
    Result.ttest.regional(s,:) = [p_analysis_f(s,:) p_analysis_c(s,:) p_analysis_p(s,:) p_analysis_t(s,:) p_analysis_o(s,:)];

    
    
    [h, p, ci, stats] = ttest2(avg.RPL.delta_f(lowIdx), avg.RPL.delta_f(highIdx));
    p_delta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.theta_f(lowIdx), avg.RPL.theta_f(highIdx));
    p_theta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.alpha_f(lowIdx), avg.RPL.alpha_f(highIdx));
    p_alpha_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.beta_f(lowIdx), avg.RPL.beta_f(highIdx));
    p_beta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.gamma_f(lowIdx), avg.RPL.gamma_f(highIdx));
    p_gamma_f(s) = p;
    
    p_analysis_RPL_f(s,1:5) = [p_delta_f(s), p_theta_f(s), p_alpha_f(s), p_beta_f(s), p_gamma_f(s)];
    p_analysis_RPL_f_min(s) = find(p_analysis_RPL_f(s,1:5)==min(p_analysis_RPL_f(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.RPL.delta_c(lowIdx), avg.RPL.delta_c(highIdx));
    p_delta_c(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.theta_c(lowIdx), avg.RPL.theta_c(highIdx));
    p_theta_c(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.alpha_c(lowIdx), avg.RPL.alpha_c(highIdx));
    p_alpha_c(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.beta_c(lowIdx), avg.RPL.beta_c(highIdx));
    p_beta_c(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.gamma_c(lowIdx), avg.RPL.gamma_c(highIdx));
    p_gamma_c(s) = p;
    
    p_analysis_RPL_c(s,1:5) = [p_delta_c(s), p_theta_c(s), p_alpha_c(s), p_beta_c(s), p_gamma_c(s)];
    p_analysis_RPL_c_min(s) = find(p_analysis_RPL_c(s,1:5)==min(p_analysis_RPL_c(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.RPL.delta_p(lowIdx), avg.RPL.delta_p(highIdx));
    p_delta_p(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.theta_p(lowIdx), avg.RPL.theta_p(highIdx));
    p_theta_p(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.alpha_p(lowIdx), avg.RPL.alpha_p(highIdx));
    p_alpha_p(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.beta_p(lowIdx), avg.RPL.beta_p(highIdx));
    p_beta_p(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.gamma_p(lowIdx), avg.RPL.gamma_p(highIdx));
    p_gamma_p(s) = p;
    
    p_analysis_RPL_p(s,1:5) = [p_delta_p(s), p_theta_p(s), p_alpha_p(s), p_beta_p(s), p_gamma_p(s)];
    p_analysis_RPL_p_min(s) = find(p_analysis_RPL_p(s,1:5)==min(p_analysis_RPL_p(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.RPL.delta_t(lowIdx), avg.RPL.delta_t(highIdx));
    p_delta_t(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.theta_t(lowIdx), avg.RPL.theta_t(highIdx));
    p_theta_t(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.alpha_t(lowIdx), avg.RPL.alpha_t(highIdx));
    p_alpha_t(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.beta_t(lowIdx), avg.RPL.beta_t(highIdx));
    p_beta_t(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.gamma_t(lowIdx), avg.RPL.gamma_t(highIdx));
    p_gamma_t(s) = p;
    
    p_analysis_RPL_t(s,1:5) = [p_delta_t(s), p_theta_t(s), p_alpha_t(s), p_beta_t(s), p_gamma_t(s)];
    p_analysis_RPL_t_min(s) = find(p_analysis_RPL_t(s,1:5)==min(p_analysis_RPL_t(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.RPL.delta_o(lowIdx), avg.RPL.delta_o(highIdx));
    p_delta_o(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.theta_o(lowIdx), avg.RPL.theta_o(highIdx));
    p_theta_o(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.alpha_o(lowIdx), avg.RPL.alpha_o(highIdx));
    p_alpha_o(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.beta_o(lowIdx), avg.RPL.beta_o(highIdx));
    p_beta_o(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.gamma_o(lowIdx), avg.RPL.gamma_o(highIdx));
    p_gamma_o(s) = p;
    
    p_analysis_RPL_o(s,1:5) = [p_delta_o(s), p_theta_o(s), p_alpha_o(s), p_beta_o(s), p_gamma_o(s)];
    p_analysis_RPL_o_min(s) = find(p_analysis_RPL_o(s,1:5)==min(p_analysis_RPL_o(s,1:5)));
    
    Result.ttest.regional_RPL(s,:) = [p_analysis_RPL_f(s,:) p_analysis_RPL_c(s,:) p_analysis_RPL_p(s,:) p_analysis_RPL_t(s,:) p_analysis_RPL_o(s,:)];

    
    
    [h, p, ci, stats] = ttest2(avg.z.Regional_f(lowIdx,1), avg.z.Regional_f(highIdx,1));
    p_delta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_f(lowIdx,2), avg.z.Regional_f(highIdx,2));
    p_theta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_f(lowIdx,3), avg.z.Regional_f(highIdx,3));
    p_alpha_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_f(lowIdx,4), avg.z.Regional_f(highIdx,4));
    p_beta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_f(lowIdx,5), avg.z.Regional_f(highIdx,5));
    p_gamma_f(s) = p;
    
    p_analysis_z_f(s,1:5) = [p_delta_f(s), p_theta_f(s), p_alpha_f(s), p_beta_f(s), p_gamma_f(s)];
    p_analysis_z_f_min(s) = find(p_analysis_z_f(s,1:5)==min(p_analysis_z_f(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.z.Regional_c(lowIdx,1), avg.z.Regional_c(highIdx,1));
    p_delta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_c(lowIdx,2), avg.z.Regional_c(highIdx,2));
    p_theta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_c(lowIdx,3), avg.z.Regional_c(highIdx,3));
    p_alpha_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_c(lowIdx,4), avg.z.Regional_c(highIdx,4));
    p_beta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_c(lowIdx,5), avg.z.Regional_c(highIdx,5));
    p_gamma_f(s) = p;
    
    p_analysis_z_c(s,1:5) = [p_delta_c(s), p_theta_c(s), p_alpha_c(s), p_beta_c(s), p_gamma_c(s)];
    p_analysis_z_c_min(s) = find(p_analysis_z_c(s,1:5)==min(p_analysis_z_c(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.z.Regional_p(lowIdx,1), avg.z.Regional_p(highIdx,1));
    p_delta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_p(lowIdx,2), avg.z.Regional_p(highIdx,2));
    p_theta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_p(lowIdx,3), avg.z.Regional_p(highIdx,3));
    p_alpha_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_p(lowIdx,4), avg.z.Regional_p(highIdx,4));
    p_beta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_p(lowIdx,5), avg.z.Regional_p(highIdx,5));
    p_gamma_f(s) = p;
    
    p_analysis_z_p(s,1:5) = [p_delta_p(s), p_theta_p(s), p_alpha_p(s), p_beta_p(s), p_gamma_p(s)];
    p_analysis_z_p_min(s) = find(p_analysis_z_p(s,1:5)==min(p_analysis_z_p(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.z.Regional_t(lowIdx,1), avg.z.Regional_t(highIdx,1));
    p_delta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_t(lowIdx,2), avg.z.Regional_t(highIdx,2));
    p_theta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_t(lowIdx,3), avg.z.Regional_t(highIdx,3));
    p_alpha_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_t(lowIdx,4), avg.z.Regional_t(highIdx,4));
    p_beta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_t(lowIdx,5), avg.z.Regional_t(highIdx,5));
    p_gamma_f(s) = p;
    
    p_analysis_z_t(s,1:5) = [p_delta_t(s), p_theta_t(s), p_alpha_t(s), p_beta_t(s), p_gamma_t(s)];
    p_analysis_z_t_min(s) = find(p_analysis_z_t(s,1:5)==min(p_analysis_z_t(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.z.Regional_o(lowIdx,1), avg.z.Regional_o(highIdx,1));
    p_delta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_o(lowIdx,2), avg.z.Regional_o(highIdx,2));
    p_theta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_o(lowIdx,3), avg.z.Regional_o(highIdx,3));
    p_alpha_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_o(lowIdx,4), avg.z.Regional_o(highIdx,4));
    p_beta_f(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_o(lowIdx,5), avg.z.Regional_o(highIdx,5));
    p_gamma_f(s) = p;
    
    p_analysis_z_o(s,1:5) = [p_delta_o(s), p_theta_o(s), p_alpha_o(s), p_beta_o(s), p_gamma_o(s)];
    p_analysis_z_o_min(s) = find(p_analysis_z_o(s,1:5)==min(p_analysis_z_o(s,1:5)));
    
    Result.ttest.regional_z(s,:) = [p_analysis_f(s,:) p_analysis_c(s,:) p_analysis_p(s,:) p_analysis_t(s,:) p_analysis_o(s,:)];

    
    
    
    
    
    % Regional Regional T-TEST 분석
    [h, p, ci, stats] = ttest2(avg.delta_f1(lowIdx), avg.delta_f1(highIdx));
    p_delta_f1(s) = p;
    [h, p, ci, stats] = ttest2(avg.theta_f1(lowIdx), avg.theta_f1(highIdx));
    p_theta_f1(s) = p;
    [h, p, ci, stats] = ttest2(avg.alpha_f1(lowIdx), avg.alpha_f1(highIdx));
    p_alpha_f1(s) = p;
    [h, p, ci, stats] = ttest2(avg.beta_f1(lowIdx), avg.beta_f1(highIdx));
    p_beta_f1(s) = p;
    [h, p, ci, stats] = ttest2(avg.gamma_f1(lowIdx), avg.gamma_f1(highIdx));
    p_gamma_f1(s) = p;
    
    p_analysis_f1(s,1:5) = [p_delta_f1(s), p_theta_f1(s), p_alpha_f1(s), p_beta_f1(s), p_gamma_f1(s)];
    p_analysis_f1_min(s) = find(p_analysis_f1(s,1:5)==min(p_analysis_f1(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.delta_f2(lowIdx), avg.delta_f2(highIdx));
    p_delta_f2(s) = p;
    [h, p, ci, stats] = ttest2(avg.theta_f2(lowIdx), avg.theta_f2(highIdx));
    p_theta_f2(s) = p;
    [h, p, ci, stats] = ttest2(avg.alpha_f2(lowIdx), avg.alpha_f2(highIdx));
    p_alpha_f2(s) = p;
    [h, p, ci, stats] = ttest2(avg.beta_f2(lowIdx), avg.beta_f2(highIdx));
    p_beta_f2(s) = p;
    [h, p, ci, stats] = ttest2(avg.gamma_f2(lowIdx), avg.gamma_f2(highIdx));
    p_gamma_f2(s) = p;
    
    p_analysis_f2(s,1:5) = [p_delta_f2(s), p_theta_f2(s), p_alpha_f2(s), p_beta_f2(s), p_gamma_f2(s)];
    p_analysis_f2_min(s) = find(p_analysis_f2(s,1:5)==min(p_analysis_f2(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.delta_f3(lowIdx), avg.delta_f3(highIdx));
    p_delta_f3(s) = p;
    [h, p, ci, stats] = ttest2(avg.theta_f3(lowIdx), avg.theta_f3(highIdx));
    p_theta_f3(s) = p;
    [h, p, ci, stats] = ttest2(avg.alpha_f3(lowIdx), avg.alpha_f3(highIdx));
    p_alpha_f3(s) = p;
    [h, p, ci, stats] = ttest2(avg.beta_f3(lowIdx), avg.beta_f3(highIdx));
    p_beta_f3(s) = p;
    [h, p, ci, stats] = ttest2(avg.gamma_f3(lowIdx), avg.gamma_f3(highIdx));
    p_gamma_f3(s) = p;
    
    p_analysis_f3(s,1:5) = [p_delta_f3(s), p_theta_f3(s), p_alpha_f3(s), p_beta_f3(s), p_gamma_f3(s)];
    p_analysis_f3_min(s) = find(p_analysis_f3(s,1:5)==min(p_analysis_f3(s,1:5)));
    
    
    
    [h, p, ci, stats] = ttest2(avg.delta_c1(lowIdx), avg.delta_c1(highIdx));
    p_delta_c1(s) = p;
    [h, p, ci, stats] = ttest2(avg.theta_c1(lowIdx), avg.theta_c1(highIdx));
    p_theta_c1(s) = p;
    [h, p, ci, stats] = ttest2(avg.alpha_c1(lowIdx), avg.alpha_c1(highIdx));
    p_alpha_c1(s) = p;
    [h, p, ci, stats] = ttest2(avg.beta_c1(lowIdx), avg.beta_c1(highIdx));
    p_beta_c1(s) = p;
    [h, p, ci, stats] = ttest2(avg.gamma_c1(lowIdx), avg.gamma_c1(highIdx));
    p_gamma_c1(s) = p;
    
    p_analysis_c1(s,1:5) = [p_delta_c1(s), p_theta_c1(s), p_alpha_c1(s), p_beta_c1(s), p_gamma_c1(s)];
    p_analysis_c1_min(s) = find(p_analysis_c1(s,1:5)==min(p_analysis_c1(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.delta_c2(lowIdx), avg.delta_c2(highIdx));
    p_delta_c2(s) = p;
    [h, p, ci, stats] = ttest2(avg.theta_c2(lowIdx), avg.theta_c2(highIdx));
    p_theta_c2(s) = p;
    [h, p, ci, stats] = ttest2(avg.alpha_c2(lowIdx), avg.alpha_c2(highIdx));
    p_alpha_c2(s) = p;
    [h, p, ci, stats] = ttest2(avg.beta_c2(lowIdx), avg.beta_c2(highIdx));
    p_beta_c2(s) = p;
    [h, p, ci, stats] = ttest2(avg.gamma_c2(lowIdx), avg.gamma_c2(highIdx));
    p_gamma_c2(s) = p;
    
    p_analysis_c2(s,1:5) = [p_delta_c2(s), p_theta_c2(s), p_alpha_c2(s), p_beta_c2(s), p_gamma_c2(s)];
    p_analysis_c2_min(s) = find(p_analysis_c2(s,1:5)==min(p_analysis_c2(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.delta_c3(lowIdx), avg.delta_c3(highIdx));
    p_delta_c3(s) = p;
    [h, p, ci, stats] = ttest2(avg.theta_c3(lowIdx), avg.theta_c3(highIdx));
    p_theta_c3(s) = p;
    [h, p, ci, stats] = ttest2(avg.alpha_c3(lowIdx), avg.alpha_c3(highIdx));
    p_alpha_c3(s) = p;
    [h, p, ci, stats] = ttest2(avg.beta_c3(lowIdx), avg.beta_c3(highIdx));
    p_beta_c3(s) = p;
    [h, p, ci, stats] = ttest2(avg.gamma_c3(lowIdx), avg.gamma_c3(highIdx));
    p_gamma_c3(s) = p;
    
    p_analysis_c3(s,1:5) = [p_delta_c3(s), p_theta_c3(s), p_alpha_c3(s), p_beta_c3(s), p_gamma_c3(s)];
    p_analysis_c3_min(s) = find(p_analysis_c3(s,1:5)==min(p_analysis_c3(s,1:5)));
    
    

    
    
    [h, p, ci, stats] = ttest2(avg.delta_p1(lowIdx), avg.delta_p1(highIdx));
    p_delta_p1(s) = p;
    [h, p, ci, stats] = ttest2(avg.theta_p1(lowIdx), avg.theta_p1(highIdx));
    p_theta_p1(s) = p;
    [h, p, ci, stats] = ttest2(avg.alpha_p1(lowIdx), avg.alpha_p1(highIdx));
    p_alpha_p1(s) = p;
    [h, p, ci, stats] = ttest2(avg.beta_p1(lowIdx), avg.beta_p1(highIdx));
    p_beta_p1(s) = p;
    [h, p, ci, stats] = ttest2(avg.gamma_p1(lowIdx), avg.gamma_p1(highIdx));
    p_gamma_p1(s) = p;
    
    p_analysis_p1(s,1:5) = [p_delta_p1(s), p_theta_p1(s), p_alpha_p1(s), p_beta_p1(s), p_gamma_p1(s)];
    p_analysis_p1_min(s) = find(p_analysis_p1(s,1:5)==min(p_analysis_p1(s,1:5)));
    
    
    
    [h, p, ci, stats] = ttest2(avg.delta_p2(lowIdx), avg.delta_p2(highIdx));
    p_delta_p2(s) = p;
    [h, p, ci, stats] = ttest2(avg.theta_p2(lowIdx), avg.theta_p2(highIdx));
    p_theta_p2(s) = p;
    [h, p, ci, stats] = ttest2(avg.alpha_p2(lowIdx), avg.alpha_p2(highIdx));
    p_alpha_p2(s) = p;
    [h, p, ci, stats] = ttest2(avg.beta_p2(lowIdx), avg.beta_p2(highIdx));
    p_beta_p2(s) = p;
    [h, p, ci, stats] = ttest2(avg.gamma_p2(lowIdx), avg.gamma_p2(highIdx));
    p_gamma_p2(s) = p;
    
    p_analysis_p2(s,1:5) = [p_delta_p2(s), p_theta_p2(s), p_alpha_p2(s), p_beta_p2(s), p_gamma_p2(s)];
    p_analysis_p2_min(s) = find(p_analysis_p2(s,1:5)==min(p_analysis_p2(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.delta_p3(lowIdx), avg.delta_p3(highIdx));
    p_delta_p3(s) = p;
    [h, p, ci, stats] = ttest2(avg.theta_p3(lowIdx), avg.theta_p3(highIdx));
    p_theta_p3(s) = p;
    [h, p, ci, stats] = ttest2(avg.alpha_p3(lowIdx), avg.alpha_p3(highIdx));
    p_alpha_p3(s) = p;
    [h, p, ci, stats] = ttest2(avg.beta_p3(lowIdx), avg.beta_p3(highIdx));
    p_beta_p3(s) = p;
    [h, p, ci, stats] = ttest2(avg.gamma_p3(lowIdx), avg.gamma_p3(highIdx));
    p_gamma_p3(s) = p;
    
    p_analysis_p3(s,1:5) = [p_delta_p3(s), p_theta_p3(s), p_alpha_p3(s), p_beta_p3(s), p_gamma_p3(s)];
    p_analysis_p3_min(s) = find(p_analysis_p3(s,1:5)==min(p_analysis_p3(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.delta_t1(lowIdx), avg.delta_t1(highIdx));
    p_delta_t1(s) = p;
    [h, p, ci, stats] = ttest2(avg.theta_t1(lowIdx), avg.theta_t1(highIdx));
    p_theta_t1(s) = p;
    [h, p, ci, stats] = ttest2(avg.alpha_t1(lowIdx), avg.alpha_t1(highIdx));
    p_alpha_t1(s) = p;
    [h, p, ci, stats] = ttest2(avg.beta_t1(lowIdx), avg.beta_t1(highIdx));
    p_beta_t1(s) = p;
    [h, p, ci, stats] = ttest2(avg.gamma_t1(lowIdx), avg.gamma_t1(highIdx));
    p_gamma_t1(s) = p;
    
    p_analysis_t1(s,1:5) = [p_delta_t1(s), p_theta_t1(s), p_alpha_t1(s), p_beta_t1(s), p_gamma_t1(s)];
    p_analysis_t1_min(s) = find(p_analysis_t1(s,1:5)==min(p_analysis_t1(s,1:5)));
    
    [h, p, ci, stats] = ttest2(avg.delta_t2(lowIdx), avg.delta_t2(highIdx));
    p_delta_t2(s) = p;
    [h, p, ci, stats] = ttest2(avg.theta_t2(lowIdx), avg.theta_t2(highIdx));
    p_theta_t2(s) = p;
    [h, p, ci, stats] = ttest2(avg.alpha_t2(lowIdx), avg.alpha_t2(highIdx));
    p_alpha_t2(s) = p;
    [h, p, ci, stats] = ttest2(avg.beta_t2(lowIdx), avg.beta_t2(highIdx));
    p_beta_t2(s) = p;
    [h, p, ci, stats] = ttest2(avg.gamma_t2(lowIdx), avg.gamma_t2(highIdx));
    p_gamma_t2(s) = p;
    
    p_analysis_t2(s,1:5) = [p_delta_t2(s), p_theta_t2(s), p_alpha_t2(s), p_beta_t2(s), p_gamma_t2(s)];
    p_analysis_t2_min(s) = find(p_analysis_t2(s,1:5)==min(p_analysis_t2(s,1:5)));
    
    
    
    [h, p, ci, stats] = ttest2(avg.delta_o1(lowIdx), avg.delta_o1(highIdx));
    p_delta_o1(s) = p;
    [h, p, ci, stats] = ttest2(avg.theta_o1(lowIdx), avg.theta_o1(highIdx));
    p_theta_o1(s) = p;
    [h, p, ci, stats] = ttest2(avg.alpha_o1(lowIdx), avg.alpha_o1(highIdx));
    p_alpha_o1(s) = p;
    [h, p, ci, stats] = ttest2(avg.beta_o1(lowIdx), avg.beta_o1(highIdx));
    p_beta_o1(s) = p;
    [h, p, ci, stats] = ttest2(avg.gamma_o1(lowIdx), avg.gamma_o1(highIdx));
    p_gamma_o1(s) = p;
    
    p_analysis_o1(s,1:5) = [p_delta_o1(s), p_theta_o1(s), p_alpha_o1(s), p_beta_o1(s), p_gamma_o1(s)];
    p_analysis_o1_min(s) = find(p_analysis_o1(s,1:5)==min(p_analysis_o1(s,1:5)));
    
    [h, p, ci, stats] = ttest2(avg.delta_o2(lowIdx), avg.delta_o2(highIdx));
    p_delta_o2(s) = p;
    [h, p, ci, stats] = ttest2(avg.theta_o2(lowIdx), avg.theta_o2(highIdx));
    p_theta_o2(s) = p;
    [h, p, ci, stats] = ttest2(avg.alpha_o2(lowIdx), avg.alpha_o2(highIdx));
    p_alpha_o2(s) = p;
    [h, p, ci, stats] = ttest2(avg.beta_o2(lowIdx), avg.beta_o2(highIdx));
    p_beta_o2(s) = p;
    [h, p, ci, stats] = ttest2(avg.gamma_o2(lowIdx), avg.gamma_o2(highIdx));
    p_gamma_o2(s) = p;
    
    p_analysis_o2(s,1:5) = [p_delta_o2(s), p_theta_o2(s), p_alpha_o2(s), p_beta_o2(s), p_gamma_o2(s)];
    p_analysis_o2_min(s) = find(p_analysis_o2(s,1:5)==min(p_analysis_o2(s,1:5)));
    
    [h, p, ci, stats] = ttest2(avg.delta_o3(lowIdx), avg.delta_o3(highIdx));
    p_delta_o3(s) = p;
    [h, p, ci, stats] = ttest2(avg.theta_o3(lowIdx), avg.theta_o3(highIdx));
    p_theta_o3(s) = p;
    [h, p, ci, stats] = ttest2(avg.alpha_o3(lowIdx), avg.alpha_o3(highIdx));
    p_alpha_o3(s) = p;
    [h, p, ci, stats] = ttest2(avg.beta_o3(lowIdx), avg.beta_o3(highIdx));
    p_beta_o3(s) = p;
    [h, p, ci, stats] = ttest2(avg.gamma_o3(lowIdx), avg.gamma_o3(highIdx));
    p_gamma_o3(s) = p;
    
    p_analysis_o3(s,1:5) = [p_delta_o3(s), p_theta_o3(s), p_alpha_o3(s), p_beta_o3(s), p_gamma_o3(s)];
    p_analysis_o3_min(s) = find(p_analysis_o3(s,1:5)==min(p_analysis_o3(s,1:5)));
    
    Result.ttest.vertical(s,:) = [p_analysis_f1(s,:) p_analysis_f2(s,:) p_analysis_f3(s,:) p_analysis_c1(s,:) p_analysis_c2(s,:) p_analysis_c3(s,:) p_analysis_p1(s,:) p_analysis_p2(s,:) p_analysis_p3(s,:) p_analysis_t1(s,:) p_analysis_t2(s,:) p_analysis_o1(s,:) p_analysis_o2(s,:) p_analysis_o3(s,:)];
    
    
    
        % Regional Vertical T-TEST 분석
    [h, p, ci, stats] = ttest2(avg.RPL.delta_f1(lowIdx), avg.RPL.delta_f1(highIdx));
    p_delta_f1(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.theta_f1(lowIdx), avg.RPL.theta_f1(highIdx));
    p_theta_f1(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.alpha_f1(lowIdx), avg.RPL.alpha_f1(highIdx));
    p_alpha_f1(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.beta_f1(lowIdx), avg.RPL.beta_f1(highIdx));
    p_beta_f1(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.gamma_f1(lowIdx), avg.RPL.gamma_f1(highIdx));
    p_gamma_f1(s) = p;
    
    p_analysis_f1_rpl(s,1:5) = [p_delta_f1(s), p_theta_f1(s), p_alpha_f1(s), p_beta_f1(s), p_gamma_f1(s)];
    p_analysis_f1_rpl_min(s) = find(p_analysis_f1_rpl(s,1:5)==min(p_analysis_f1_rpl(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.RPL.delta_f2(lowIdx), avg.RPL.delta_f2(highIdx));
    p_delta_f2(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.theta_f2(lowIdx), avg.RPL.theta_f2(highIdx));
    p_theta_f2(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.alpha_f2(lowIdx), avg.RPL.alpha_f2(highIdx));
    p_alpha_f2(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.beta_f2(lowIdx), avg.RPL.beta_f2(highIdx));
    p_beta_f2(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.gamma_f2(lowIdx), avg.RPL.gamma_f2(highIdx));
    p_gamma_f2(s) = p;
    
    p_analysis_f2_rpl(s,1:5) = [p_delta_f2(s), p_theta_f2(s), p_alpha_f2(s), p_beta_f2(s), p_gamma_f2(s)];
    p_analysis_f2_rpl_min(s) = find(p_analysis_f2_rpl(s,1:5)==min(p_analysis_f2_rpl(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.RPL.delta_f3(lowIdx), avg.RPL.delta_f3(highIdx));
    p_delta_f3(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.theta_f3(lowIdx), avg.RPL.theta_f3(highIdx));
    p_theta_f3(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.alpha_f3(lowIdx), avg.RPL.alpha_f3(highIdx));
    p_alpha_f3(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.beta_f3(lowIdx), avg.RPL.beta_f3(highIdx));
    p_beta_f3(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.gamma_f3(lowIdx), avg.RPL.gamma_f3(highIdx));
    p_gamma_f3(s) = p;
    
    p_analysis_f3_rpl(s,1:5) = [p_delta_f3(s), p_theta_f3(s), p_alpha_f3(s), p_beta_f3(s), p_gamma_f3(s)];
    p_analysis_f3_rpl_min(s) = find(p_analysis_f3_rpl(s,1:5)==min(p_analysis_f3_rpl(s,1:5)));
    
    
    
    [h, p, ci, stats] = ttest2(avg.RPL.delta_c1(lowIdx), avg.RPL.delta_c1(highIdx));
    p_delta_c1(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.theta_c1(lowIdx), avg.RPL.theta_c1(highIdx));
    p_theta_c1(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.alpha_c1(lowIdx), avg.RPL.alpha_c1(highIdx));
    p_alpha_c1(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.beta_c1(lowIdx), avg.RPL.beta_c1(highIdx));
    p_beta_c1(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.gamma_c1(lowIdx), avg.RPL.gamma_c1(highIdx));
    p_gamma_c1(s) = p;
    
    p_analysis_c1_rpl(s,1:5) = [p_delta_c1(s), p_theta_c1(s), p_alpha_c1(s), p_beta_c1(s), p_gamma_c1(s)];
    p_analysis_c1_rpl_min(s) = find(p_analysis_c1_rpl(s,1:5)==min(p_analysis_c1_rpl(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.RPL.delta_c2(lowIdx), avg.RPL.delta_c2(highIdx));
    p_delta_c2(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.theta_c2(lowIdx), avg.RPL.theta_c2(highIdx));
    p_theta_c2(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.alpha_c2(lowIdx), avg.RPL.alpha_c2(highIdx));
    p_alpha_c2(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.beta_c2(lowIdx), avg.RPL.beta_c2(highIdx));
    p_beta_c2(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.gamma_c2(lowIdx), avg.RPL.gamma_c2(highIdx));
    p_gamma_c2(s) = p;
    
    p_analysis_c2_rpl(s,1:5) = [p_delta_c2(s), p_theta_c2(s), p_alpha_c2(s), p_beta_c2(s), p_gamma_c2(s)];
    p_analysis_c2_rpl_min(s) = find(p_analysis_c2_rpl(s,1:5)==min(p_analysis_c2_rpl(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.RPL.delta_c3(lowIdx), avg.RPL.delta_c3(highIdx));
    p_delta_c3(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.theta_c3(lowIdx), avg.RPL.theta_c3(highIdx));
    p_theta_c3(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.alpha_c3(lowIdx), avg.RPL.alpha_c3(highIdx));
    p_alpha_c3(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.beta_c3(lowIdx), avg.RPL.beta_c3(highIdx));
    p_beta_c3(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.gamma_c3(lowIdx), avg.RPL.gamma_c3(highIdx));
    p_gamma_c3(s) = p;
    
    p_analysis_c3_rpl(s,1:5) = [p_delta_c3(s), p_theta_c3(s), p_alpha_c3(s), p_beta_c3(s), p_gamma_c3(s)];
    p_analysis_c3_rpl_min(s) = find(p_analysis_c3_rpl(s,1:5)==min(p_analysis_c3_rpl(s,1:5)));
    
    

    
    
    [h, p, ci, stats] = ttest2(avg.RPL.delta_p1(lowIdx), avg.RPL.delta_p1(highIdx));
    p_delta_p1(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.theta_p1(lowIdx), avg.RPL.theta_p1(highIdx));
    p_theta_p1(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.alpha_p1(lowIdx), avg.RPL.alpha_p1(highIdx));
    p_alpha_p1(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.beta_p1(lowIdx), avg.RPL.beta_p1(highIdx));
    p_beta_p1(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.gamma_p1(lowIdx), avg.RPL.gamma_p1(highIdx));
    p_gamma_p1(s) = p;
    
    p_analysis_p1_rpl(s,1:5) = [p_delta_p1(s), p_theta_p1(s), p_alpha_p1(s), p_beta_p1(s), p_gamma_p1(s)];
    p_analysis_p1_rpl_min(s) = find(p_analysis_p1_rpl(s,1:5)==min(p_analysis_p1_rpl(s,1:5)));
    
    
    
    [h, p, ci, stats] = ttest2(avg.RPL.delta_p2(lowIdx), avg.RPL.delta_p2(highIdx));
    p_delta_p2(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.theta_p2(lowIdx), avg.RPL.theta_p2(highIdx));
    p_theta_p2(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.alpha_p2(lowIdx), avg.RPL.alpha_p2(highIdx));
    p_alpha_p2(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.beta_p2(lowIdx), avg.RPL.beta_p2(highIdx));
    p_beta_p2(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.gamma_p2(lowIdx), avg.RPL.gamma_p2(highIdx));
    p_gamma_p2(s) = p;
    
    p_analysis_p2_rpl(s,1:5) = [p_delta_p2(s), p_theta_p2(s), p_alpha_p2(s), p_beta_p2(s), p_gamma_p2(s)];
    p_analysis_p2_rpl_min(s) = find(p_analysis_p2_rpl(s,1:5)==min(p_analysis_p2_rpl(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.RPL.delta_p3(lowIdx), avg.RPL.delta_p3(highIdx));
    p_delta_p3(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.theta_p3(lowIdx), avg.RPL.theta_p3(highIdx));
    p_theta_p3(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.alpha_p3(lowIdx), avg.RPL.alpha_p3(highIdx));
    p_alpha_p3(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.beta_p3(lowIdx), avg.RPL.beta_p3(highIdx));
    p_beta_p3(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.gamma_p3(lowIdx), avg.RPL.gamma_p3(highIdx));
    p_gamma_p3(s) = p;
    
    p_analysis_p3_rpl(s,1:5) = [p_delta_p3(s), p_theta_p3(s), p_alpha_p3(s), p_beta_p3(s), p_gamma_p3(s)];
    p_analysis_p3_rpl_min(s) = find(p_analysis_p3_rpl(s,1:5)==min(p_analysis_p3_rpl(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.RPL.delta_t1(lowIdx), avg.RPL.delta_t1(highIdx));
    p_delta_t1(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.theta_t1(lowIdx), avg.RPL.theta_t1(highIdx));
    p_theta_t1(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.alpha_t1(lowIdx), avg.RPL.alpha_t1(highIdx));
    p_alpha_t1(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.beta_t1(lowIdx), avg.RPL.beta_t1(highIdx));
    p_beta_t1(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.gamma_t1(lowIdx), avg.RPL.gamma_t1(highIdx));
    p_gamma_t1(s) = p;
    
    p_analysis_t1_rpl(s,1:5) = [p_delta_t1(s), p_theta_t1(s), p_alpha_t1(s), p_beta_t1(s), p_gamma_t1(s)];
    p_analysis_t1_rpl_min(s) = find(p_analysis_t1_rpl(s,1:5)==min(p_analysis_t1_rpl(s,1:5)));
    
    [h, p, ci, stats] = ttest2(avg.RPL.delta_t2(lowIdx), avg.RPL.delta_t2(highIdx));
    p_delta_t2(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.theta_t2(lowIdx), avg.RPL.theta_t2(highIdx));
    p_theta_t2(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.alpha_t2(lowIdx), avg.RPL.alpha_t2(highIdx));
    p_alpha_t2(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.beta_t2(lowIdx), avg.RPL.beta_t2(highIdx));
    p_beta_t2(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.gamma_t2(lowIdx), avg.RPL.gamma_t2(highIdx));
    p_gamma_t2(s) = p;
    
    p_analysis_t2_rpl(s,1:5) = [p_delta_t2(s), p_theta_t2(s), p_alpha_t2(s), p_beta_t2(s), p_gamma_t2(s)];
    p_analysis_t2_rpl_min(s) = find(p_analysis_t2_rpl(s,1:5)==min(p_analysis_t2_rpl(s,1:5)));
    
    
    
    [h, p, ci, stats] = ttest2(avg.RPL.delta_o1(lowIdx), avg.RPL.delta_o1(highIdx));
    p_delta_o1(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.theta_o1(lowIdx), avg.RPL.theta_o1(highIdx));
    p_theta_o1(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.alpha_o1(lowIdx), avg.RPL.alpha_o1(highIdx));
    p_alpha_o1(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.beta_o1(lowIdx), avg.RPL.beta_o1(highIdx));
    p_beta_o1(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.gamma_o1(lowIdx), avg.RPL.gamma_o1(highIdx));
    p_gamma_o1(s) = p;
    
    p_analysis_o1_rpl(s,1:5) = [p_delta_o1(s), p_theta_o1(s), p_alpha_o1(s), p_beta_o1(s), p_gamma_o1(s)];
    p_analysis_o1_rpl_min(s) = find(p_analysis_o1_rpl(s,1:5)==min(p_analysis_o1_rpl(s,1:5)));
    
    [h, p, ci, stats] = ttest2(avg.RPL.delta_o2(lowIdx), avg.RPL.delta_o2(highIdx));
    p_delta_o2(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.theta_o2(lowIdx), avg.RPL.theta_o2(highIdx));
    p_theta_o2(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.alpha_o2(lowIdx), avg.RPL.alpha_o2(highIdx));
    p_alpha_o2(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.beta_o2(lowIdx), avg.RPL.beta_o2(highIdx));
    p_beta_o2(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.gamma_o2(lowIdx), avg.RPL.gamma_o2(highIdx));
    p_gamma_o2(s) = p;
    
    p_analysis_o2_rpl(s,1:5) = [p_delta_o2(s), p_theta_o2(s), p_alpha_o2(s), p_beta_o2(s), p_gamma_o2(s)];
    p_analysis_o2_rpl_min(s) = find(p_analysis_o2_rpl(s,1:5)==min(p_analysis_o2_rpl(s,1:5)));
    
    [h, p, ci, stats] = ttest2(avg.RPL.delta_o3(lowIdx), avg.RPL.delta_o3(highIdx));
    p_delta_o3(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.theta_o3(lowIdx), avg.RPL.theta_o3(highIdx));
    p_theta_o3(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.alpha_o3(lowIdx), avg.RPL.alpha_o3(highIdx));
    p_alpha_o3(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.beta_o3(lowIdx), avg.RPL.beta_o3(highIdx));
    p_beta_o3(s) = p;
    [h, p, ci, stats] = ttest2(avg.RPL.gamma_o3(lowIdx), avg.RPL.gamma_o3(highIdx));
    p_gamma_o3(s) = p;
    
    p_analysis_o3_rpl(s,1:5) = [p_delta_o3(s), p_theta_o3(s), p_alpha_o3(s), p_beta_o3(s), p_gamma_o3(s)];
    p_analysis_o3_rpl_min(s) = find(p_analysis_o3_rpl(s,1:5)==min(p_analysis_o3_rpl(s,1:5)));
    
    Result.ttest.vertical_rpl(s,:) = [p_analysis_f1(s,:) p_analysis_f2(s,:) p_analysis_f3(s,:) p_analysis_c1(s,:) p_analysis_c2(s,:) p_analysis_c3(s,:) p_analysis_p1(s,:) p_analysis_p2(s,:) p_analysis_p3(s,:) p_analysis_t1(s,:) p_analysis_t2(s,:) p_analysis_o1(s,:) p_analysis_o2(s,:) p_analysis_o3(s,:)];
    
    
    
    
    
    
    
            % Regional Vertical T-TEST 분석
    [h, p, ci, stats] = ttest2(avg.z.Regional_f1(lowIdx,1), avg.z.Regional_f1(highIdx,1));
    p_delta_f1(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_f1(lowIdx,2), avg.z.Regional_f1(highIdx,2));
    p_theta_f1(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_f1(lowIdx,3), avg.z.Regional_f1(highIdx,3));
    p_alpha_f1(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_f1(lowIdx,4), avg.z.Regional_f1(highIdx,4));
    p_beta_f1(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_f1(lowIdx,5), avg.z.Regional_f1(highIdx,5));
    p_gamma_f1(s) = p;
    
    p_analysis_f1_z(s,1:5) = [p_delta_f1(s), p_theta_f1(s), p_alpha_f1(s), p_beta_f1(s), p_gamma_f1(s)];
    p_analysis_f1_z_min(s) = find(p_analysis_f1_z(s,1:5)==min(p_analysis_f1_z(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.z.Regional_f2(lowIdx,1), avg.z.Regional_f2(highIdx,1));
    p_delta_f2(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_f2(lowIdx,2), avg.z.Regional_f2(highIdx,2));
    p_theta_f2(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_f2(lowIdx,3), avg.z.Regional_f2(highIdx,3));
    p_alpha_f2(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_f2(lowIdx,4), avg.z.Regional_f2(highIdx,4));
    p_beta_f2(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_f2(lowIdx,5), avg.z.Regional_f2(highIdx,5));
    p_gamma_f2(s) = p;
    
    p_analysis_f2_z(s,1:5) = [p_delta_f2(s), p_theta_f2(s), p_alpha_f2(s), p_beta_f2(s), p_gamma_f2(s)];
    p_analysis_f2_z_min(s) = find(p_analysis_f2_z(s,1:5)==min(p_analysis_f2_z(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.z.Regional_f3(lowIdx,1), avg.z.Regional_f3(highIdx,1));
    p_delta_f3(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_f3(lowIdx,2), avg.z.Regional_f3(highIdx,2));
    p_theta_f3(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_f3(lowIdx,3), avg.z.Regional_f3(highIdx,3));
    p_alpha_f3(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_f3(lowIdx,4), avg.z.Regional_f3(highIdx,4));
    p_beta_f3(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_f3(lowIdx,5), avg.z.Regional_f3(highIdx,5));
    p_gamma_f3(s) = p;
    
    p_analysis_f3_z(s,1:5) = [p_delta_f3(s), p_theta_f3(s), p_alpha_f3(s), p_beta_f3(s), p_gamma_f3(s)];
    p_analysis_f3_z_min(s) = find(p_analysis_f3_z(s,1:5)==min(p_analysis_f3_z(s,1:5)));
    
    
    
    [h, p, ci, stats] = ttest2(avg.z.Regional_c1(lowIdx,1), avg.z.Regional_c1(highIdx,1));
    p_delta_c1(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_c1(lowIdx,2), avg.z.Regional_c1(highIdx,2));
    p_theta_c1(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_c1(lowIdx,3), avg.z.Regional_c1(highIdx,3));
    p_alpha_c1(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_c1(lowIdx,4), avg.z.Regional_c1(highIdx,4));
    p_beta_c1(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_c1(lowIdx,5), avg.z.Regional_c1(highIdx,5));
    p_gamma_c1(s) = p;
    
    p_analysis_c1_z(s,1:5) = [p_delta_c1(s), p_theta_c1(s), p_alpha_c1(s), p_beta_c1(s), p_gamma_c1(s)];
    p_analysis_c1_z_min(s) = find(p_analysis_c1_z(s,1:5)==min(p_analysis_c1_z(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.z.Regional_c2(lowIdx,1), avg.z.Regional_c2(highIdx,1));
    p_delta_c2(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_c2(lowIdx,2), avg.z.Regional_c2(highIdx,2));
    p_theta_c2(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_c2(lowIdx,3), avg.z.Regional_c2(highIdx,3));
    p_alpha_c2(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_c2(lowIdx,4), avg.z.Regional_c2(highIdx,4));
    p_beta_c2(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_c2(lowIdx,5), avg.z.Regional_c2(highIdx,5));
    p_gamma_c2(s) = p;
    
    p_analysis_c2_z(s,1:5) = [p_delta_c2(s), p_theta_c2(s), p_alpha_c2(s), p_beta_c2(s), p_gamma_c2(s)];
    p_analysis_c2_z_min(s) = find(p_analysis_c2_z(s,1:5)==min(p_analysis_c2_z(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.z.Regional_c3(lowIdx,1), avg.z.Regional_c3(highIdx,1));
    p_delta_c3(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_c3(lowIdx,2), avg.z.Regional_c3(highIdx,2));
    p_theta_c3(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_c3(lowIdx,3), avg.z.Regional_c3(highIdx,3));
    p_alpha_c3(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_c3(lowIdx,4), avg.z.Regional_c3(highIdx,4));
    p_beta_c3(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_c3(lowIdx,5), avg.z.Regional_c3(highIdx,5));
    p_gamma_c3(s) = p;
    
    p_analysis_c3_z(s,1:5) = [p_delta_c3(s), p_theta_c3(s), p_alpha_c3(s), p_beta_c3(s), p_gamma_c3(s)];
    p_analysis_c3_z_min(s) = find(p_analysis_c3_z(s,1:5)==min(p_analysis_c3_z(s,1:5)));
    
    

    
    
    [h, p, ci, stats] = ttest2(avg.z.Regional_p1(lowIdx,1), avg.z.Regional_p1(highIdx,1));
    p_delta_p1(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_p1(lowIdx,2), avg.z.Regional_p1(highIdx,2));
    p_theta_p1(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_p1(lowIdx,3), avg.z.Regional_p1(highIdx,3));
    p_alpha_p1(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_p1(lowIdx,4), avg.z.Regional_p1(highIdx,4));
    p_beta_p1(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_p1(lowIdx,5), avg.z.Regional_p1(highIdx,5));
    p_gamma_p1(s) = p;
    
    p_analysis_p1_z(s,1:5) = [p_delta_p1(s), p_theta_p1(s), p_alpha_p1(s), p_beta_p1(s), p_gamma_p1(s)];
    p_analysis_p1_z_min(s) = find(p_analysis_p1_z(s,1:5)==min(p_analysis_p1_z(s,1:5)));
    
    
    
    [h, p, ci, stats] = ttest2(avg.z.Regional_p2(lowIdx,1), avg.z.Regional_p2(highIdx,1));
    p_delta_p2(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_p2(lowIdx,2), avg.z.Regional_p2(highIdx,2));
    p_theta_p2(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_p2(lowIdx,3), avg.z.Regional_p2(highIdx,3));
    p_alpha_p2(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_p2(lowIdx,4), avg.z.Regional_p2(highIdx,4));
    p_beta_p2(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_p2(lowIdx,5), avg.z.Regional_p2(highIdx,5));
    p_gamma_p2(s) = p;
    
    p_analysis_p2_z(s,1:5) = [p_delta_p2(s), p_theta_p2(s), p_alpha_p2(s), p_beta_p2(s), p_gamma_p2(s)];
    p_analysis_p2_z_min(s) = find(p_analysis_p2_z(s,1:5)==min(p_analysis_p2_z(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.z.Regional_p3(lowIdx,1), avg.z.Regional_p3(highIdx,1));
    p_delta_p3(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_p3(lowIdx,2), avg.z.Regional_p3(highIdx,2));
    p_theta_p3(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_p3(lowIdx,3), avg.z.Regional_p3(highIdx,3));
    p_alpha_p3(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_p3(lowIdx,4), avg.z.Regional_p3(highIdx,4));
    p_beta_p3(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_p3(lowIdx,5), avg.z.Regional_p3(highIdx,5));
    p_gamma_p3(s) = p;
    
    p_analysis_p3_z(s,1:5) = [p_delta_p3(s), p_theta_p3(s), p_alpha_p3(s), p_beta_p3(s), p_gamma_p3(s)];
    p_analysis_p3_z_min(s) = find(p_analysis_p3_z(s,1:5)==min(p_analysis_p3_z(s,1:5)));
    
    
    [h, p, ci, stats] = ttest2(avg.z.Regional_t1(lowIdx,1), avg.z.Regional_t1(highIdx,1));
    p_delta_t1(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_t1(lowIdx,2), avg.z.Regional_t1(highIdx,2));
    p_theta_t1(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_t1(lowIdx,3), avg.z.Regional_t1(highIdx,3));
    p_alpha_t1(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_t1(lowIdx,4), avg.z.Regional_t1(highIdx,4));
    p_beta_t1(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_t1(lowIdx,5), avg.z.Regional_t1(highIdx,5));
    p_gamma_t1(s) = p;
    
    p_analysis_t1_z(s,1:5) = [p_delta_t1(s), p_theta_t1(s), p_alpha_t1(s), p_beta_t1(s), p_gamma_t1(s)];
    p_analysis_t1_z_min(s) = find(p_analysis_t1_z(s,1:5)==min(p_analysis_t1_z(s,1:5)));
    
    [h, p, ci, stats] = ttest2(avg.z.Regional_t2(lowIdx,1), avg.z.Regional_t2(highIdx,1));
    p_delta_t2(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_t2(lowIdx,2), avg.z.Regional_t2(highIdx,2));
    p_theta_t2(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_t2(lowIdx,3), avg.z.Regional_t2(highIdx,3));
    p_alpha_t2(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_t2(lowIdx,4), avg.z.Regional_t2(highIdx,4));
    p_beta_t2(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_t2(lowIdx,5), avg.z.Regional_t2(highIdx,5));
    p_gamma_t2(s) = p;
    
    p_analysis_t2_z(s,1:5) = [p_delta_t2(s), p_theta_t2(s), p_alpha_t2(s), p_beta_t2(s), p_gamma_t2(s)];
    p_analysis_t2_z_min(s) = find(p_analysis_t2_z(s,1:5)==min(p_analysis_t2_z(s,1:5)));
    
    
    
    [h, p, ci, stats] = ttest2(avg.z.Regional_o1(lowIdx,1), avg.z.Regional_o1(highIdx,1));
    p_delta_o1(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_o1(lowIdx,2), avg.z.Regional_o1(highIdx,2));
    p_theta_o1(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_o1(lowIdx,3), avg.z.Regional_o1(highIdx,3));
    p_alpha_o1(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_o1(lowIdx,4), avg.z.Regional_o1(highIdx,4));
    p_beta_o1(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_o1(lowIdx,5), avg.z.Regional_o1(highIdx,5));
    p_gamma_o1(s) = p;
    
    p_analysis_o1_z(s,1:5) = [p_delta_o1(s), p_theta_o1(s), p_alpha_o1(s), p_beta_o1(s), p_gamma_o1(s)];
    p_analysis_o1_z_min(s) = find(p_analysis_o1_z(s,1:5)==min(p_analysis_o1_z(s,1:5)));
    
    [h, p, ci, stats] = ttest2(avg.z.Regional_o2(lowIdx,1), avg.z.Regional_o2(highIdx,1));
    p_delta_o2(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_o2(lowIdx,2), avg.z.Regional_o2(highIdx,2));
    p_theta_o2(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_o2(lowIdx,3), avg.z.Regional_o2(highIdx,3));
    p_alpha_o2(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_o2(lowIdx,4), avg.z.Regional_o2(highIdx,4));
    p_beta_o2(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_o2(lowIdx,5), avg.z.Regional_o2(highIdx,5));
    p_gamma_o2(s) = p;
    
    p_analysis_o2_z(s,1:5) = [p_delta_o2(s), p_theta_o2(s), p_alpha_o2(s), p_beta_o2(s), p_gamma_o2(s)];
    p_analysis_o2_z_min(s) = find(p_analysis_o2_z(s,1:5)==min(p_analysis_o2_z(s,1:5)));
    
    [h, p, ci, stats] = ttest2(avg.z.Regional_o3(lowIdx,1), avg.z.Regional_o3(highIdx,1));
    p_delta_o3(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_o3(lowIdx,2), avg.z.Regional_o3(highIdx,2));
    p_theta_o3(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_o3(lowIdx,3), avg.z.Regional_o3(highIdx,3));
    p_alpha_o3(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_o3(lowIdx,4), avg.z.Regional_o3(highIdx,4));
    p_beta_o3(s) = p;
    [h, p, ci, stats] = ttest2(avg.z.Regional_o3(lowIdx,5), avg.z.Regional_o3(highIdx,5));
    p_gamma_o3(s) = p;
    
    p_analysis_o3_z(s,1:5) = [p_delta_o3(s), p_theta_o3(s), p_alpha_o3(s), p_beta_o3(s), p_gamma_o3(s)];
    p_analysis_o3_z_min(s) = find(p_analysis_o3_z(s,1:5)==min(p_analysis_o3_z(s,1:5)));
    
    Result.ttest.vertical_z(s,:) = [p_analysis_f1(s,:) p_analysis_f2(s,:) p_analysis_f3(s,:) p_analysis_c1(s,:) p_analysis_c2(s,:) p_analysis_c3(s,:) p_analysis_p1(s,:) p_analysis_p2(s,:) p_analysis_p3(s,:) p_analysis_t1(s,:) p_analysis_t2(s,:) p_analysis_o1(s,:) p_analysis_o2(s,:) p_analysis_o3(s,:)];
    
    
    
    
    
    
    
    
    
    
    %% Feature Selection
    feature_ALL = avg.PSD;
    feature_PSD = avg.PSD(:,find(p_analysis(s,:)<0.05));
    feature_Regional = [avg.Regional_f(:,find(p_analysis_f(s,:)<0.05)),avg.Regional_c(:,find(p_analysis_c(s,:)<0.05)),avg.Regional_p(:,find(p_analysis_p(s,:)<0.05)),avg.Regional_t(:,find(p_analysis_t(s,:)<0.05)),avg.Regional_o(:,find(p_analysis_o(s,:)<0.05))];
    feature_Vertical = [avg.Regional_f1(:,find(p_analysis_f1(s,:)<0.05)),avg.Regional_f2(:,find(p_analysis_f2(s,:)<0.05)),avg.Regional_f3(:,find(p_analysis_f3(s,:)<0.05)),avg.Regional_c1(:,find(p_analysis_c1(s,:)<0.05)),avg.Regional_c2(:,find(p_analysis_c2(s,:)<0.05)),avg.Regional_c3(:,find(p_analysis_c3(s,:)<0.05)),avg.Regional_p1(:,find(p_analysis_p1(s,:)<0.05)),avg.Regional_p2(:,find(p_analysis_p2(s,:)<0.05)),avg.Regional_p3(:,find(p_analysis_p3(s,:)<0.05)),avg.Regional_t1(:,find(p_analysis_t1(s,:)<0.05)),avg.Regional_t2(:,find(p_analysis_t2(s,:)<0.05)),avg.Regional_o1(:,find(p_analysis_o1(s,:)<0.05)),avg.Regional_o2(:,find(p_analysis_o2(s,:)<0.05)),avg.Regional_o3(:,find(p_analysis_o3(s,:)<0.05))];
    
    feature_ALL_RPL = avg.RPL.PSD;
    feature_PSD_RPL = avg.RPL.PSD(:,find(p_analysis_RPL(s,:)<0.05));
    feature_Regional_RPL = [avg.RPL.Regional_f(:,find(p_analysis_RPL_f(s,:)<0.05)),avg.RPL.Regional_c(:,find(p_analysis_RPL_c(s,:)<0.05)),avg.RPL.Regional_p(:,find(p_analysis_RPL_p(s,:)<0.05)),avg.RPL.Regional_t(:,find(p_analysis_RPL_t(s,:)<0.05)),avg.RPL.Regional_o(:,find(p_analysis_RPL_o(s,:)<0.05))];
    feature_Vertical_RPL = [avg.RPL.Regional_f1(:,find(p_analysis_f1_rpl(s,:)<0.05)),avg.RPL.Regional_f2(:,find(p_analysis_f2_rpl(s,:)<0.05)),avg.RPL.Regional_f3(:,find(p_analysis_f3_rpl(s,:)<0.05)),avg.RPL.Regional_c1(:,find(p_analysis_c1_rpl(s,:)<0.05)),avg.RPL.Regional_c2(:,find(p_analysis_c2_rpl(s,:)<0.05)),avg.RPL.Regional_c3(:,find(p_analysis_c3_rpl(s,:)<0.05)),avg.RPL.Regional_p1(:,find(p_analysis_p1_rpl(s,:)<0.05)),avg.RPL.Regional_p2(:,find(p_analysis_p2_rpl(s,:)<0.05)),avg.RPL.Regional_p3(:,find(p_analysis_p3_rpl(s,:)<0.05)),avg.RPL.Regional_t1(:,find(p_analysis_t1_rpl(s,:)<0.05)),avg.RPL.Regional_t2(:,find(p_analysis_t2_rpl(s,:)<0.05)),avg.RPL.Regional_o1(:,find(p_analysis_o1_rpl(s,:)<0.05)),avg.RPL.Regional_o2(:,find(p_analysis_o2_rpl(s,:)<0.05)),avg.RPL.Regional_o3(:,find(p_analysis_o3_rpl(s,:)<0.05))];

    feature_ALL_z = avg.z.PSD;
    feature_PSD_z = avg.z.PSD(:,find(p_analysis_z(s,:)<0.05));
    feature_Regional_z = [avg.z.Regional_f(:,find(p_analysis_z_f(s,:)<0.05)),avg.z.Regional_c(:,find(p_analysis_z_c(s,:)<0.05)),avg.z.Regional_p(:,find(p_analysis_z_p(s,:)<0.05)),avg.z.Regional_t(:,find(p_analysis_z_t(s,:)<0.05)),avg.z.Regional_o(:,find(p_analysis_z_o(s,:)<0.05))];
    feature_Vertical_z = [avg.z.Regional_f1(:,find(p_analysis_f1_z(s,:)<0.05)),avg.z.Regional_f2(:,find(p_analysis_f2_z(s,:)<0.05)),avg.z.Regional_f3(:,find(p_analysis_f3_z(s,:)<0.05)),avg.z.Regional_c1(:,find(p_analysis_c1_z(s,:)<0.05)),avg.z.Regional_c2(:,find(p_analysis_c2_z(s,:)<0.05)),avg.z.Regional_c3(:,find(p_analysis_c3_z(s,:)<0.05)),avg.z.Regional_p1(:,find(p_analysis_p1_z(s,:)<0.05)),avg.z.Regional_p2(:,find(p_analysis_p2_z(s,:)<0.05)),avg.z.Regional_p3(:,find(p_analysis_p3_z(s,:)<0.05)),avg.z.Regional_t1(:,find(p_analysis_t1_z(s,:)<0.05)),avg.z.Regional_t2(:,find(p_analysis_t2_z(s,:)<0.05)),avg.z.Regional_o1(:,find(p_analysis_o1_z(s,:)<0.05)),avg.z.Regional_o2(:,find(p_analysis_o2_z(s,:)<0.05)),avg.z.Regional_o3(:,find(p_analysis_o3_z(s,:)<0.05))];
    
    %% Classification
    [Accuracy_ALL AUC_ALL] = LDA_4CV(feature_ALL, highIdx, lowIdx, s);
    [Accuracy_PSD AUC_PSD] = LDA_4CV(feature_PSD, highIdx, lowIdx, s);
    [Accuracy_Regional AUC_Regional] = LDA_4CV(feature_Regional, highIdx, lowIdx, s);
    [Accuracy_Vertical AUC_Vertical] = LDA_4CV(feature_Vertical, highIdx, lowIdx, s);
    
    [Accuracy_ALL_RPL AUC_ALL_RPL] = LDA_4CV(feature_ALL_RPL, highIdx, lowIdx, s);
    [Accuracy_PSD_RPL AUC_PSD_RPL] = LDA_4CV(feature_PSD_RPL, highIdx, lowIdx, s);
    [Accuracy_Regional_RPL AUC_Regional_RPL] = LDA_4CV(feature_Regional_RPL, highIdx, lowIdx, s);
    [Accuracy_Vertical_RPL AUC_Vertical_RPL] = LDA_4CV(feature_Vertical_RPL, highIdx, lowIdx, s);
    
    [Accuracy_ALL_z AUC_ALL_z] = LDA_4CV(feature_ALL_z, highIdx, lowIdx, s);
    [Accuracy_PSD_z AUC_PSD_z] = LDA_4CV(feature_PSD_z, highIdx, lowIdx, s);
    [Accuracy_Regional_z AUC_Regional_z] = LDA_4CV(feature_Regional_z, highIdx, lowIdx, s);
    [Accuracy_Vertical_z AUC_Vertical_z] = LDA_4CV(feature_Vertical_z, highIdx, lowIdx, s);
    
    Result.Fatigue_Accuracy(s,:) = [Accuracy_ALL Accuracy_PSD Accuracy_Regional Accuracy_Vertical];
    Result.Fatigue_AUC(s,:) = [AUC_ALL AUC_PSD AUC_Regional AUC_Vertical];
    
    Result.Fatigue_Accuracy_RPL(s,:) = [Accuracy_ALL_RPL Accuracy_PSD_RPL Accuracy_Regional_RPL Accuracy_Vertical_RPL];
    Result.Fatigue_AUC_RPL(s,:) = [AUC_ALL_RPL AUC_PSD_RPL AUC_Regional_RPL AUC_Vertical_RPL];
    
    Result.Fatigue_Accuracy_z(s,:) = [Accuracy_ALL_z Accuracy_PSD_z Accuracy_Regional_z Accuracy_Vertical_z];
    Result.Fatigue_AUC_z(s,:) = [AUC_ALL_z AUC_PSD_z AUC_Regional_z AUC_Vertical_z];
    
%     %% Regression
%     [Prediction_ALL, RMSE_ALL, Error_Rate_ALL] = MLR_4CV(feature_ALL, kss, s);
%     [Prediction_PSD, RMSE_PSD, Error_Rate_PSD] = MLR_4CV(feature_PSD, kss, s);
%     [Prediction_Regional, RMSE_Regional, Error_Rate_Regional] = MLR_4CV(feature_Regional, kss, s);
%     [Prediction_Vertical, RMSE_Vertical, Error_Rate_Vertical] = MLR_4CV(feature_Vertical, kss, s);
%     
%     [Prediction_ALL_RPL, RMSE_ALL_RPL, Error_Rate_ALL_RPL] = MLR_4CV(feature_ALL_RPL, kss, s);
%     [Prediction_PSD_RPL, RMSE_PSD_RPL, Error_Rate_PSD_RPL] = MLR_4CV(feature_PSD_RPL, kss, s);
%     [Prediction_Regional_RPL, RMSE_Regional_RPL, Error_Rate_Regional_RPL] = MLR_4CV(feature_Regional_RPL, kss, s);
%     [Prediction_Vertical_RPL, RMSE_Vertical_RPL, Error_Rate_Vertical_RPL] = MLR_4CV(feature_Vertical_RPL, kss, s);
%     
%     [Prediction_ALL_z, RMSE_ALL_z, Error_Rate_ALL_z] = MLR_4CV(feature_ALL_z, kss, s);
%     [Prediction_PSD_z, RMSE_PSD_z, Error_Rate_PSD_z] = MLR_4CV(feature_PSD_z, kss, s);
%     [Prediction_Regional_z, RMSE_Regional_z, Error_Rate_Regional_z] = MLR_4CV(feature_Regional_z, kss, s);
%     [Prediction_Vertical_z, RMSE_Vertical_z, Error_Rate_Vertical_z] = MLR_4CV(feature_Vertical_z, kss, s);
%  
%     Result.Fatigue_Prediction(s,:) = [Prediction_ALL Prediction_PSD Prediction_Regional Prediction_Vertical];
%     Result.Fatigue_RMSE(s,:) = [RMSE_ALL RMSE_PSD RMSE_Regional RMSE_Vertical];   
%     Result.Fatigue_Error(s,:) = [Error_Rate_ALL Error_Rate_PSD Error_Rate_Regional Error_Rate_Vertical];   
%     
%     Result.Fatigue_Prediction_RPL(s,:) = [Prediction_ALL_RPL Prediction_PSD_RPL Prediction_Regional_RPL Prediction_Vertical_RPL];
%     Result.Fatigue_RMSE_RPL(s,:) = [RMSE_ALL_RPL RMSE_PSD_RPL RMSE_Regional_RPL RMSE_Vertical_RPL];   
%     Result.Fatigue_Error_RPL(s,:) = [Error_Rate_ALL_RPL Error_Rate_PSD_RPL Error_Rate_Regional_RPL Error_Rate_Vertical_RPL];   
%     
%     Result.Fatigue_Prediction_z(s,:) = [Prediction_ALL_z Prediction_PSD_z Prediction_Regional_z Prediction_Vertical_z];
%     Result.Fatigue_RMSE_z(s,:) = [RMSE_ALL_z RMSE_PSD_z RMSE_Regional_z RMSE_Vertical_z];   
%     Result.Fatigue_Error_z(s,:) = [Error_Rate_ALL_z Error_Rate_PSD_z Error_Rate_Regional_z Error_Rate_Vertical_z];   

 
    %% Save
    Result.Fatigue_Accuracy_30_2(s,:) = Result.Fatigue_Accuracy(s,:);
    Result.Fatigue_AUC_30_2(s,:) = Result.Fatigue_AUC(s,:);  
%     Result.Fatigue_Prediction_30_2(s,:) = Result.Fatigue_Prediction(s,:);
%     Result.Fatigue_RMSE_30_2(s,:) = Result.Fatigue_RMSE(s,:);
%     Result.Fatigue_Error_30_2(s,:) = Result.Fatigue_Error(s,:);
    
    Result.Fatigue_Accuracy_30_2_RPL(s,:) = Result.Fatigue_Accuracy_RPL(s,:);
    Result.Fatigue_AUC_30_2_RPL(s,:) = Result.Fatigue_AUC_RPL(s,:);  
%     Result.Fatigue_Prediction_30_2_RPL(s,:) = Result.Fatigue_Prediction_RPL(s,:);
%     Result.Fatigue_RMSE_30_2_RPL(s,:) = Result.Fatigue_RMSE_RPL(s,:);
%     Result.Fatigue_Error_30_2_RPL(s,:) = Result.Fatigue_Error_RPL(s,:);
    
    Result.Fatigue_Accuracy_30_2_z(s,:) = Result.Fatigue_Accuracy_z(s,:);
    Result.Fatigue_AUC_30_2_z(s,:) = Result.Fatigue_AUC_z(s,:);  
%     Result.Fatigue_Prediction_30_2_z(s,:) = Result.Fatigue_Prediction_z(s,:);
%     Result.Fatigue_RMSE_30_2_z(s,:) = Result.Fatigue_RMSE_z(s,:);
%     Result.Fatigue_Error_30_2_z(s,:) = Result.Fatigue_Error_z(s,:);

    
end
