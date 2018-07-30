clear all; close all; clc;
load('BTB.mat')

subjectList = {'DNJO_20180414_workload','KTKIM_20180420_workload','DJLEE_20180421_workload'};
file = 'F:\Matlab\Data\Pilot\Workload\MATLAB\MATLAB2';
savefile = 'F:\Matlab\Plot\Workload';

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
     
    %% Segmentation of bio-signals with 1 seconds length of epoch
%     epoch = Segmentation_ICMI_Workload_1sec_EEG(cnt, mrk);
            epoch = SegmentationWorkload_EEG(cnt, mrk);
    
    %% Regional Channel
    
    channel_f=epoch.x(:,[1:5,31:36],:);
    channel_c=epoch.x(:,[6:9,11:13,16:19,39:40,43:46,48:50],:);
    channel_p=epoch.x(:,[21:25,52:55],:);
    channel_t=epoch.x(:,[10,14:15,20,37:48,41:42,47,51],:);
    channel_o=epoch.x(:,[26:30,56:60],:);
    
    channel_f1=epoch.x(:,[1,3,31,33],:);
    channel_f2=epoch.x(:,[2,5,32,36],:);
    channel_f3=epoch.x(:,[3,34:35],:);
    channel_c1=epoch.x(:,[6:7,11,16,17,39,43,44,48],:);
    channel_c2=epoch.x(:,[8:9,13,18,19,40,45,46,50],:);
    channel_c3=epoch.x(:,[12,49],:);
    channel_p1=epoch.x(:,[21:22,52,53],:);
    channel_p2=epoch.x(:,[24:25,54,55],:);
    channel_p3=epoch.x(:,[25,58],:);
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
    [cca.gamma, avg.gamma] = Power_spectrum(epoch.x, [30 50], cnt.fs);
    avg.delta = avg.delta'; avg.theta=avg.theta'; avg.alpha=avg.alpha'; avg.beta = avg.beta'; avg.gamma = avg.gamma';
    
    cca.PSD = [cca.delta cca.theta cca.alpha cca.beta cca.gamma];
    avg.PSD = [avg.delta avg.theta avg.alpha avg.beta avg.gamma];
    
    % Regional Feature Extraction
    [cca.delta_f,avg.delta_f] = Power_spectrum(channel_f, [0.5 4], cnt.fs);
    [cca.theta_f,avg.theta_f] = Power_spectrum(channel_f, [4 8], cnt.fs);
    [cca.alpha_f,avg.alpha_f] = Power_spectrum(channel_f, [8 13], cnt.fs);
    [cca.beta_f,avg.beta_f] = Power_spectrum(channel_f, [13 30], cnt.fs);
    [cca.gamma_f,avg.gamma_f] = Power_spectrum(channel_f, [30 50], cnt.fs);
    avg.delta_f = avg.delta_f'; avg.theta_f=avg.theta_f'; avg.alpha_f=avg.alpha_f'; avg.beta_f = avg.beta_f'; avg.gamma_f = avg.gamma_f';
    
    cca.Regional_f = [cca.delta_f cca.theta_f cca.alpha_f cca.beta_f cca.gamma_f];
    avg.Regional_f = [avg.delta_f avg.theta_f avg.alpha_f avg.beta_f avg.gamma_f];
    
    [cca.delta_c,avg.delta_c] = Power_spectrum(channel_c, [0.5 4], cnt.fs);
    [cca.theta_c,avg.theta_c] = Power_spectrum(channel_c, [4 8], cnt.fs);
    [cca.alpha_c,avg.alpha_c] = Power_spectrum(channel_c, [8 13], cnt.fs);
    [cca.beta_c,avg.beta_c] = Power_spectrum(channel_c, [13 30], cnt.fs);
    [cca.gamma_c,avg.gamma_c] = Power_spectrum(channel_c, [30 50], cnt.fs);
    avg.delta_c = avg.delta_c'; avg.theta_c=avg.theta_c'; avg.alpha_c=avg.alpha_c'; avg.beta_c = avg.beta_c'; avg.gamma_c = avg.gamma_c';
    
    cca.Regional_c = [cca.delta_c cca.theta_c cca.alpha_c cca.beta_c cca.gamma_c];
    avg.Regional_c = [avg.delta_c avg.theta_c avg.alpha_c avg.beta_c avg.gamma_c];
    
    [cca.delta_p,avg.delta_p] = Power_spectrum(channel_p, [0.5 4], cnt.fs);
    [cca.theta_p,avg.theta_p] = Power_spectrum(channel_p, [4 8], cnt.fs);
    [cca.alpha_p,avg.alpha_p] = Power_spectrum(channel_p, [8 13], cnt.fs);
    [cca.beta_p,avg.beta_p] = Power_spectrum(channel_p, [13 30], cnt.fs);
    [cca.gamma_p,avg.gamma_p] = Power_spectrum(channel_p, [30 50], cnt.fs);
    avg.delta_p = avg.delta_p'; avg.theta_p=avg.theta_p'; avg.alpha_p=avg.alpha_p'; avg.beta_p = avg.beta_p'; avg.gamma_p = avg.gamma_p';
    
    cca.Regional_p = [cca.delta_p cca.theta_p cca.alpha_p cca.beta_p cca.gamma_p];
    avg.Regional_p = [avg.delta_p avg.theta_p avg.alpha_p avg.beta_p avg.gamma_p];
    
    [cca.delta_t,avg.delta_t] = Power_spectrum(channel_t, [0.5 4], cnt.fs);
    [cca.theta_t,avg.theta_t] = Power_spectrum(channel_t, [4 8], cnt.fs);
    [cca.alpha_t,avg.alpha_t] = Power_spectrum(channel_t, [8 13], cnt.fs);
    [cca.beta_t,avg.beta_t] = Power_spectrum(channel_t, [13 30], cnt.fs);
    [cca.gamma_t,avg.gamma_t] = Power_spectrum(channel_t, [30 50], cnt.fs);
    avg.delta_t = avg.delta_t'; avg.theta_t=avg.theta_t'; avg.alpha_t=avg.alpha_t'; avg.beta_t = avg.beta_t'; avg.gamma_t = avg.gamma_t';
    
    cca.Regional_t = [cca.delta_t cca.theta_t cca.alpha_t cca.beta_t cca.gamma_t];
    avg.Regional_t = [avg.delta_t avg.theta_t avg.alpha_t avg.beta_t avg.gamma_t];
    
    [cca.delta_o,avg.delta_o] = Power_spectrum(channel_o, [0.5 4], cnt.fs);
    [cca.theta_o,avg.theta_o] = Power_spectrum(channel_o, [4 8], cnt.fs);
    [cca.alpha_o,avg.alpha_o] = Power_spectrum(channel_o, [8 13], cnt.fs);
    [cca.beta_o,avg.beta_o] = Power_spectrum(channel_o, [13 30], cnt.fs);
    [cca.gamma_o,avg.gamma_o] = Power_spectrum(channel_o, [30 50], cnt.fs);
    avg.delta_o = avg.delta_o'; avg.theta_o=avg.theta_o'; avg.alpha_o=avg.alpha_o'; avg.beta_o = avg.beta_o'; avg.gamma_o = avg.gamma_o';
    
    cca.Regional_o = [cca.delta_o cca.theta_o cca.alpha_o cca.beta_o cca.gamma_o];
    avg.Regional_o = [avg.delta_o avg.theta_o avg.alpha_o avg.beta_o avg.gamma_o];
    
    
    % Regional Vertical Feature Extraction
    [cca.delta_f1,avg.delta_f1] = Power_spectrum(channel_f1, [0.5 4], cnt.fs);
    [cca.theta_f1,avg.theta_f1] = Power_spectrum(channel_f1, [4 8], cnt.fs);
    [cca.alpha_f1,avg.alpha_f1] = Power_spectrum(channel_f1, [8 13], cnt.fs);
    [cca.beta_f1,avg.beta_f1] = Power_spectrum(channel_f1, [13 30], cnt.fs);
    [cca.gamma_f1,avg.gamma_f1] = Power_spectrum(channel_f1, [30 50], cnt.fs);
    avg.delta_f1 = avg.delta_f1'; avg.theta_f1=avg.theta_f1'; avg.alpha_f1=avg.alpha_f1'; avg.beta_f1 = avg.beta_f1'; avg.gamma_f1 = avg.gamma_f1';
    
    cca.Vertical_f1 = [cca.delta_f1 cca.theta_f1 cca.alpha_f1 cca.beta_f1 cca.gamma_f1];
    avg.Vertical_f1 = [avg.delta_f1 avg.theta_f1 avg.alpha_f1 avg.beta_f1 avg.gamma_f1];
    
    
    [cca.delta_f2,avg.delta_f2] = Power_spectrum(channel_f2, [0.5 4], cnt.fs);
    [cca.theta_f2,avg.theta_f2] = Power_spectrum(channel_f2, [4 8], cnt.fs);
    [cca.alpha_f2,avg.alpha_f2] = Power_spectrum(channel_f2, [8 13], cnt.fs);
    [cca.beta_f2,avg.beta_f2] = Power_spectrum(channel_f2, [13 30], cnt.fs);
    [cca.gamma_f2,avg.gamma_f2] = Power_spectrum(channel_f2, [30 50], cnt.fs);
    avg.delta_f2 = avg.delta_f2'; avg.theta_f2=avg.theta_f2'; avg.alpha_f2=avg.alpha_f2'; avg.beta_f2 = avg.beta_f2'; avg.gamma_f2 = avg.gamma_f2';
    
    cca.Vertical_f2 = [cca.delta_f2 cca.theta_f2 cca.alpha_f2 cca.beta_f2 cca.gamma_f2];
    avg.Vertical_f2 = [avg.delta_f2 avg.theta_f2 avg.alpha_f2 avg.beta_f2 avg.gamma_f2];
    
    [cca.delta_f3,avg.delta_f3] = Power_spectrum(channel_f3, [0.5 4], cnt.fs);
    [cca.theta_f3,avg.theta_f3] = Power_spectrum(channel_f3, [4 8], cnt.fs);
    [cca.alpha_f3,avg.alpha_f3] = Power_spectrum(channel_f3, [8 13], cnt.fs);
    [cca.beta_f3,avg.beta_f3] = Power_spectrum(channel_f3, [13 30], cnt.fs);
    [cca.gamma_f3,avg.gamma_f3] = Power_spectrum(channel_f3, [30 50], cnt.fs);
    avg.delta_f3 = avg.delta_f3'; avg.theta_f3=avg.theta_f3'; avg.alpha_f3=avg.alpha_f3'; avg.beta_f3 = avg.beta_f3'; avg.gamma_f3 = avg.gamma_f3';
    
    cca.Vertical_f3 = [cca.delta_f3 cca.theta_f3 cca.alpha_f3 cca.beta_f3 cca.gamma_f3];
    avg.Vertical_f3 = [avg.delta_f3 avg.theta_f3 avg.alpha_f3 avg.beta_f3 avg.gamma_f3];
    
    
    [cca.delta_c1,avg.delta_c1] = Power_spectrum(channel_c1, [0.5 4], cnt.fs);
    [cca.theta_c1,avg.theta_c1] = Power_spectrum(channel_c1, [4 8], cnt.fs);
    [cca.alpha_c1,avg.alpha_c1] = Power_spectrum(channel_c1, [8 13], cnt.fs);
    [cca.beta_c1,avg.beta_c1] = Power_spectrum(channel_c1, [13 30], cnt.fs);
    [cca.gamma_c1,avg.gamma_c1] = Power_spectrum(channel_c1, [30 50], cnt.fs);
    avg.delta_c1 = avg.delta_c1'; avg.theta_c1=avg.theta_c1'; avg.alpha_c1=avg.alpha_c1'; avg.beta_c1 = avg.beta_c1'; avg.gamma_c1 = avg.gamma_c1';
    
    cca.Vertical_c1 = [cca.delta_c1 cca.theta_c1 cca.alpha_c1 cca.beta_c1 cca.gamma_c1];
    avg.Vertical_c1 = [avg.delta_c1 avg.theta_c1 avg.alpha_c1 avg.beta_c1 avg.gamma_c1];
    
    
    [cca.delta_c2,avg.delta_c2] = Power_spectrum(channel_c2, [0.5 4], cnt.fs);
    [cca.theta_c2,avg.theta_c2] = Power_spectrum(channel_c2, [4 8], cnt.fs);
    [cca.alpha_c2,avg.alpha_c2] = Power_spectrum(channel_c2, [8 13], cnt.fs);
    [cca.beta_c2,avg.beta_c2] = Power_spectrum(channel_c2, [13 30], cnt.fs);
    [cca.gamma_c2,avg.gamma_c2] = Power_spectrum(channel_c2, [30 50], cnt.fs);
    avg.delta_c2 = avg.delta_c2'; avg.theta_c2=avg.theta_c2'; avg.alpha_c2=avg.alpha_c2'; avg.beta_c2 = avg.beta_c2'; avg.gamma_c2 = avg.gamma_c2';
    
    cca.Vertical_c2 = [cca.delta_c2 cca.theta_c2 cca.alpha_c2 cca.beta_c2 cca.gamma_c2];
    avg.Vertical_c2 = [avg.delta_c2 avg.theta_c2 avg.alpha_c2 avg.beta_c2 avg.gamma_c2];
    
    
    [cca.delta_c3,avg.delta_c3] = Power_spectrum(channel_c3, [0.5 4], cnt.fs);
    [cca.theta_c3,avg.theta_c3] = Power_spectrum(channel_c3, [4 8], cnt.fs);
    [cca.alpha_c3,avg.alpha_c3] = Power_spectrum(channel_c3, [8 13], cnt.fs);
    [cca.beta_c3,avg.beta_c3] = Power_spectrum(channel_c3, [13 30], cnt.fs);
    [cca.gamma_c3,avg.gamma_c3] = Power_spectrum(channel_c3, [30 50], cnt.fs);
    avg.delta_c3 = avg.delta_c3'; avg.theta_c3=avg.theta_c3'; avg.alpha_c3=avg.alpha_c3'; avg.beta_c3 = avg.beta_c3'; avg.gamma_c3 = avg.gamma_c3';
    
    cca.Vertical_c3 = [cca.delta_c3 cca.theta_c3 cca.alpha_c3 cca.beta_c3 cca.gamma_c3];
    avg.Vertical_c3 = [avg.delta_c3 avg.theta_c3 avg.alpha_c3 avg.beta_c3 avg.gamma_c3];
    
    
    [cca.delta_p1,avg.delta_p1] = Power_spectrum(channel_p1, [0.5 4], cnt.fs);
    [cca.theta_p1,avg.theta_p1] = Power_spectrum(channel_p1, [4 8], cnt.fs);
    [cca.alpha_p1,avg.alpha_p1] = Power_spectrum(channel_p1, [8 13], cnt.fs);
    [cca.beta_p1,avg.beta_p1] = Power_spectrum(channel_p1, [13 30], cnt.fs);
    [cca.gamma_p1,avg.gamma_p1] = Power_spectrum(channel_p1, [30 50], cnt.fs);
    avg.delta_p1 = avg.delta_p1'; avg.theta_p1=avg.theta_p1'; avg.alpha_p1=avg.alpha_p1'; avg.beta_p1 = avg.beta_p1'; avg.gamma_p1 = avg.gamma_p1';
    
    cca.Vertical_p1 = [cca.delta_p1 cca.theta_p1 cca.alpha_p1 cca.beta_p1 cca.gamma_p1];
    avg.Vertical_p1 = [avg.delta_p1 avg.theta_p1 avg.alpha_p1 avg.beta_p1 avg.gamma_p1];
    
    
    [cca.delta_p2,avg.delta_p2] = Power_spectrum(channel_p2, [0.5 4], cnt.fs);
    [cca.theta_p2,avg.theta_p2] = Power_spectrum(channel_p2, [4 8], cnt.fs);
    [cca.alpha_p2,avg.alpha_p2] = Power_spectrum(channel_p2, [8 13], cnt.fs);
    [cca.beta_p2,avg.beta_p2] = Power_spectrum(channel_p2, [13 30], cnt.fs);
    [cca.gamma_p2,avg.gamma_p2] = Power_spectrum(channel_p2, [30 50], cnt.fs);
    avg.delta_p2 = avg.delta_p2'; avg.theta_p2=avg.theta_p2'; avg.alpha_p2=avg.alpha_p2'; avg.beta_p2 = avg.beta_p2'; avg.gamma_p2 = avg.gamma_p2';
    
    cca.Vertical_p2 = [cca.delta_p2 cca.theta_p2 cca.alpha_p2 cca.beta_p2 cca.gamma_p2];
    avg.Vertical_p2 = [avg.delta_p2 avg.theta_p2 avg.alpha_p2 avg.beta_p2 avg.gamma_p2];
    
    
    [cca.delta_p3,avg.delta_p3] = Power_spectrum(channel_p3, [0.5 4], cnt.fs);
    [cca.theta_p3,avg.theta_p3] = Power_spectrum(channel_p3, [4 8], cnt.fs);
    [cca.alpha_p3,avg.alpha_p3] = Power_spectrum(channel_p3, [8 13], cnt.fs);
    [cca.beta_p3,avg.beta_p3] = Power_spectrum(channel_p3, [13 30], cnt.fs);
    [cca.gamma_p3,avg.gamma_p3] = Power_spectrum(channel_p3, [30 50], cnt.fs);
    avg.delta_p3 = avg.delta_p3'; avg.theta_p3=avg.theta_p3'; avg.alpha_p3=avg.alpha_p3'; avg.beta_p3 = avg.beta_p3'; avg.gamma_p3 = avg.gamma_p3';
    
    cca.Vertical_p3 = [cca.delta_p3 cca.theta_p3 cca.alpha_p3 cca.beta_p3 cca.gamma_p3];
    avg.Vertical_p3 = [avg.delta_p3 avg.theta_p3 avg.alpha_p3 avg.beta_p3 avg.gamma_p3];
    
    
    
    [cca.delta_t1,avg.delta_t1] = Power_spectrum(channel_t1, [0.5 4], cnt.fs);
    [cca.theta_t1,avg.theta_t1] = Power_spectrum(channel_t1, [4 8], cnt.fs);
    [cca.alpha_t1,avg.alpha_t1] = Power_spectrum(channel_t1, [8 13], cnt.fs);
    [cca.beta_t1,avg.beta_t1] = Power_spectrum(channel_t1, [13 30], cnt.fs);
    [cca.gamma_t1,avg.gamma_t1] = Power_spectrum(channel_t1, [30 50], cnt.fs);
    avg.delta_t1 = avg.delta_t1'; avg.theta_t1=avg.theta_t1'; avg.alpha_t1=avg.alpha_t1'; avg.beta_t1 = avg.beta_t1'; avg.gamma_t1 = avg.gamma_t1';
    
    cca.Vertical_t1 = [cca.delta_t1 cca.theta_t1 cca.alpha_t1 cca.beta_t1 cca.gamma_t1];
    avg.Vertical_t1 = [avg.delta_t1 avg.theta_t1 avg.alpha_t1 avg.beta_t1 avg.gamma_t1];
    
    
    [cca.delta_t2,avg.delta_t2] = Power_spectrum(channel_t2, [0.5 4], cnt.fs);
    [cca.theta_t2,avg.theta_t2] = Power_spectrum(channel_t2, [4 8], cnt.fs);
    [cca.alpha_t2,avg.alpha_t2] = Power_spectrum(channel_t2, [8 13], cnt.fs);
    [cca.beta_t2,avg.beta_t2] = Power_spectrum(channel_t2, [13 30], cnt.fs);
    [cca.gamma_t2,avg.gamma_t2] = Power_spectrum(channel_t2, [30 50], cnt.fs);
    avg.delta_t2 = avg.delta_t2'; avg.theta_t2=avg.theta_t2'; avg.alpha_t2=avg.alpha_t2'; avg.beta_t2 = avg.beta_t2'; avg.gamma_t2 = avg.gamma_t2';
    
    cca.Vertical_t2 = [cca.delta_t2 cca.theta_t2 cca.alpha_t2 cca.beta_t2 cca.gamma_t2];
    avg.Vertical_t2 = [avg.delta_t2 avg.theta_t2 avg.alpha_t2 avg.beta_t2 avg.gamma_t2];
    
    
    [cca.delta_o1,avg.delta_o1] = Power_spectrum(channel_o1, [0.5 4], cnt.fs);
    [cca.theta_o1,avg.theta_o1] = Power_spectrum(channel_o1, [4 8], cnt.fs);
    [cca.alpha_o1,avg.alpha_o1] = Power_spectrum(channel_o1, [8 13], cnt.fs);
    [cca.beta_o1,avg.beta_o1] = Power_spectrum(channel_o1, [13 30], cnt.fs);
    [cca.gamma_o1,avg.gamma_o1] = Power_spectrum(channel_o1, [30 50], cnt.fs);
    avg.delta_o1 = avg.delta_o1'; avg.theta_o1=avg.theta_o1'; avg.alpha_o1=avg.alpha_o1'; avg.beta_o1 = avg.beta_o1'; avg.gamma_o1 = avg.gamma_o1';
    
    cca.Vertical_o1 = [cca.delta_o1 cca.theta_o1 cca.alpha_o1 cca.beta_o1 cca.gamma_o1];
    avg.Vertical_o1 = [avg.delta_o1 avg.theta_o1 avg.alpha_o1 avg.beta_o1 avg.gamma_o1];
    
    [cca.delta_o2,avg.delta_o2] = Power_spectrum(channel_o2, [0.5 4], cnt.fs);
    [cca.theta_o2,avg.theta_o2] = Power_spectrum(channel_o2, [4 8], cnt.fs);
    [cca.alpha_o2,avg.alpha_o2] = Power_spectrum(channel_o2, [8 13], cnt.fs);
    [cca.beta_o2,avg.beta_o2] = Power_spectrum(channel_o2, [13 30], cnt.fs);
    [cca.gamma_o2,avg.gamma_o2] = Power_spectrum(channel_o2, [30 50], cnt.fs);
    avg.delta_o2 = avg.delta_o2'; avg.theta_o2=avg.theta_o2'; avg.alpha_o2=avg.alpha_o2'; avg.beta_o2 = avg.beta_o2'; avg.gamma_o2 = avg.gamma_o2';
    
    cca.Vertical_o2 = [cca.delta_o2 cca.theta_o2 cca.alpha_o2 cca.beta_o2 cca.gamma_o2];
    avg.Vertical_o2 = [avg.delta_o2 avg.theta_o2 avg.alpha_o2 avg.beta_o2 avg.gamma_o2];
    
    
    
    [cca.delta_o3,avg.delta_o3] = Power_spectrum(channel_o3, [0.5 4], cnt.fs);
    [cca.theta_o3,avg.theta_o3] = Power_spectrum(channel_o3, [4 8], cnt.fs);
    [cca.alpha_o3,avg.alpha_o3] = Power_spectrum(channel_o3, [8 13], cnt.fs);
    [cca.beta_o3,avg.beta_o3] = Power_spectrum(channel_o3, [13 30], cnt.fs);
    [cca.gamma_o3,avg.gamma_o3] = Power_spectrum(channel_o3, [30 50], cnt.fs);
    avg.delta_o3 = avg.delta_o3'; avg.theta_o3=avg.theta_o3'; avg.alpha_o3=avg.alpha_o3'; avg.beta_o3 = avg.beta_o3'; avg.gamma_o3 = avg.gamma_o3';
    
    cca.Vertical_o3 = [cca.delta_o3 cca.theta_o3 cca.alpha_o3 cca.beta_o3 cca.gamma_o3];
    avg.Vertical_o3 = [avg.delta_o3 avg.theta_o3 avg.alpha_o3 avg.beta_o3 avg.gamma_o3];
    
    %% T-Test
    
    lowIdx = (epoch.label==0);
    lowIdx = find(lowIdx==1);
    lowIdx = lowIdx';
    lowIdx = lowIdx(:,1);
    
    highIdx = (epoch.label==3);
    highIdx = find(highIdx==1);
    highIdx = highIdx';
    highIdx = highIdx(:,1);
    
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
    
    Result.ttest.totallv0(s,:) = p_analysis(s,1:5);
    
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
    
    Result.ttest.regionallv0(s,:) = [p_analysis_f(s,:) p_analysis_c(s,:) p_analysis_p(s,:) p_analysis_t(s,:) p_analysis_o(s,:)];
    
    
    % Regional Vertical T-TEST 분석
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
    
    Result.ttest.verticallv0(s,:) = [p_analysis_f1(s,:) p_analysis_f2(s,:) p_analysis_f3(s,:) p_analysis_c1(s,:) p_analysis_c2(s,:) p_analysis_c3(s,:) p_analysis_p1(s,:) p_analysis_p2(s,:) p_analysis_p3(s,:) p_analysis_t1(s,:) p_analysis_t2(s,:) p_analysis_o1(s,:) p_analysis_o2(s,:) p_analysis_o3(s,:)];
    
    %% Feature Selection
    feature_ALL = avg.PSD;
    feature_PSD = avg.PSD(:,find(p_analysis(s,:)<0.05));
    feature_Regional = [avg.Regional_f(:,find(p_analysis_f(s,:)<0.05)),avg.Regional_c(:,find(p_analysis_c(s,:)<0.05)),avg.Regional_p(:,find(p_analysis_p(s,:)<0.05)),avg.Regional_t(:,find(p_analysis_t(s,:)<0.05)),avg.Regional_o(:,find(p_analysis_o(s,:)<0.05))];
    feature_Vertical = [avg.Vertical_f1(:,find(p_analysis_f1(s,:)<0.05)),avg.Vertical_f2(:,find(p_analysis_f2(s,:)<0.05)),avg.Vertical_f3(:,find(p_analysis_f3(s,:)<0.05)),avg.Vertical_c1(:,find(p_analysis_c1(s,:)<0.05)),avg.Vertical_c2(:,find(p_analysis_c2(s,:)<0.05)),avg.Vertical_c3(:,find(p_analysis_c3(s,:)<0.05)),avg.Vertical_p1(:,find(p_analysis_p1(s,:)<0.05)),avg.Vertical_p2(:,find(p_analysis_p2(s,:)<0.05)),avg.Vertical_p3(:,find(p_analysis_p3(s,:)<0.05)),avg.Vertical_t1(:,find(p_analysis_t1(s,:)<0.05)),avg.Vertical_t2(:,find(p_analysis_t2(s,:)<0.05)),avg.Vertical_o1(:,find(p_analysis_o1(s,:)<0.05)),avg.Vertical_o2(:,find(p_analysis_o2(s,:)<0.05)),avg.Vertical_o3(:,find(p_analysis_o3(s,:)<0.05))];
    
    %% Classification
    [Accuracy_ALL AUC_ALL] = LDA_4CV(feature_ALL, highIdx, lowIdx, s);
    [Accuracy_PSD AUC_PSD] = LDA_4CV(feature_PSD, highIdx, lowIdx, s);
    [Accuracy_Regional AUC_Regional] = LDA_4CV(feature_Regional, highIdx, lowIdx, s);
    [Accuracy_Vertical AUC_Vertical] = LDA_4CV(feature_Vertical, highIdx, lowIdx, s);
    
    Results_Workload_0_Accuracy(s,:) = [Accuracy_ALL Accuracy_PSD Accuracy_Regional Accuracy_Vertical];
    Results_Workload_0_AUC(s,:) = [AUC_ALL AUC_PSD AUC_Regional AUC_Vertical];
    
    cca.total_lv0_normal(:,:,s) = mean(cca.PSD(lowIdx,:));
    cca.totalstd_lv0_normal(:,:,s) = [std(cca.delta(lowIdx,:)) std(cca.theta(lowIdx,:)) std(cca.alpha(lowIdx,:)) std(cca.beta(lowIdx,:)) std(cca.gamma(lowIdx,:))];
    
    cca.total_lv0_workload(:,:,s) = mean(cca.PSD(highIdx,:));
    cca.totalstd_lv0_workload(:,:,s) = [std(cca.delta(highIdx,:)) std(cca.theta(highIdx,:)) std(cca.alpha(highIdx,:)) std(cca.beta(highIdx,:)) std(cca.gamma(highIdx,:))];
    
        epoch.label(end) = [];
     
    
    %% Classification : 레벨1 vs 레벨2
    
    lowIdx = (epoch.label==1);
    lowIdx = find(lowIdx==1);
    lowIdx = lowIdx';
    lowIdx = lowIdx(:,1);
    
    highIdx = (epoch.label==2);
    highIdx = find(highIdx==1);
    highIdx = highIdx';
    highIdx = highIdx(:,1);
    
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
    
    Result.ttest.totallv1(s,:) = p_analysis(s,1:5);
    
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
    
    Result.ttest.regionallv1(s,:) = [p_analysis_f(s,:) p_analysis_c(s,:) p_analysis_p(s,:) p_analysis_t(s,:) p_analysis_o(s,:)];
    
    
    % Regional Vertical T-TEST 분석
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
    
    Result.ttest.verticallv1(s,:) = [p_analysis_f1(s,:) p_analysis_f2(s,:) p_analysis_f3(s,:) p_analysis_c1(s,:) p_analysis_c2(s,:) p_analysis_c3(s,:) p_analysis_p1(s,:) p_analysis_p2(s,:) p_analysis_p3(s,:) p_analysis_t1(s,:) p_analysis_t2(s,:) p_analysis_o1(s,:) p_analysis_o2(s,:) p_analysis_o3(s,:)];
    
    %% Feature Selection
    feature_ALL = avg.PSD;
    feature_PSD = avg.PSD(:,find(p_analysis(s,:)<0.05));
    feature_Regional = [avg.Regional_f(:,find(p_analysis_f(s,:)<0.05)),avg.Regional_c(:,find(p_analysis_c(s,:)<0.05)),avg.Regional_p(:,find(p_analysis_p(s,:)<0.05)),avg.Regional_t(:,find(p_analysis_t(s,:)<0.05)),avg.Regional_o(:,find(p_analysis_o(s,:)<0.05))];
    feature_Vertical = [avg.Vertical_f1(:,find(p_analysis_f1(s,:)<0.05)),avg.Vertical_f2(:,find(p_analysis_f2(s,:)<0.05)),avg.Vertical_f3(:,find(p_analysis_f3(s,:)<0.05)),avg.Vertical_c1(:,find(p_analysis_c1(s,:)<0.05)),avg.Vertical_c2(:,find(p_analysis_c2(s,:)<0.05)),avg.Vertical_c3(:,find(p_analysis_c3(s,:)<0.05)),avg.Vertical_p1(:,find(p_analysis_p1(s,:)<0.05)),avg.Vertical_p2(:,find(p_analysis_p2(s,:)<0.05)),avg.Vertical_p3(:,find(p_analysis_p3(s,:)<0.05)),avg.Vertical_t1(:,find(p_analysis_t1(s,:)<0.05)),avg.Vertical_t2(:,find(p_analysis_t2(s,:)<0.05)),avg.Vertical_o1(:,find(p_analysis_o1(s,:)<0.05)),avg.Vertical_o2(:,find(p_analysis_o2(s,:)<0.05)),avg.Vertical_o3(:,find(p_analysis_o3(s,:)<0.05))];
    
    
    %% Classification
    [Accuracy_ALL AUC_ALL] = LDA_4CV(feature_ALL, highIdx, lowIdx, s);
    [Accuracy_PSD AUC_PSD] = LDA_4CV(feature_PSD, highIdx, lowIdx, s);
    [Accuracy_Regional AUC_Regional] = LDA_4CV(feature_Regional, highIdx, lowIdx, s);
    [Accuracy_Vertical AUC_Vertical] = LDA_4CV(feature_Vertical, highIdx, lowIdx, s);
    
    Results_Workload_1_Accuracy(s,:) = [Accuracy_ALL Accuracy_PSD Accuracy_Regional Accuracy_Vertical];
    Results_Workload_1_AUC(s,:) = [AUC_ALL AUC_PSD AUC_Regional AUC_Vertical];
    
    cca.total_lv1_normal(:,:,s) = mean(cca.PSD(lowIdx,:));
    cca.totalstd_lv1_normal(:,:,s) = [std(cca.delta(lowIdx,:)) std(cca.theta(lowIdx,:)) std(cca.alpha(lowIdx,:)) std(cca.beta(lowIdx,:)) std(cca.gamma(lowIdx,:))];
    
    cca.total_lv1_workload(:,:,s) = mean(cca.PSD(highIdx,:));
    cca.totalstd_lv1_workload(:,:,s) = [std(cca.delta(highIdx,:)) std(cca.theta(highIdx,:)) std(cca.alpha(highIdx,:)) std(cca.beta(highIdx,:)) std(cca.gamma(highIdx,:))];
    
    %% Classification : 레벨1 vs 레벨3
    
    lowIdx = (epoch.label==1);
    lowIdx = find(lowIdx==1);
    lowIdx = lowIdx';
    lowIdx = lowIdx(:,1);
    
    highIdx = (epoch.label==3);
    highIdx = find(highIdx==1);
    highIdx = highIdx';
    highIdx = highIdx(:,1);
    
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
    
    Result.ttest.totallv2(s,:) = p_analysis(s,1:5);
    
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
    
    Result.ttest.regionallv2(s,:) = [p_analysis_f(s,:) p_analysis_c(s,:) p_analysis_p(s,:) p_analysis_t(s,:) p_analysis_o(s,:)];
    
    
    % Regional Vertical T-TEST 분석
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
    
    Result.ttest.verticallv2(s,:) = [p_analysis_f1(s,:) p_analysis_f2(s,:) p_analysis_f3(s,:) p_analysis_c1(s,:) p_analysis_c2(s,:) p_analysis_c3(s,:) p_analysis_p1(s,:) p_analysis_p2(s,:) p_analysis_p3(s,:) p_analysis_t1(s,:) p_analysis_t2(s,:) p_analysis_o1(s,:) p_analysis_o2(s,:) p_analysis_o3(s,:)];
    
    %% Feature Selection
    feature_ALL = avg.PSD;
    feature_PSD = avg.PSD(:,find(p_analysis(s,:)<0.05));
    feature_Regional = [avg.Regional_f(:,find(p_analysis_f(s,:)<0.05)),avg.Regional_c(:,find(p_analysis_c(s,:)<0.05)),avg.Regional_p(:,find(p_analysis_p(s,:)<0.05)),avg.Regional_t(:,find(p_analysis_t(s,:)<0.05)),avg.Regional_o(:,find(p_analysis_o(s,:)<0.05))];
    feature_Vertical = [avg.Vertical_f1(:,find(p_analysis_f1(s,:)<0.05)),avg.Vertical_f2(:,find(p_analysis_f2(s,:)<0.05)),avg.Vertical_f3(:,find(p_analysis_f3(s,:)<0.05)),avg.Vertical_c1(:,find(p_analysis_c1(s,:)<0.05)),avg.Vertical_c2(:,find(p_analysis_c2(s,:)<0.05)),avg.Vertical_c3(:,find(p_analysis_c3(s,:)<0.05)),avg.Vertical_p1(:,find(p_analysis_p1(s,:)<0.05)),avg.Vertical_p2(:,find(p_analysis_p2(s,:)<0.05)),avg.Vertical_p3(:,find(p_analysis_p3(s,:)<0.05)),avg.Vertical_t1(:,find(p_analysis_t1(s,:)<0.05)),avg.Vertical_t2(:,find(p_analysis_t2(s,:)<0.05)),avg.Vertical_o1(:,find(p_analysis_o1(s,:)<0.05)),avg.Vertical_o2(:,find(p_analysis_o2(s,:)<0.05)),avg.Vertical_o3(:,find(p_analysis_o3(s,:)<0.05))];
    
    %% Classification
    [Accuracy_ALL AUC_ALL] = LDA_4CV(feature_ALL, highIdx, lowIdx, s);
    [Accuracy_PSD AUC_PSD] = LDA_4CV(feature_PSD, highIdx, lowIdx, s);
    [Accuracy_Regional AUC_Regional] = LDA_4CV(feature_Regional, highIdx, lowIdx, s);
    [Accuracy_Vertical AUC_Vertical] = LDA_4CV(feature_Vertical, highIdx, lowIdx, s);
    
    Results_Workload_2_Accuracy(s,:) = [Accuracy_ALL Accuracy_PSD Accuracy_Regional Accuracy_Vertical];
    Results_Workload_2_AUC(s,:) = [AUC_ALL AUC_PSD AUC_Regional AUC_Vertical];
    
    cca.total_lv2_normal(:,:,s) = mean(cca.PSD(lowIdx,:));
    cca.totalstd_lv2_normal(:,:,s) = [std(cca.delta(lowIdx,:)) std(cca.theta(lowIdx,:)) std(cca.alpha(lowIdx,:)) std(cca.beta(lowIdx,:)) std(cca.gamma(lowIdx,:))];
    
    cca.total_lv2_workload(:,:,s) = mean(cca.PSD(highIdx,:));
    cca.totalstd_lv2_workload(:,:,s) = [std(cca.delta(highIdx,:)) std(cca.theta(highIdx,:)) std(cca.alpha(highIdx,:)) std(cca.beta(highIdx,:)) std(cca.gamma(highIdx,:))];
    
    %% Classification : 레벨2 vs 레벨3
    
    lowIdx = (epoch.label==2);
    lowIdx = find(lowIdx==1);
    lowIdx = lowIdx';
    lowIdx = lowIdx(:,1);
    
    highIdx = (epoch.label==3);
    highIdx = find(highIdx==1);
    highIdx = highIdx';
    highIdx = highIdx(:,1);
    
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
    
    Result.ttest.totallv3(s,:) = p_analysis(s,1:5);
    
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
    
    Result.ttest.regionallv3(s,:) = [p_analysis_f(s,:) p_analysis_c(s,:) p_analysis_p(s,:) p_analysis_t(s,:) p_analysis_o(s,:)];
    
    
    % Regional Vertical T-TEST 분석
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
    
    Result.ttest.verticallv3(s,:) = [p_analysis_f1(s,:) p_analysis_f2(s,:) p_analysis_f3(s,:) p_analysis_c1(s,:) p_analysis_c2(s,:) p_analysis_c3(s,:) p_analysis_p1(s,:) p_analysis_p2(s,:) p_analysis_p3(s,:) p_analysis_t1(s,:) p_analysis_t2(s,:) p_analysis_o1(s,:) p_analysis_o2(s,:) p_analysis_o3(s,:)];
    
    %% Feature Selection
    feature_ALL = avg.PSD;
    feature_PSD = avg.PSD(:,find(p_analysis(s,:)<0.05));
    feature_Regional = [avg.Regional_f(:,find(p_analysis_f(s,:)<0.05)),avg.Regional_c(:,find(p_analysis_c(s,:)<0.05)),avg.Regional_p(:,find(p_analysis_p(s,:)<0.05)),avg.Regional_t(:,find(p_analysis_t(s,:)<0.05)),avg.Regional_o(:,find(p_analysis_o(s,:)<0.05))];
    feature_Vertical = [avg.Vertical_f1(:,find(p_analysis_f1(s,:)<0.05)),avg.Vertical_f2(:,find(p_analysis_f2(s,:)<0.05)),avg.Vertical_f3(:,find(p_analysis_f3(s,:)<0.05)),avg.Vertical_c1(:,find(p_analysis_c1(s,:)<0.05)),avg.Vertical_c2(:,find(p_analysis_c2(s,:)<0.05)),avg.Vertical_c3(:,find(p_analysis_c3(s,:)<0.05)),avg.Vertical_p1(:,find(p_analysis_p1(s,:)<0.05)),avg.Vertical_p2(:,find(p_analysis_p2(s,:)<0.05)),avg.Vertical_p3(:,find(p_analysis_p3(s,:)<0.05)),avg.Vertical_t1(:,find(p_analysis_t1(s,:)<0.05)),avg.Vertical_t2(:,find(p_analysis_t2(s,:)<0.05)),avg.Vertical_o1(:,find(p_analysis_o1(s,:)<0.05)),avg.Vertical_o2(:,find(p_analysis_o2(s,:)<0.05)),avg.Vertical_o3(:,find(p_analysis_o3(s,:)<0.05))];
    
    %% Classification
    [Accuracy_ALL AUC_ALL] = LDA_4CV(feature_ALL, highIdx, lowIdx, s);
    [Accuracy_PSD AUC_PSD] = LDA_4CV(feature_PSD, highIdx, lowIdx, s);
    [Accuracy_Regional AUC_Regional] = LDA_4CV(feature_Regional, highIdx, lowIdx, s);
    [Accuracy_Vertical AUC_Vertical] = LDA_4CV(feature_Vertical, highIdx, lowIdx, s);
    
    Results_Workload_3_Accuracy(s,:) = [Accuracy_ALL Accuracy_PSD Accuracy_Regional Accuracy_Vertical];
    Results_Workload_3_AUC(s,:) = [AUC_ALL AUC_PSD AUC_Regional AUC_Vertical];
    
    cca.total_lv3_normal(:,:,s) = mean(cca.PSD(lowIdx,:));
    cca.totalstd_lv3_normal(:,:,s) = [std(cca.delta(lowIdx,:)) std(cca.theta(lowIdx,:)) std(cca.alpha(lowIdx,:)) std(cca.beta(lowIdx,:)) std(cca.gamma(lowIdx,:))];
    
    cca.total_lv3_workload(:,:,s) = mean(cca.PSD(highIdx,:));
    cca.totalstd_lv3_workload(:,:,s) = [std(cca.delta(highIdx,:)) std(cca.theta(highIdx,:)) std(cca.alpha(highIdx,:)) std(cca.beta(highIdx,:)) std(cca.gamma(highIdx,:))];
    
end
