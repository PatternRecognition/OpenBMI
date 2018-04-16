clear all; close all; clc;

load('BTB.mat')
global BTB
BTB.MatDir='D:\ICMI\Data\Distraction\';
subjectList = {'HSRYU_20180406_distraction','JHHWANG_20180407_distraction'};
saveFile = 'D:\ICMI\Figure\';
file = 'D:\ICMI\Data\Distraction\MATLAB';

for s = 1 : length(subjectList)
    [cnt, mrk, mnt] = eegfile_loadMatlab(strcat(file, '\', subjectList{s}));
       clear psd_feature_cca psd_feature_cca_f psd_feature_cca_c psd_feature_cca_t psd_feature_cca_o psd_feature_cca_p psd_feature_cca_f1 psd_feature_cca_f2 psd_feature_cca_f3 psd_feature_cca_c1 psd_feature_cca_c2 psd_feature_cca_c3 psd_feature_cca_p1 psd_feature_cca_p2 psd_feature_cca_p3 psd_feature_cca_t1 psd_feature_cca_t2 psd_feature_cca_o1 psd_feature_cca_o2 psd_feature_cca_o3;

    %% Band-pass Filter and Arteface Removal (ICA)
    
    [b, a] = butter(4, [0.5 50] / 100, 'bandpass');
    y = filter(b, a, cnt.x);
    cnt.x=cnt.x(:,1:64);
    cnt.clab=cnt.clab(:,1:64);
    
    cnt.x=fastica(cnt.x');
    cnt.x=cnt.x';
    
    %% Segmentation of bio-signals with 1 seconds length of epoch
        epoch = segmentationDistraction_R(cnt, mrk);
    %% Regional Channel Selection & Feature Extraction
    
    % PSD Feature Extraction
    [cca_delta,avg_delta] = spectrumAnalysisSleep_R(epoch.x, [0.5 4], cnt.fs);
    [cca_theta,avg_theta] = spectrumAnalysisSleep_R(epoch.x, [4 8], cnt.fs);
    [cca_alpha,avg_alpha] = spectrumAnalysisSleep_R(epoch.x, [8 13], cnt.fs);
    [cca_beta,avg_beta] = spectrumAnalysisSleep_R(epoch.x, [13 30], cnt.fs);
    [cca_gamma,avg_gamma] = spectrumAnalysisSleep_R(epoch.x, [30 50], cnt.fs);
    avg_delta = avg_delta'; avg_theta=avg_theta'; avg_alpha=avg_alpha'; avg_beta = avg_beta'; avg_gamma = avg_gamma'
    
    % RPL Feature Extraction
    cca_delta_RPL = cca_delta./(cca_delta+cca_theta+cca_alpha+cca_beta+cca_gamma);
    cca_theta_RPL = cca_theta./(cca_delta+cca_theta+cca_alpha+cca_beta+cca_gamma);
    cca_alpha_RPL = cca_alpha./(cca_delta+cca_theta+cca_alpha+cca_beta+cca_gamma);
    cca_beta_RPL = cca_beta./(cca_delta+cca_theta+cca_alpha+cca_beta+cca_gamma);
    cca_gamma_RPL = cca_gamma./(cca_delta+cca_theta+cca_alpha+cca_beta+cca_gamma);
    
    avg_delta_RPL = avg_delta./(avg_delta+avg_theta+avg_alpha+avg_beta+avg_gamma);
    avg_theta_RPL = avg_theta./(avg_delta+avg_theta+avg_alpha+avg_beta+avg_gamma);
    avg_alpha_RPL = avg_alpha./(avg_delta+avg_theta+avg_alpha+avg_beta+avg_gamma);
    avg_beta_RPL = avg_beta./(avg_delta+avg_theta+avg_alpha+avg_beta+avg_gamma);
    avg_gamma_RPL = avg_gamma./(avg_delta+avg_theta+avg_alpha+avg_beta+avg_gamma);
    
    %% T-Test
    
    % 일단 간단하게 최소 KSS +120 과 최대 KSS -120 데이터 가지고서 분류
    lowIdx = 1:180;
    highIdx = 200:380;
    
     % PSD T-TEST 분석
    [h, p, ci, stats] = ttest2(avg_delta(lowIdx), avg_delta(highIdx));
    p_delta(s) = p;
    [h, p, ci, stats] = ttest2(avg_theta(lowIdx), avg_theta(highIdx));
    p_theta(s) = p;
    [h, p, ci, stats] = ttest2(avg_alpha(lowIdx), avg_alpha(highIdx));
    p_alpha(s) = p;
    [h, p, ci, stats] = ttest2(avg_beta(lowIdx), avg_beta(highIdx));
    p_beta(s) = p;
    [h, p, ci, stats] = ttest2(avg_gamma(lowIdx), avg_gamma(highIdx));
    p_gamma(s) = p;
    
    p_analysis(s,1:5) = [p_delta(s), p_theta(s), p_alpha(s), p_beta(s), p_gamma(s)];
    p_analysis(s) = find(p_analysis(s,1:5)==min(p_analysis(s,1:5)));
    
    %% Classification
    
    feature_PSD = [avg_delta avg_theta avg_alpha avg_beta avg_gamma];
    feature_RPL = [avg_delta_RPL avg_theta_RPL avg_alpha_RPL avg_beta_RPL avg_gamma_RPL];
    
    [accuracy_PSD,AUC_PSD] = LDA_4CV(feature_PSD,highIdx,lowIdx,s);
    [accuracy_RPL,AUC_RPL] = LDA_4CV(feature_RPL,highIdx,lowIdx,s);
    
    Results_Distraction(s,1:4) = [accuracy_PSD,AUC_PSD,accuracy_RPL,AUC_RPL]
end