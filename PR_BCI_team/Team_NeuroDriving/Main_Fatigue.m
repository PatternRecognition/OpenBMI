clear all; close all; clc;

load('BTB.mat')
global BTB
BTB.MatDir='D:\ICMI\Data\Fatigue\';
subjectList = {'HSRYU_20180406_fatigue','JHHWANG_20180407_fatigue'};
saveFile = 'D:\ICMI\Figure\';
file = 'D:\ICMI\Data\Fatigue\MATLAB';

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
    
    epoch = segmentationSleep_R(cnt, mrk, interKSS);
    kssIdx = find(strcmp(epoch.mClab, 'KSS'));
    kss = squeeze(mean(epoch.misc(:, kssIdx, :), 1));
    
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
    lowIdx = find(kss==min(kss)):1:find(kss==min(kss))+200;
    highIdx = find(kss==max(kss))-200:1:find(kss==max(kss));
    
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
    
    % RPL T-TEST 분석
    [h, p, ci, stats] = ttest2(avg_delta_RPL(lowIdx), avg_delta_RPL(highIdx));
    p_delta_RPL(s) = p;
    [h, p, ci, stats] = ttest2(avg_theta_RPL(lowIdx), avg_theta_RPL(highIdx));
    p_theta_RPL(s) = p;
    [h, p, ci, stats] = ttest2(avg_alpha_RPL(lowIdx), avg_alpha_RPL(highIdx));
    p_alpha_RPL(s) = p;
    [h, p, ci, stats] = ttest2(avg_beta_RPL(lowIdx), avg_beta_RPL(highIdx));
    p_beta_RPL(s) = p;
    [h, p, ci, stats] = ttest2(avg_gamma_RPL(lowIdx), avg_gamma_RPL(highIdx));
    p_gamma_RPL(s) = p;
    
    p_analysis_RPL(s,1:5) = [p_delta_RPL(s), p_theta_RPL(s), p_alpha_RPL(s), p_beta_RPL(s), p_gamma_RPL(s)];
    p_analysis_RPL(s) = find(p_analysis_RPL(s,1:5)==min(p_analysis_RPL(s,1:5)));
    
    %% Classification
    
    feature_PSD = [avg_delta avg_theta avg_alpha avg_beta avg_gamma];
    feature_RPL = [avg_delta_RPL avg_theta_RPL avg_alpha_RPL avg_beta_RPL avg_gamma_RPL];
    
    [accuracy_PSD,AUC_PSD] = LDA_4CV(feature_PSD,highIdx,lowIdx,s);
    [accuracy_RPL,AUC_RPL] = LDA_4CV(feature_RPL,highIdx,lowIdx,s);
    
    Results_Fatigue(s,1:4) = [accuracy_PSD,AUC_PSD,accuracy_RPL,AUC_RPL]
    
    %% Regression
    %% Statistical Analysis
    
    %% Visualization
end