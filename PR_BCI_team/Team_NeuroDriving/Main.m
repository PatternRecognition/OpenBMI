clear all; close all; clc;

load('BTB.mat')
global BTB
BTB.MatDir='D:\ICMI\Data\Fatigue\';
subjectList = {'HSRYU_20180406_fatigue','JHHWANG_20180407_fatigue','GSPARK_20180413_fatigue','DNJO_20180414_fatigue','KTKIM_20180420_fatigue','DJLEE_20180421_fatigue'};
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
    
    epoch = segmentationFatigue_R(cnt, mrk, interKSS);
    kssIdx = find(strcmp(epoch.mClab, 'KSS'));
    kss = squeeze(mean(epoch.misc(:, kssIdx, :), 1));
    
    %% Regional Channel Selection & Feature Extraction
    
    % PSD Feature Extraction
    [cca_delta,avg_delta] = spectrumAnalysisSleep_R(epoch.x, [0.5 4], cnt.fs);
    [cca_theta,avg_theta] = spectrumAnalysisSleep_R(epoch.x, [4 8], cnt.fs);
    [cca_alpha,avg_alpha] = spectrumAnalysisSleep_R(epoch.x, [8 13], cnt.fs);
    [cca_beta,avg_beta] = spectrumAnalysisSleep_R(epoch.x, [13 30], cnt.fs);
    [cca_gamma,avg_gamma] = spectrumAnalysisSleep_R(epoch.x, [30 50], cnt.fs);
    avg_delta = avg_delta'; avg_theta=avg_theta'; avg_alpha=avg_alpha'; avg_beta = avg_beta'; avg_gamma = avg_gamma';
    
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
    
    if s==1
        s1.kss = kss;
        s1.avg_delta = avg_delta;
        s1.avg_theta = avg_theta;
        s1.avg_alpha = avg_alpha;
        s1.avg_beta = avg_beta;
        s1.avg_gamma = avg_gamma;
        s1.cca_delta = cca_delta;
        s1.cca_theta = cca_theta;
        s1.cca_alpha = cca_alpha;
        s1.cca_beta = cca_beta;
        s1.cca_gamma = cca_gamma;
    else if s==2
            s2.kss = kss;
            s2.avg_delta = avg_delta;
            s2.avg_theta = avg_theta;
            s2.avg_alpha = avg_alpha;
            s2.avg_beta = avg_beta;
            s2.avg_gamma = avg_gamma;
            s2.cca_delta = cca_delta;
            s2.cca_theta = cca_theta;
            s2.cca_alpha = cca_alpha;
            s2.cca_beta = cca_beta;
            s2.cca_gamma = cca_gamma;
        else if s==3
                s3.kss = kss;
                s3.avg_delta = avg_delta;
                s3.avg_theta = avg_theta;
                s3.avg_alpha = avg_alpha;
                s3.avg_beta = avg_beta;
                s3.avg_gamma = avg_gamma;
                s3.cca_delta = cca_delta;
                s3.cca_theta = cca_theta;
                s3.cca_alpha = cca_alpha;
                s3.cca_beta = cca_beta;
                s3.cca_gamma = cca_gamma;
            else if s==4
                    s4.kss = kss;
                    s4.avg_delta = avg_delta;
                    s4.avg_theta = avg_theta;
                    s4.avg_alpha = avg_alpha;
                    s4.avg_beta = avg_beta;
                    s4.avg_gamma = avg_gamma;
                    s4.cca_delta = cca_delta;
                    s4.cca_theta = cca_theta;
                    s4.cca_alpha = cca_alpha;
                    s4.cca_beta = cca_beta;
                    s4.cca_gamma = cca_gamma;
                else if s==5
                        s5.kss = kss;
                        s5.avg_delta = avg_delta;
                        s5.avg_theta = avg_theta;
                        s5.avg_alpha = avg_alpha;
                        s5.avg_beta = avg_beta;
                        s5.avg_gamma = avg_gamma;
                        s5.cca_delta = cca_delta;
                        s5.cca_theta = cca_theta;
                        s5.cca_alpha = cca_alpha;
                        s5.cca_beta = cca_beta;
                        s5.cca_gamma = cca_gamma;
                    else if s==6
                            s6.kss = kss;
                            s6.avg_delta = avg_delta;
                            s6.avg_theta = avg_theta;
                            s6.avg_alpha = avg_alpha;
                            s6.avg_beta = avg_beta;
                            s6.avg_gamma = avg_gamma;
                            s6.cca_delta = cca_delta;
                            s6.cca_theta = cca_theta;
                            s6.cca_alpha = cca_alpha;
                            s6.cca_beta = cca_beta;
                            s6.cca_gamma = cca_gamma;
                        end
                    end
                end
            end
        end
    end
    
    %% T-Test
    
    % 일단 간단하게 최소 KSS +120 과 최대 KSS -120 데이터 가지고서 분류
    lowIdx = find(kss==min(kss)):1:find(kss==min(kss))+200;
    lowIdx = lowIdx';
    highIdx = find(kss==max(kss))-200:1:find(kss==max(kss));
    highIdx = highIdx';
    
    %     최소+1, 최대-1
    %         lowIdx = min(kss)<kss & min(kss)+1>kss;
    %         highIdx = max(kss)>kss & max(kss)-1<kss;
    
    %     KSS 5미만, 5이상
    %     lowIdx = kss<5;
    %     highIdx = kss>=5;
    
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
    p_analysis_min(s) = find(p_analysis(s,1:5)==min(p_analysis(s,1:5)));
    
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
    p_analysis_RPL_min(s) = find(p_analysis_RPL(s,1:5)==min(p_analysis_RPL(s,1:5)));
    
    %% Classification
    
    feature_PSD = [avg_delta avg_theta avg_alpha avg_beta avg_gamma];
    feature_RPL = [avg_delta_RPL avg_theta_RPL avg_alpha_RPL avg_beta_RPL avg_gamma_RPL];
    feature_min_PSD = [feature_PSD(:,p_analysis_min(s))];
    feature_min_RPL = [feature_RPL(:,p_analysis_RPL_min(s))];
    
    feature_delta = avg_delta; feature_theta = avg_theta; feature_alpha = avg_alpha; feature_beta = avg_beta; feature_gamma = avg_gamma;
    
    [accuracy_PSD,AUC_PSD] = LDA_4CV(feature_PSD,highIdx,lowIdx,s);
    [accuracy_RPL,AUC_RPL] = LDA_4CV(feature_RPL,highIdx,lowIdx,s);
    [accuracy_min_PSD,AUC_min_PSD] = LDA_4CV(feature_min_PSD,highIdx,lowIdx,s);
    [accuracy_min_RPL,AUC_min_RPL] = LDA_4CV(feature_min_RPL,highIdx,lowIdx,s);
    
    [accuracy_delta,AUC_delta] = LDA_4CV(feature_delta,highIdx,lowIdx,s);
    [accuracy_theta,AUC_theta] = LDA_4CV(feature_theta,highIdx,lowIdx,s);
    [accuracy_alpha,AUC_alpha] = LDA_4CV(feature_alpha,highIdx,lowIdx,s);
    [accuracy_beta,AUC_beta] = LDA_4CV(feature_beta,highIdx,lowIdx,s);
    [accuracy_gamma,AUC_gamma] = LDA_4CV(feature_gamma,highIdx,lowIdx,s);
    
    Results_Fatigue_Classification(s,1:8) = [accuracy_PSD,AUC_PSD,accuracy_RPL,AUC_RPL,accuracy_min_PSD,AUC_min_PSD,accuracy_min_RPL,AUC_min_RPL]
    Results_Fatigue_Classification_2(s,1:10) = [accuracy_delta,AUC_delta,accuracy_theta,AUC_theta,accuracy_alpha,AUC_alpha,accuracy_beta,AUC_beta,accuracy_gamma,AUC_gamma]
    
    %% Statistical Analysis
    
    feature1 = [kss feature_PSD];
    feature2 = [kss feature_RPL];
    feature3 = [kss feature_min_PSD];
    feature4 = [kss feature_min_RPL];
    
    feature5 = [kss feature_delta];
    feature6 = [kss feature_theta];
    feature7 = [kss feature_alpha];
    feature8 = [kss feature_beta];
    feature9 = [kss feature_gamma];
    
    [A B r U1 V1] = canoncorr(feature1(:,2:end), kss); temp = corrcoef(U1, V1); cca_eeg_kss_PSD = temp(2);
    [A B r U2 V2] = canoncorr(feature2(:,2:end), kss); temp = corrcoef(U2, V2); cca_eeg_kss_RPL = temp(2);
    [A B r U3 V3] = canoncorr(feature3(:,2:end), kss); temp = corrcoef(U3, V3); cca_eeg_kss_min_PSD = temp(2);
    [A B r U4 V4] = canoncorr(feature4(:,2:end), kss); temp = corrcoef(U4, V4); cca_eeg_kss_min_RPL = temp(2);
    
    [A B r U5 V5] = canoncorr(feature5(:,2:end), kss); temp = corrcoef(U5, V5); cca_eeg_kss_delta = temp(2);
    [A B r U6 V6] = canoncorr(feature6(:,2:end), kss); temp = corrcoef(U6, V6); cca_eeg_kss_theta = temp(2);
    [A B r U7 V7] = canoncorr(feature7(:,2:end), kss); temp = corrcoef(U7, V7); cca_eeg_kss_alpha = temp(2);
    [A B r U8 V8] = canoncorr(feature8(:,2:end), kss); temp = corrcoef(U8, V8); cca_eeg_kss_beta = temp(2);
    [A B r U9 V9] = canoncorr(feature9(:,2:end), kss); temp = corrcoef(U9, V9); cca_eeg_kss_gamma = temp(2);
    
    Results_Fatigue_Correlation(s,1:4) = [cca_eeg_kss_PSD cca_eeg_kss_RPL cca_eeg_kss_min_PSD cca_eeg_kss_min_RPL]
    Results_Fatigue_Correlation_2(s,1:5) = [cca_eeg_kss_delta cca_eeg_kss_theta cca_eeg_kss_alpha cca_eeg_kss_beta cca_eeg_kss_gamma]
    
    %     h = figure ('Color', [1 1 1]);
    %     subplot(1,4,1)
    %     s1 = plot(U1, V1,'b+');
    %     set(s1,'MarkerSize', 8,'LineWidth', 2);
    %     %%% regression line
    %     hold on
    %     l = lsline ;
    %     set(l,'LineWidth', 2,'color','m')
    %     xlim([-10 10])
    %     ylim([-5 5])
    %
    %     subplot(1,4,2)
    %     s1 = plot(U2, V2,'b+');
    %     set(s1,'MarkerSize', 8,'LineWidth', 2);
    %     %%% regression line
    %     hold on
    %     l = lsline ;
    %     set(l,'LineWidth', 2,'color','m')
    %     xlim([-10 10])
    %     ylim([-5 5])
    %
    %     subplot(1,4,3)
    %     s1 = plot(U3, V3,'b+');
    %     set(s1,'MarkerSize', 8,'LineWidth', 2);
    %     %%% regression line
    %     hold on
    %     l = lsline ;
    %     set(l,'LineWidth', 2,'color','m')
    %     xlim([-10 10])
    %     ylim([-5 5])
    %
    %     subplot(1,4,4)
    %     s1 = plot(U4, V4,'b+');
    %     set(s1,'MarkerSize', 8,'LineWidth', 2);
    %     %%% regression line
    %     hold on
    %     l = lsline ;
    %     set(l,'LineWidth', 2,'color','m')
    %     xlim([-10 10])
    %     ylim([-5 5])
    
    %% Visualization
    
    %     grd= sprintf(['_,_,_,FP1,_,_,_,FP2,_,_,_\n', ...
    %         '_,_,_,AF3,_,_,_,AF4,_,_,_\n'...
    %         '_,_,F5,F3,F1,Fz,F2,F4,F6,_,_\n', ...
    %         'FT9,FT7,FC5,FC3,FC1,_,FC2,FC4,FC6,FT8,FT10\n', ...
    %         '_,T7,C5,C3,C1,Cz,C2,C4,C6,T8,_\n', ...
    %         'TP9,TP7,CP5,CP3,CP1,CPz,CP2,CP4,CP6,TP8,TP10\n', ...
    %         '_,P7,P5,P3,P1,Pz,P2,P4,P6,P8,_\n'...
    %         '_,_,PO7,PO3,POz,PO4,PO8,_,_,_,_\n'...
    %         'legend,_,_,PO9,O1,Oz,O2,PO10,_,_,scale\n']);
    %
    %     mnt= mnt_setGrid(mnt, grd);
    %     mnt.x=mnt.x(1:64,1);
    %     mnt.y=mnt.y(1:64,1);
    %     mnt.pos_3d=mnt.pos_3d(:,1:64);
    %     mnt.clab=mnt.clab(1,1:64);
    %
    %     feature_cca_PSD = [cca_delta cca_theta cca_alpha cca_beta cca_gamma];
    %     feature_cca_PSD = feature_cca_PSD(:,p_analysis_min(s)*64-63:p_analysis_min(s)*64);
    %     feature_cca_RPL = [cca_delta_RPL cca_theta_RPL cca_alpha_RPL cca_beta_RPL cca_gamma_RPL];
    %     feature_cca_RPL = feature_cca_RPL(:,p_analysis_RPL_min(s)*64-63:p_analysis_RPL_min(s)*64);
    %
    %     Low = mean(feature_cca_PSD(min(kss)<kss & min(kss)+1>kss,:));
    %     High = mean(feature_cca_PSD(max(kss)>kss & max(kss)-1<kss,:));
    %     figure
    %     subplot(1,2,1)
    %     [H, Ctour] = plot_scalp(mnt, Low,'CLim','0tomax');
    %     title('PSD Low')
    %     subplot(1,2,2)
    %     [H, Ctour] = plot_scalp(mnt, High,'CLim','0tomax');
    %     title('PSD High')
    %
    %     Low = mean(feature_cca_RPL(min(kss)<kss & min(kss)+1>kss,:));
    %     High = mean(feature_cca_RPL(max(kss)>kss & max(kss)-1<kss,:));
    %     figure
    %     subplot(1,2,1)
    %     [H, Ctour] = plot_scalp(mnt, Low,'CLim','sym');
    %     title('RPL Low')
    %     subplot(1,2,2)
    %     [H, Ctour] = plot_scalp(mnt, High,'CLim','sym');
    %     title('RPL High')
    %
    %
    
end
