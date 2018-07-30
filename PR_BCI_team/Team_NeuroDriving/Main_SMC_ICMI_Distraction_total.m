clear all; close all; clc;

load('BTB.mat')

subjectList = {'DJLEE_20180421_distraction'};
file = 'F:\Matlab\Data\Pilot\Distraction\MATLAB';
savefile = 'F:\Matlab\Plot\Distraction';

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
    
    Result_1 = Distraction_1(cnt,mrk,s);
    Result_10 = Distraction_10(cnt,mrk,s);
    Result_20 = Distraction_20(cnt,mrk,s);    
    Result_30 = Distraction_30(cnt,mrk,s);    
    Result_30_2 = Distraction_30_2(cnt,mrk,s);
    
end