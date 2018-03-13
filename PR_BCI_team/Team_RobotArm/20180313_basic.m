clc; clear all; close all;

%% get converted data

dd='E:\Wam_code_2018\Converted Data\';
filelist={'20180213_jhpark_grasp_MI','20180213_jhpark_grasp_realMove'};
%filelist={'20180207_msyun_grasp_MI','20180207_msyun_grasp_realMove','20180209_jgyoon_grasp_MI','20180209_jgyoon_grasp_realMove','20180213_jhpark_grasp_MI','20180213_jhpark_grasp_realMove'};

Result=zeros(length(filelist),1);
Result_Std=zeros(length(filelist),1);

for i=1:length(filelist)
    [cnt,mrk,mnt]=eegfile_loadMatlab([dd filelist{i}]);
    ival=[0 3000];
    
    %% band pass filtering, order of 5, range of 8-15Hz
    cnt=proc_filtButter(cnt,5, [8 15]);
    
    %% small laplacian - default : small
    % accuracy : Small - 0.635 / Large - 0.623 / Without - 0.632
    %[Dat_lap, Lap_w]=proc_laplacian(cnt,'filter_type','large');
    %[Dat_lap,Lap_w]=proc_laplacian(cnt);
    
    %% cnt to epoch
    epo=cntToEpo(cnt,mrk,ival);
    %epo=cntToEpo(Dat_lap,mrk,ival);
    
    %% The amount of Rest is always bigger than the other classes
    % need to extract rest randomly
    grasp_y=sum(epo.y(1,:));    %25
    open_y=sum(epo.y(2,:));     %25
    rest_y=sum(epo.y(3,:));     %50
    %% grouped by each class
    
    %% CSP - FEATURE EXTRACTION
    [csp_fv,csp_w,csp_eig]=proc_csp3(epo,3);
    proc=struct('memo','csp_w');
    
    proc.train= ['[fv,csp_w]=  proc_csp3(fv, 3); ' ...
        'fv= proc_variance(fv); ' ...
        'fv= proc_logarithm(fv);'];
    
    proc.apply= ['fv= proc_linearDerivation(fv, csp_w); ','fv= proc_variance(fv); ' ,'fv= proc_logarithm(fv);'];
    %% CLASSIFIER
    
    [C_eeg, loss_eeg_std, out_eeg.out, memo] = xvalidation(epo,'RLDAshrink','proc',proc, 'kfold', 5);
    Result(i)=1-C_eeg;
    Result_Std(i)=loss_eeg_std;
    
    All_csp_w(:,:,i)=csp_w;
end

Result
Result_Std

figure(1);
bar(Result);
ylim([0,1]);


