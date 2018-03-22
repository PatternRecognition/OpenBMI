clc; clear all; close all;

%% get converted data

dd='eeg_data_dir';
filelist={'subj1','subj2','subj3'};

Result=zeros(length(filelist),1);
Result_Std=zeros(length(filelist),1);

for i=1:length(filelist)
    [cnt,mrk,mnt]=eegfile_loadMatlab([dd filelist{i}]);
    ival=[0 3000];
    
    %% band pass filtering, order of 5, range of 8-15Hz
    cnt=proc_filtButter(cnt,5, [8 15]);
    
    %% cnt to epoch
    epo=cntToEpo(cnt,mrk,ival);
    
    %% CSP - FEATURE EXTRACTION
    [csp_fv,csp_w,csp_eig]=proc_multicsp(epo,3);
    proc=struct('memo','csp_w');
    
    proc.train= ['[fv,csp_w]=  proc_multicsp(fv, 3); ' ...
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


