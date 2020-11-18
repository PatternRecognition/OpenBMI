%% Comparison of Brain Activation during Motor Imagery and Motor Execution Using EEG Signals
% topoplot ME & MI

%% Initializing
clc;
close all;
clear all;

%% time check
tic;

%% file directory
dd='G:\biosig4octmat-3.6.0.tar\run1_MIME\';
filelist={'subject01_ME'};

fold=5;

ival = [0 2001];
% ival= [0 1000; 2000 3000; 4000 5000;];
% selected_channel= [1 2 4 5 6 8 10 11 12 14 16 18 19 21 23 25 26 27 28 30 31 32];
% selected_channel= [1:32];



%%
for i = 1:length(filelist)
    [cnt,mrk,mnt] = eegfile_loadMatlab([dd filelist{i}]);
    
    %% band pass filtering, order of 5, range of 7-30Hz
    cnt=proc_filtButter(cnt,2, [0.5 40]);
    %     cnt=proc_selectChannels(cnt, selected_channel);
    
    %% cnt to epoch
    epo=cntToEpo(cnt,mrk,ival);
    
    %% declare variables
    classes=size(epo.className,2);
    trial=size(epo.x,3)/2/(classes-1);
    
    eachClassFold_no=trial/fold;
    
    %% CSP - FEATURE EXTRACTION
    [csp_fv,csp_w,csp_eig]=proc_multicsp(epo, 3);
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
    
    %% Visualization
    
    %csp patterns
    figure('Name', 'CSP Patterns');
    plotCSPatterns(csp_fv, mnt, csp_w, csp_fv.y)
    
    
    %     class topographies
    %     figure('Name', 'Class Topographies');
    %     plotClassTopographies(epo, mnt, ival);
    
    
    %     epoLeft=proc_selectClasses(epo,{'Left'});
    %     epoRight=proc_selectClasses(epo,{'Right'});
    %     epoRest=proc_selectClasses(epo,{'Rest'});
    
%     saveas ('CSP Patterns','dslim_reaching_MI.png');
    
end

%% Visualization

%%class topographies
% figure('Name', 'Class Topographies');
% plotClassTopographies(epo_all, mnt, ival);

%%csp patterns
% figure('Name', 'CSP Patterns');
% plotCSPatterns(csp_fv, mnt, csp_w, csp_fv.y);

%[hp, hl]= showEEG(epo_all, ival, mrk)


