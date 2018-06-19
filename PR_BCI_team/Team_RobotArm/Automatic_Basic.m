%% ÀÚµ¿È­
tic;
clc; close all; clear all;

dd='dir';
filelist={'subj1'};


fold=5;

ival=[0 3000];

selected_class=[8 9 10 11 13 14 15 18 19 20 21 43 44 47 48 49 50 52 53 54];

for i = 1:length(filelist)
    [cnt,mrk,mnt]=eegfile_loadMatlab([dd filelist{i}]);
    
    %% band pass filtering, order of 5, range of 8-15Hz
    cnt=proc_filtButter(cnt,2, [8 15]);
    cnt=proc_selectChannels(cnt,selected_class);
   %% cnt to epoch
    epo=cntToEpo(cnt,mrk,ival);
    %% declare variables
    classes=size(epo.className,2);
    trial=size(epo.x,3)/2/(classes-1);
    
    eachClassFold_no=trial/fold;
    
    %% Extract the Rest class
    for ii =1:classes
        if strcmp(epo.className{ii},'Rest')
            epoRest=proc_selectClasses(epo,{epo.className{ii}});
            epoRest.x=datasample(epoRest.x,trial,3,'Replace',false);
            epoRest.y=datasample(epoRest.y,trial,2,'Replace',false);
        else
            epo_check(ii)=proc_selectClasses(epo,{epo.className{ii}});
            % random sampling
            epo_check(ii).x=datasample(epo_check(ii).x,trial,3,'Replace',false);
            epo_check(ii).y=datasample(epo_check(ii).y,trial,2,'Replace',false);
        end
    end
    %% class should be lower then 7
    if classes<7
        epo_check(size(epo_check,2)+1)=epoRest;
    end
    
    %% concatenate the classes
    for ii=1:size(epo_check,2)
        if ii==1
            concatEpo=epo_check(ii);
        else
            concatEpo=proc_appendEpochs(concatEpo,epo_check(ii));
        end
    end
    %% CSP - FEATURE EXTRACTION
    [csp_fv,csp_w,csp_eig]=proc_multicsp(concatEpo,3);
    proc=struct('memo','csp_w');
    
    proc.train= ['[fv,csp_w]=  proc_multicsp(fv, 3); ' ...
        'fv= proc_variance(fv); ' ...
        'fv= proc_logarithm(fv);'];
    
    proc.apply= ['fv= proc_linearDerivation(fv, csp_w); ','fv= proc_variance(fv); ' ,'fv= proc_logarithm(fv);'];
    %% CLASSIFIER
    
    [C_eeg, loss_eeg_std, out_eeg.out, memo] = xvalidation(concatEpo,'RLDAshrink','proc',proc, 'kfold', 5);
    Result(i)=1-C_eeg;
    Result_Std(i)=loss_eeg_std;
    
    All_csp_w(:,:,i)=csp_w;
    
        
%     epoLeft=proc_selectClasses(epo,{'Left'});
%     epoRight=proc_selectClasses(epo,{'Right'});
%     epoRest=proc_selectClasses(epo,{'Rest'});

end
toc
