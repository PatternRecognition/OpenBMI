clc; close all; clear all;

dd='dir';
%% each seg contains two classes class1-1, class1-2 and class2-1, class2-2
filename={{'seg1_subj1','seg2_subj1'},{'seg1_subj2','seg2_subj2'}};

ival=[0 3000];
trial=20;

freq_band={[8 30]}

for j=1:length(filename)
    [cnt_class1,mrk_class1,mnt_class1]=eegfile_loadMatlab([dd filename{j}{1}]);
    cnt_class1=proc_filtButter(cnt_class1,2,freq_band);
    epo_class1=cntToEpo(cnt_class1,mrk_class1,ival);
    
    [cnt_class2,mrk_class2,mnt_class2]=eegfile_loadMatlab([dd filename{j}{1}]);
    cnt_class2=proc_filtButter(cnt_class2,2,freq_band);
    epo_class2=cntToEpo(cnt_class2,mrk_class2,ival);

    
    %% append epoch
    temp_epo_all=proc_appendEpochs(epo_class1, epo_class2,mrk_class1,mrk_class2);
    
    %% grouped by each class
    epoNRest=proc_selectClasses(temp_epo_all,{'class1-1','class1-2','class2-1','class2-2'});
    epoRest=proc_selectClasses(temp_epo_call,{'Rest'});
    
    %% extract the same amount of Rest as other classe
    % here the amount of trials per class is 20
    epoRest.x=datasample(epoRest.x,trial,3,'Replace',false);
    epoRest.y=datasample(epoRest.y,trial,2,'Replace',false);
    
    %% concatenate
    epo_all=proc_appendEpochs(epoNRest,epoRest);
end
