clc; clear all; close all;

dd='dir';
filelist={'file1'};

classes=3;
fold=5;

ival=[0 3000];

selected_class=[8 9 10 11 13 14 15 18 19 20 21 43 44 47 48 49 50 52 53 54];

for i = 1:length(filelist)
    [cnt,mrk,mnt]=eegfile_loadMatlab([dd filelist{1}]);
    %% band pass filtering, order of 5, range of 8-15Hz
    cnt=proc_filtButter(cnt,2, [8 15]);
    cnt=proc_selectChannels(cnt,selected_class);
   %% cnt to epoch
    epo=cntToEpo(cnt,mrk,ival);
    %% Extract the Rest class
    epoLeft=proc_selectClasses(epo,{'class1'});
    epoRight=proc_selectClasses(epo,{'class2'});
    epoRest=proc_selectClasses(epo,{'class3'});
    
    %% extract the number of trial
    trial=size(epoLeft.x,3);
    
    %% randomize the sequence of trials
    epoLeft.x=datasample(epoLeft.x,trial,3,'Replace',false);
    epoRight.x=datasample(epoRight.x,trial,3,'Replace',false);
    %% extract the same amount of Rest as other classes
    epoRest.x=datasample(epoRest.x,trial,3,'Replace',false);
    epoRest.y=datasample(epoRest.y,trial,2,'Replace',false);
    
    %% fold 0.068534sec
    fold_each_class_no=size(epoLeft.x,3)/fold;
    for ii=1:trial/fold_each_class_no
        epo_temp_1=epoLeft;
        epo_temp_1.x=epoLeft.x(:,:,fold_each_class_no*(ii-1)+1:fold_each_class_no*ii);
        epo_temp_1.y=epoLeft.y(:,fold_each_class_no*(ii-1)+1:fold_each_class_no*ii);
        
        epo_temp_2=epoRight;
        epo_temp_2.x=epoRight.x(:,:,fold_each_class_no*(ii-1)+1:fold_each_class_no*ii);
        epo_temp_2.y=epoRight.y(:,fold_each_class_no*(ii-1)+1:fold_each_class_no*ii);
        
        epo_temp_3=epoRest;
        epo_temp_3.x=epoRest.x(:,:,fold_each_class_no*(ii-1)+1:fold_each_class_no*ii);
        epo_temp_3.y=epoRest.y(:,fold_each_class_no*(ii-1)+1:fold_each_class_no*ii);
        
        %% epo_temp will have fold X classes
        %% possible algorithm error
        % each fold has follow sequence
        % 5 consecutive left class, 5 consecutive right class
        % 5 consecutive Rest class
        epo_temp(ii,1)=epo_temp_1;
        epo_temp(ii,2)=epo_temp_2;
        epo_temp(ii,3)=epo_temp_3;
        
    end
    %% append data by row
    for ii=1:length(epo_temp)
        epo_temp_all(ii)=proc_appendEpochs(epo_temp(ii,1),epo_temp(ii,2));
        epo_all(ii)=proc_appendEpochs(epo_temp_all(ii),epo_temp(ii,3));
    end
    
    %% feature selection
    %% classifier
end


