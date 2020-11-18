%% FBCSP+LDA
% doyeunlee

%% initalizing
clc; 
close all; 
clear all;

%%
% 경과시간 tic toc
tic;

%%
% file 경로
% subject: 20180219_bwyu,20180222_shchoi, GIGA_20190710_dslim, GIGA_20190708_jmlee, GIGA_20190712_wjyun 

% sub1
dd='G:\biosig4octmat-3.6.0.tar\S05_MI\';
filelist={'subject05_MI'};

% sub2
% dd='C:\Users\HANSEUL\Desktop\AAAI\shchoi\';
% filelist={'20180222_shchoi_reaching_MI'};

% sub7
% dd='C:\Users\HANSEUL\Desktop\AAAI\GIGA_20190708_jmlee\';
% filelist={'GIGA_20190715_jmlee_reaching_MI'};

% sub8
% dd='C:\Users\HANSEUL\Desktop\AAAI\GIGA_20190710_dslim\';
% filelist={'GIGA_20190710_dslim_reaching_MI'};

% sub9
% dd='C:\Users\HANSEUL\Desktop\AAAI\GIGA_20190712_wjyoon\';
% filelist={'GIGA_20190719_wjyoon_reaching_MI'};




% 1:1
% fold = 2;

% 시간 0~3 s
ival=[0 1500];

% Channel 선택
selected_channel=[1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20];

%%
for i = 1:length(filelist)
    [cnt,mrk,mnt]=eegfile_loadMatlab([dd filelist{i}]);
    
    % band pass filtering, order of 5, range of 8-15Hz
    % IIR filter (Butterworth)
    
    % 9개
    filterBank = {'[4 8]'};
    
    for filt = 1:length(filterBank)
        filtBank = {filt};
        
        cnt = proc_filtButter(cnt, 5, filtBank{1});
        
        % cnt=proc_filtButter(cnt,5,[8 15]);
        %         cnt=proc_selectChannels(cnt,selected_channel);
        % cnt to epoch
        
        % spatial filtering
        cnt = proc_commonAverageReference(cnt);
        
        epo=cntToEpo(cnt,mrk,ival);
        
        % Select classes (6 classes)
%         epo = proc_selectClasses(epo, {'Forward', 'Grasp', 'Twist'});
        
        % Select channels
        epo = proc_selectChannels(epo, selected_channel);
        
%         epoLeft=proc_selectClasses(epo,{'Left'});
%         epoRight=proc_selectClasses(epo,{'Right'});
%         epoForward=proc_selectClasses(epo,{'Forward'});
%         epoBackward=proc_selectClasses(epo,{'Backward'});
%         epoUp=proc_selectClasses(epo,{'Up'});
%         epoDown=proc_selectClasses(epo,{'Down'});
% %         
%         epoForward=proc_selectClasses(epo,{'Forward'});
%         epoGrasp=proc_selectClasses(epo,{'Grasp'});
%         epoTwist=proc_selectClasses(epo,{'Twist'});
%         
        
        % declare variables
        classes=size(epo.className,2);
        trial=50;
        
%         eachClassFold_no=trial/fold;
        
        % Extract the Rest class
        for ii =1:classes
            if strcmp(epo.className{ii},'rest')
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
        if classes<7
            epo_check(size(epo_check,2)+1)=eporest;
        end
        
        % concatenate the classes
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
        
        [C_eeg, loss_eeg_std, out_4eg.out, memo] = xvalidation(concatEpo,'RLDAshrink','proc',proc, 'kfold', 5);
        Result(i)= 1-C_eeg;
        Result_Std(i)=loss_eeg_std;
        N=Result(i);
        All_csp_w(:,:,i)=csp_w;
        
%         epoForward=proc_selectClasses(epo,{'Forward'});
%         epoGrasp=proc_selectClasses(epo,{'Grasp'});
%         epoTwist=proc_selectClasses(epo,{'Twist'});
%         
%         
        %     epoLeft=proc_selectClasses(epo,{'Left'});
        %     epoRight=proc_selectClasses(epo,{'Right'});
        %     epoRest=proc_selectClasses(epo,{'Rest'});
        
        % save data into excel file
%         filename = '1.xlsx';
% %         A = [Result(i)];
%         dlmwrite('test.csv',N,'delimiter',',','-append');
%         sheet = 2;
%         xlswrite(filename,A);
        
    end
end

toc
