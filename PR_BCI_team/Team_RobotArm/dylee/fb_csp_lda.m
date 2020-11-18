% FBCSP+LDA ME & MI

%% Initializing
clc;
close all;
clear all;

%% time check
% tic;


%% file
% dd='C:\Users\Doyeunlee\Desktop\Analysis\Converted data\';
dd='H:\Dataset_I\0_ConvertedData\';
%dd='C:\Users\Doyeunlee\Desktop\AAAI_20190717\2_2DData\MI\250\';
% dd='C:\Users\Doyeunlee\Desktop\AAAI_20190717\2_2DData\realMove\250\';

% classes={'Backward','Down','Forward','Grasp','Left','Rest','Right','Twist','Up'};

% Motor Imagery
% filelist={'session1_sub1_reaching_MI','session1_sub2_reaching_MI','session1_sub3_reaching_MI','session1_sub4_reaching_MI','session1_sub5_reaching_MI','session1_sub6_reaching_MI','session1_sub7_reaching_MI','session1_sub8_reaching_MI','session1_sub9_reaching_MI','session1_sub10_reaching_MI'};
filelist = {'session1_sub7', 'session1_sub22', 'session2_sub7', 'session2_sub17', 'session2_sub18', 'session3_sub7', 'session3_sub8', 'session3_sub11', 'session3_sub19', 'session3_sub25'};
selected_channel={'F3','F1','Fz','F2','F4','FC3','FC1','FCz','FC2','FC4','C5','C3','C1', 'Cz', 'C2', 'C4','C6','CP3','CP1','CPz','CP2','CP4','P3','P1','Pz','P2','P4','POz'};

% 0~3 s
ival=[0 4000];

%%
for sub = 1:length(filelist)
%     [cnt,mrk,mnt]=eegfile_loadMatlab([dd filelist{i}]);
    [cntReach,mrkReach,mntReach]=eegfile_loadMatlab([dd '\' filelist{sub} '_reaching_' 'MI']);
    [cntGrasp,mrkGrasp,mntGrasp]=eegfile_loadMatlab([dd '\' filelist{sub} '_multigrasp_' 'MI']);
    [cntTwist,mrkTwist,mntTwist]=eegfile_loadMatlab([dd '\' filelist{sub} '_twist_' 'MI']);
    [cntReach_ME,mrkReach_ME,mntReach_ME]=eegfile_loadMatlab([dd '\' filelist{sub} '_reaching_' 'realMove']);
    [cntGrasp_ME,mrkGrasp_ME,mntGrasp_ME]=eegfile_loadMatlab([dd '\' filelist{sub} '_multigrasp_' 'realMove']);
    [cntTwist_ME,mrkTwist_ME,mntTwist_ME]=eegfile_loadMatlab([dd '\' filelist{sub} '_twist_' 'realMove']);
      
    % [FB] ---> 9 : 4-40 Hz
    filterBank = {'[4 8]','[8 12]','[12 16]','[16 20]','[20 24]','[24 28]','[28 32]','[32 36]','[36 40]'};
    
%     for filt = 1:length(filterBank)
%         filtBank = {filt};
        % IIR filter (Butterworth)
%         cnt = proc_filtButter(cnt, 3, filtBank{1});

        cntReach = proc_selectChannels(cntReach, selected_channel);
        cntGrasp = proc_selectChannels(cntGrasp, selected_channel);
        cntTwist = proc_selectChannels(cntTwist, selected_channel);
%         cntReach_ME = proc_selectChannels(cntReach_ME, selected_channel);
%         cntGrasp_ME = proc_selectChannels(cntGrasp_ME, selected_channel);
%         cntTwist_ME = proc_selectChannels(cntTwist_ME, selected_channel);
        cntReach=proc_filtButter(cntReach,3,[4 40]);
        cntGrasp=proc_filtButter(cntGrasp,3,[4 40]);
        cntTwist=proc_filtButter(cntTwist,3,[4 40]);
%         cntGrasp_ME=proc_filtButter(cntGrasp_ME,3,[4 80]);
%         cntTwist_ME=proc_filtButter(cntTwist_ME,3,[4 80]);
%         cntReach_ME=proc_filtButter(cntReach_ME,3,[4 80]);
        % Preprocessing, spatial filtering - CAR
%         cnt = proc_commonAverageReference(cnt);
%         epo = cntToEpo(cnt,mrk,ival);
        epoGrasp=cntToEpo(cntGrasp,mrkGrasp,ival);
        epoTwist=cntToEpo(cntTwist,mrkTwist,ival);
        epoReach=cntToEpo(cntReach,mrkReach,ival);
%         epoGrasp_ME=cntToEpo(cntGrasp_ME,mrkGrasp_ME,ival);
%         epoTwist_ME=cntToEpo(cntTwist_ME,mrkTwist_ME,ival);
%         epoReach_ME=cntToEpo(cntReach_ME,mrkReach_ME,ival);
        % Select classes (6 classes)
        %         epo = proc_selectClasses(epo, {'Forward','Backward','Right','Left','Up','Down'});
        
        % Select channels
%         epo=proc_selectChannels(epo,{'F3','F1','Fz','F2','F4',...
%             'FC3','FC1','FCz','FC2','FC4',...
%             'C3','C1', 'Cz', 'C2', 'C4', ...
%             'CP3','CP1','CPz','CP2','CP4',...
%             'P3','P1','Pz','P2','P4'});
        %
%         epoForward=proc_selectClasses(epo,{'Forward'});
%         epoBackward=proc_selectClasses(epo,{'Backward'});
%         epoRight=proc_selectClasses(epo,{'Right'});
%         epoLeft=proc_selectClasses(epo,{'Left'});
%         epoUp=proc_selectClasses(epo,{'Up'});
%         epoDown=proc_selectClasses(epo,{'Down'});
        epoUp=proc_selectClasses(epoReach,{'Up'});
        epoLumbrical=proc_selectClasses(epoGrasp,{'Lumbrical'});
        epoLeftTwist=proc_selectClasses(epoTwist,{'Left'});
        epoRest=proc_selectClasses(epoReach,{'Rest'});
        
        
        epo = proc_appendEpochs(epoUp,epoLumbrical);
        epo = proc_appendEpochs(epo,epoLeftTwist);
        epo = proc_appendEpochs(epo,epoRest);
        
        % delare variables
        classes=size(selected_channel,2);
%         classes=size(epo.className,2);
        % task º° trial ¼ö
        trial=50;
        
        % extract the 'Rest' class
        
%         for ii=1:classes
%             if strcmp(epo.className{ii},'Rest')
%                 epoRest=proc_selectClasses(epo,{epo.className{ii}});
%                 epoRest.x=datasample(epoRest.x,trial,3,'Replace',false);
%                 epoRest.y=datasample(epoRest.y,trial,2,'Replace',false);
%             else
%                 epo_check(ii)=proc_selectClasses(epo,{epo.className{ii}});
%                 epo_check(ii).x=datasample(epo_check(ii).x,trial,3,'Replace',false);
%                 epo_check(ii).y=datasample(epo_check(ii).y,trial,2,'Replace',false);
%             end
%         end
%         if classes<3
%             epo_check(size(epo_check,2)+1)=epoRest;
%         end
%         concatenate the classes
%         for ii=1:size(epo_check,2)
%             if ii==1
%                 concatEpo=epo_check(ii);
%             else
%                 concatEpo=proc_appendEpochs(concatEpo, epo_check(ii));
%             end
%         end
        
        %% CSP - feature extraction
        [csp_fv,csp_w,csp_eig]=proc_multicsp(epo,3);
        proc=struct('memo','csp_w');
        
        proc.train=['[fv,csp_w]= proc_multicsp(fv, 3);' ...
            'fv=proc_variance(fv);' ...
            'fv=proc_logarithm(fv);'];
        
        proc.apply=['fv=proc_linearDerivation(fv, csp_w);','fv=proc_variance(fv);','fv=proc_logarithm(fv);'];
        
        %% LDA - Classifier
        [C_eeg, loss_eeg_std, out_4eg.out, memo] = xvalidation(epo, 'RLDAshrink','proc',proc,'kfold',5);
        Result(i)=1-C_eeg;
        Result_Std(i)=loss_eeg_std;
        N=Result(i);
        All_csp_w(:,:,i)=csp_w;
        
        
        X = [filelist{i},',',filt,':',Result(i),loss_eeg_std];
        disp(X);
        
        
        filename = '2.xlsx';
        dlmwrite('20191127_reaching_MI.csv',N,'delimiter',',','-append');
        sheet = 1;
%     end
end

disp('Done');

% toc

