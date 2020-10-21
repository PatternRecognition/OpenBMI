%% Comparison of Brain Activation during Motor Imagery and Motor Execution Using EEG Signals
% Event-Related Pontential(ERP) ME & MI visualization

%% Initializing
clc; close all; clear all;

%% file
dd='C:\Users\Doyeunlee\Desktop\Analysis\rawdata\';
% Motor Imagery
filelist={'dslim_reaching_MI'};
% filelist={'eslee_reaching_MI','jmlee_reaching_MI','dslim_reaching_MI'};
% filelist={'eslee_multigrasp_MI','jmlee_multigrasp_MI','dslim_multigrasp_MI'};
% filelist={'eslee_twist_MI','jmlee_twist_MI','dslim_twist_MI'};

% Motor Execution
% filelist={'eslee_reaching_realMove','jmlee_reaching_realMove','dslim_reaching_realMove'};
% filelist={'dslim_reaching_realMove'};
% filelist={'eslee_multigrasp_realMove','jmlee_multigrasp_realMove','dslim_multigrasp_realMove'};
% filelist={'eslee_twist_realMove','jmlee_twist_realMove','dslim_twist_realMove'};


% 0~3 s
% ival=[-500 3000];

% selectChannels = [1:64];
% central channels
selectChannels = [11 12 13 43 44 45 15 16 17 18 47 48 49 20 21 22 23 51 52 53];
% selectChannels = [18];

% cnt = proc_baseline(filelist, ival);


%%
for i = 1:length(filelist)
    [cnt,mrk,mnt]=eegfile_loadMatlab([dd filelist{i}]);
    
    % [FB] ---> 9 : 4-40 Hz
    %     filterBank = {'[4 8]','[8 12]','[12 16]','[16 20]','[20 24]','[24 28]','[28 32]','[32 36]','[36 40]'};
    filterBank = {'[8 12]'};
    for filt = 1:length(filterBank)
        cnt = proc_baseline(filelist, ival);
        filtBank = {filt};
        % IIR filter (Butterworth)
        cnt = proc_filtButter(cnt, 5, filtBank{1});
        cnt_flt=proc_channelwise(cnt, 'filtfilt',b,a);
        % Preprocessing, spatial filtering - CAR
        %         cnt = proc_commonAverageReference(cnt);
        ival = [-500 3000];
        epo = cntToEpo(cnt_flt, mrk, ival);
        fv = proc_rectilfyChannels(epo);
        fv = proc_movingAverage(fv, 200, 'centered');
        fv = proc_baseline(fv, [-500 0]);
       
%         epo = cntToEpo(cnt,mrk,ival);
        % Select classes (6 classes)
        epo = proc_selectClasses(epo, {'Forward','Backward','Right','Left','Up','Down'});
        % Select channels
        %         epo = proc_selectChannels(epo, {'FC5','FC3','FC1','FC2','FC4','FC6',...
        %             'C5','C3','C1', 'Cz', 'C2', 'C4', 'C6',...
        %             'CP5','CP3','CP1','CPz','CP2','CP4','CP6'});
        %         epo = proc_selectChannels(epo, selectChannels);
        %
        %         epoForward=proc_selectClasses(epo,{'Forward'});
        %         epoBackward=proc_selectClasses(epo,{'Backward'});
        %         epoRight=proc_selectClasses(epo,{'Right'});
        %         epoLeft=proc_selectClasses(epo,{'Left'});
        %         epoUp=proc_selectClasses(epo,{'Up'});
        %         epoDown=proc_selectClasses(epo,{'Down'});
        
        % delare variables
        classes=size(epo.className,2);
        % task 별 trial 수
        trial=50;
        
        % extract the 'Rest' class
        
        for ii=1:classes
            if strcmp(epo.className{ii},'Rest')
                epoRest=proc_selectClasses(epo,{epo.className{ii}});
                epoRest.x=datasample(epoRest.x,trial,3,'Replace',false);
                epoRest.y=datasample(epoRest.y,trial,2,'Replace',false);
            else
                epo_check(ii)=proc_selectClasses(epo,{epo.className{ii}});
                epo_check(ii).x=datasample(epo_check(ii).x,trial,3,'Replace',false);
                epo_check(ii).y=datasample(epo_check(ii).y,trial,2,'Replace',false);
            end
        end
        if classes<6
            epo_check(size(epo_check,2)+1)=epoRest;
        end
        %concatenate the classes
        for ii=1:size(epo_check,2)
            if ii==1
                concatEpo=epo_check(ii);
            else
                concatEpo=proc_appendEpochs(concatEpo, epo_check(ii));
            end
        end
        
        %% CSP - feature extraction
        [csp_fv,csp_w,csp_eig]=proc_multicsp(concatEpo,3);
        proc=struct('memo','csp_w');
        
        proc.train=['[fv,csp_w]= proc_multicsp(fv, 3);' ...
            'fv=proc_variance(fv);' ...
            'fv=proc_logarithm(fv);'];
        
        proc.apply=['fv=proc_linearDerivation(fv, csp_w);','fv=proc_variance(fv);','fv=proc_logarithm(fv);'];
        
        %% LDA - Classifier
        [C_eeg, loss_eeg_std, out_4eg.out, memo] = xvalidation(concatEpo, 'RLDAshrink','proc',proc,'kfold',5);
        Result(i)=1-C_eeg;
        Result_Std(i)=loss_eeg_std;
        N=Result(i);
        All_csp_w(:,:,i)=csp_w;
    end
end


%% erp visualization
% 가로축 시간 세로축 주파수 색깔 dB
band = [0 10];
[dat] = proc_specgram(epo, band, selectChannels);
tLim = [0 300000];
yLim = [0 50]; %dB
figure (1)
showSpecgramHead(dat, mnt, band, tLim, yLim);

% saveas(fig, 'dslim_reaching.png');

%% erp graph




disp('Done');

