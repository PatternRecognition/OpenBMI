%% LOAD THE OpenBMI TOOLBOX
clear all; close all; clc;
%name={'bykim','dblee','eskim','prchoi','smkang','mhlee'}; % session 1 itr, acc 맷파일 순서
%%
name={'yskim','oykwon','mjkim'}; % session 1
%name={'bykim','smkang', 'eskim','prchoi','yskim'}; % session 2
session = {'session1'};
task = 'p300_off';
fs=100;

%for save
filename1= ['p300_cnt_s1_off'];
%% Load and 'mat' save
for sub=1:length(name)
    % session 1 : 8 sub
    % session 2 : 4 sub
    % 제외: ejlee, sbsim, yelee   
    
    file3 = ['E:\Users\cvpr\Desktop\StarlabDB_2nd\',name{sub},'\'];
    BMI.EEG_DIR=['E:\Users\cvpr\Desktop\StarlabDB_2nd\',name(sub),'\',session];
    BMI.EEG_DIR=cell2mat(BMI.EEG_DIR);
    file=fullfile(BMI.EEG_DIR, task);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if isequal(task,'p300_on')  % ONLINE
        marker={'1',1;'2',2;'3',3;'4',4;'5',5;'6',6;'7',7;'8',8;'9',9;'10',10;'11',11;'12',12};
        spellerText_on='PATTERN_RECOGNITION_MACHINE_LEARNING';
%         if isequal(name{sub}, 'mhlee') % for special case .. even online, but trigger 21, 22
%             marker={'21','target';'22','nontarget'};      % minho
%         end
    else                        % OFFLINE
        marker={'1','target';'2','nontarget'};
%         if isequal(name{sub}, 'mhlee')  % for special case .. even offline but trigger 1~12
%             marker={'1',1;'2',2;'3',3;'4',4;'5',5;'6',6;'7',7;'8',8;'9',9;'10',10;'11',11;'12',12};
%             spellerText_on='PATTERN_RECOGNITION_MACHINE_LEARNING';
%         end
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    [EEG.data, EEG.marker, EEG.info]=Load_EEG(file,{'device','brainVision';'marker', marker;'fs', fs});
    field={'x','t','fs','y_dec','y_logic','y_class','class', 'chan'};
    CNT=opt_eegStruct({EEG.data, EEG.marker, EEG.info}, field);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if isequal(task,'p300_on')        % for online, trigger change to 1,2
        CNT=TriggerToTar_nTar(CNT,spellerText_on);
    end
    
%     if isequal(name{sub},'mhlee')   % when mhlee... has different trigger
%         if isequal(task,'p300_off') % when off, 1~12 --> 1,2
%             CNT=TriggerToTar_nTar(CNT,spellerText_on);
%         end
%         if isequal(task,'p300_on') % when on, 21,22 --> 1,2
%             % 21, 22 --> 1, 2
%         end
%     end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    CNTT{sub,1} = CNT;
    %         filename1= ['p300_cnt_s1_off'];
%     save([file3, filename1], 'CNT');
end
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% P300 -make classifier
for sub=1:length(name)
    %     name={'bykim','dblee','ejlee','eskim','mhlee','prchoi','sbsim','smkang','yelee','yskim'};
%     name={'bykim','eskim','prchoi','smkang','oykwon'};
%     file3 = ['E:\Users\cvpr\Desktop\StarlabDB_2nd\',name{sub},'\'];
%     filename4 = ['p300_cnt_s1_off.mat'];
    % load
%     cnt = load([file3, filename4]);
    
    % init
    segTime = [-200 800];
    baseTime =[-200 0];
    selTime =[0 800];
    nFeature = 10; 
    
    % make classifier
    cnt= prep_selectChannels(CNTT{sub,1}, {'Index',[1:32]});
    cnt=prep_filter(cnt, {'frequency', [0.5 40]});
    smt=prep_segmentation(cnt, {'interval', segTime});
    smt=prep_baseline(smt, {'Time',baseTime});
    smt=prep_selectTime(smt, {'Time',selTime});
    fv=func_featureExtraction(smt,{'feature','erpmean';'nMeans',nFeature});
    [nDat, nTrials, nChans]= size(fv.x);
    fv.x= reshape(permute(fv.x,[1 3 2]), [nDat*nChans nTrials]);
    [clf_param] = func_train(fv,{'classifier','LDA'});
    classifier{sub,1} = clf_param;
end

% spellerText_off='NEURAL_NETWORKS_AND_DEEP_LEARNING';
% spellerText_on='PATTERN_RECOGNITION_MACHINE_LEARNING';
% 
% NumberOfSeq=5;
% stimulusTime=0.135; IntervalTime=0.05;
% TimeIval=[-200 800];
% baselineTime=[-200 0];
% selectedTime=[0 800];
% numFeature=10;
% selectedFreq=[0.5 40];
%% P300 online - Prediction
for sub=1:length(name)
    name={'bykim','dblee','eskim','prchoi','smkang'}; % session 1
    %     name={'bykim','dblee','ejlee','eskim','mhlee','prchoi','sbsim','smkang','yelee','yskim'};
%     name={'bykim','eskim','prchoi','smkang','oykwon'};
%     file3 = ['E:\Users\cvpr\Desktop\StarlabDB_2nd\',name{sub},'\'];
%     filename4 = ['p300_cnt_s1_on.mat'];
%     cnt = load([file3, filename4]);
    
    
    % init
    temp3= {'A', 'B', 'C', 'D', 'E', 'F', ...
    'G', 'H', 'I', 'J', 'K', 'L', ...
    'M', 'N', 'O', 'P', 'Q', 'R', ...
    'S', 'T', 'U', 'V', 'W', 'X', ...
    'Y', 'Z', '1', '2', '3', '4', ...
    '5', '6', '7', '8', '9', '_'};

    ival = [-200 800];
    baseTime = [-200 0];
    selTime = [0 800];
    fs = cnt.CNT.fs;
    BB=1;
    nFeatures=10;
    spellerText_on='PATTERN_RECOGNITION_MACHINE_LEARNING';
    load('C:\Users\Oyeon Kwon\Documents\MATLAB\New_starlab\DataAnalysis\random_cell_order.mat')
    rc_order;
    
    % segmentation
    cnt =prep_filter(cnt.CNT, {'frequency', [0.5 40]});
    smt=prep_segmentation(cnt, {'interval', ival});
    smt=prep_baseline(smt, {'Time',baseTime});
    smt=prep_selectTime(smt, {'Time',selTime});
    smt=prep_selectChannels(smt,{'Index',[1:32]});
    dat.x= smt.x;
    dat.fs = smt.fs;
    dat.ival = smt.ival;
    dat.t = smt.t;

    % divide for each character
    for add=1:60:length(dat.t)
        dat_char_all.x(:,:,:,BB)=dat.x(:,[add:add+59],:);            % 1 stimulus : 0.1875s ; 27 stimulus : 5.0625s // 2 sequences : 4.5 s
        dat_char_all.t(:,:,:,BB)=dat.t(:,[add:add+59]);            % 1 stimulus : 0.1875s ; 27 stimulus : 5.0625s // 2 sequences : 4.5 s
        BB=BB+1;
    end
    
    % analyze for each character
    for char = 1:length(spellerText_on)
        % init
        in_nc=1;
        nc=1;
        nSeq=1;
        DAT=cell(1,36); % for random speller
        tm_Dat=zeros(36,size(clf_param.cf_param.w,1));
    
        % TM1.x = ( time signal (101) * run (60) * ch (32) ) * chracter (36)
        dat_char.x = dat_char_all.x(:,:,:,char);
        dat_char.t = dat_char_all.t(:,:,:,char);
        dat_char.fs = dat.fs;
        dat_char.ival = dat.ival;
        % feature extraction
        ft_dat=func_featureExtraction(dat_char,{'feature','erpmean';'nMeans',nFeatures}); %% revised 2017.07.28 (oyeon)
        [nDat, nTrials, nCh]= size(ft_dat.x);
        ft_dat.x = reshape(permute(ft_dat.x,[1 3 2]), [nDat*nCh nTrials]); % data reshape (features * ch, trials)-- 2017.07.31 (oyeon)
    
    % predict character
        for i=1:60
            for i2=1:6
                DAT{rc_order{nSeq}(in_nc,i2)}(end+1,:) = ft_dat.x(:,nc);
            end
            for i2=1:36
                if size(DAT{i2},1)==1
                    tm_Dat(i2,:)=DAT{i2};
                else
                    tm_Dat(i2,:)=mean(DAT{i2});
                end
            end            
            %             CF_PARAM=classifier{sub,1};
            CF_PARAM = clf_param;
            [Y]=func_predict(tm_Dat', CF_PARAM);
            [a1 a2]=min(Y);
            t_char2(char,nSeq)= temp3{a2};
            t_char22(char,nc)= temp3{a2};            
            
            nc=nc+1;
            in_nc=in_nc+1;            
            if in_nc>12
                in_nc=1;
                nSeq=nSeq+1;
            end
        end
        clear DAT tm_Dat Y a b a1 a2
    end
 clear add baseTime BB CF_PARAM char clf_param cnt dat dat_char dat_char_all
 clear ft_dat fv i i2 in_nc ival nc nCh nDat nFeature nFeatures nSeq nTrials segTime selTime smt temp3 nChans
end
%% Acc

for seq=1:5
    for nchar=1:length(spellerText_on) 
        acc2(nchar, seq)=strcmp(spellerText_on(nchar),t_char2(nchar,seq));
    end
end

no_=sum(acc2)/length(spellerText_on);
% ACC_N(sub,:)=no_;
ACC_N{iii,1}=no_;

t_1= 2.75; % 1seq :2.75s
for i=1:5
    t2(i)=t_1*i;
end

n = 36;
no_b = no_.*log2(no_) + (1-no_).*log2((1-no_)/(n-1)) + log2(n);
ind = find(isnan(no_b) == 1);
no_b(ind) = log2(n);
no_itr = [no_b ./ (t2./60)];
% ITR_N(sub,:)=no_itr;
ITR_N{iii,1}=no_itr;


mean(ACC_N);
mean(ITR_N);

ACC_N1{6,1} = ACC_N;
ITR_N1{6,1} = ITR_N;

