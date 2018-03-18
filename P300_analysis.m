clear all; close all; clc;
%% Initialization 
%data load
% name={'1_bykim','2_dblee','3_eskim','9_prchoi','10_smkang','12_yskim'}; %session 0
% name={'1_bykim','2_dblee','3_eskim','4_jhsuh','5_mgjung','7_mjkim','8_oykwon','9_prchoi','10_smkang','11_spseo','12_yskim','13_jyha','14_hmyoo','15_jwlee','16_wbsim','17_hwnoh','18_dysull','19_jyhan','20_jwbae','21_syjung','22_dkhan','23_hjwon','24_nkkil','25_khshim'}; % session 1
name={'1_bykim','2_dblee','3_eskim','4_jhsuh','5_mgjung','7_mjkim','9_prchoi','10_smkang','12_yskim','13_jyha','14_hmyoo','15_jwlee','16_wbsim','17_hwnoh','18_dysull','19_jyhan','20_jwbae','21_syjung','22_dkhan','23_hjwon'}; % session 2
session = {'session2'};
fs=100;
task = {'p300_off','p300_on'};

%pre-processing
channel_index = [1:32];
band = [0.5 40];
segTime = [-200 800];
baseTime =[-200 0];
selTime =[0 800];
nFeature = 10;
init_speller_length=1;
Nsequence=5;
one_seq_time= 2.75; % 1seq :2.75s
% init
speller_text= {'A', 'B', 'C', 'D', 'E', 'F', ...
    'G', 'H', 'I', 'J', 'K', 'L', ...
    'M', 'N', 'O', 'P', 'Q', 'R', ...
    'S', 'T', 'U', 'V', 'W', 'X', ...
    'Y', 'Z', '1', '2', '3', '4', ...
    '5', '6', '7', '8', '9', '_'};
spellerText_on='PATTERN_RECOGNITION_MACHINE_LEARNING';  
load('C:\Users\Oyeon Kwon\Documents\MATLAB\New_starlab\DataAnalysis\random_cell_order.mat')
rc_order;
%% Data load
for sub=1:length(name)
    for onoff=1:2
        file3 = ['E:\Users\cvpr\Desktop\StarlabDB_2nd\',name{sub},'\'];
        BMI.EEG_DIR=['E:\Users\cvpr\Desktop\StarlabDB_2nd\',name(sub),'\',session];
        BMI.EEG_DIR=cell2mat(BMI.EEG_DIR);
        file=fullfile(BMI.EEG_DIR, task{onoff});
        marker={'1','target';'2','nontarget'};
        [EEG.data, EEG.marker, EEG.info]=Load_EEG(file,{'device','brainVision';'marker', marker;'fs', fs});
        field={'x','t','fs','y_dec','y_logic','y_class','class', 'chan'};
        CNT{sub,onoff}=opt_eegStruct({EEG.data, EEG.marker, EEG.info}, field);
    end
    clear BMI EEG field file file3 marker
end
%% 
for sub =1:length(name)
    % offline data analysis - trian LDA classifier
    cnt_off= prep_selectChannels(CNT{sub,1}, {'Index',channel_index});
    cnt_off_filt=prep_filter(cnt_off, {'frequency', band});
    smt_off=prep_segmentation(cnt_off_filt, {'interval', segTime});
    smt_off=prep_baseline(smt_off, {'Time',baseTime});
    smt_off_select=prep_selectTime(smt_off, {'Time',selTime});
    fv_off=func_featureExtraction(smt_off_select,{'feature','erpmean';'nMeans',nFeature});
    [nDat, nTrials, nChans]= size(fv_off.x);
    fv_off.x= reshape(permute(fv_off.x,[1 3 2]), [nDat*nChans nTrials]);
    [clf_param] = func_train(fv_off,{'classifier','LDA'});
    clear cnt_off cnt_off_filt smt_off smt_off_select fv_off nDat nTrials nChans  
    
    % online data analyis
    cnt=prep_selectChannels(CNT{sub,2},{'Index',channel_index});
    cnt =prep_filter(cnt, {'frequency', band});
    smt=prep_segmentation(cnt, {'interval', segTime});
    smt=prep_baseline(smt, {'Time',baseTime});
    smt=prep_selectTime(smt, {'Time',selTime});
    
    dat.x= smt.x;
    dat.fs = smt.fs;
    dat.ival = smt.ival;
    dat.t = smt.t;
    
    % divide for each character
    for add=1:60:length(dat.t)
        dat_char_all.x(:,:,:,init_speller_length)=dat.x(:,[add:add+59],:);            % 1 stimulus : 0.1875s ; 27 stimulus : 5.0625s // 2 sequences : 4.5 s
        dat_char_all.t(:,:,:,init_speller_length)=dat.t(:,[add:add+59]);            % 1 stimulus : 0.1875s ; 27 stimulus : 5.0625s // 2 sequences : 4.5 s
        init_speller_length=init_speller_length+1;
    end

    % analyze for each character
    for char = 1:length(spellerText_on)
        % init
        in_nc=1;
        nc=1;
        nSeq=1;
        DAT=cell(1,length(speller_text)); % for random speller
        tm_Dat=zeros(length(speller_text),size(clf_param.cf_param.w,1));
        
        % TM1.x = ( time signal (101) * run (60) * ch (32) ) * chracter (36)
        dat_char.x = dat_char_all.x(:,:,:,char);
        dat_char.t = dat_char_all.t(:,:,:,char);
        dat_char.fs = dat.fs;
        dat_char.ival = dat.ival;
        % feature extraction
        ft_dat=func_featureExtraction(dat_char,{'feature','erpmean';'nMeans',nFeature}); %% revised 2017.07.28 (oyeon)
        [nDat, nTrials, nCh]= size(ft_dat.x);
        ft_dat.x = reshape(permute(ft_dat.x,[1 3 2]), [nDat*nCh nTrials]); % data reshape (features * ch, trials)-- 2017.07.31 (oyeon)
        
        % predict character
        for i=1:nTrials
            for i2=1:size(rc_order{nSeq},2)
                DAT{rc_order{nSeq}(in_nc,i2)}(end+1,:) = ft_dat.x(:,nc);
            end
            for i2=1:length(speller_text)
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
            t_char2(char,nSeq)= speller_text{a2};
%             t_char22(char,nc)= speller_text{a2};
            nc=nc+1;
            in_nc=in_nc+1;
            if in_nc>size(rc_order{nSeq},1)
                in_nc=1;
                nSeq=nSeq+1;
            end
        end
        clear DAT tm_Dat Y a b a1 a2
    end
    clear add BB CF_PARAM char clf_param cnt dat dat_char dat_char_all tm_Dat Y DAT 
    clear ft_dat fv i i2 in_nc ival nc nCh nDat nSeq nTrials smt temp3 nChans
    clear BMI EEG field file file3 marker 
    
    for seq=1:Nsequence
        for nchar=1:length(spellerText_on)
            seq_acc(nchar, seq)=strcmp(spellerText_on(nchar),t_char2(nchar,seq));
        end
    end
    
    sub_acc=sum(seq_acc)/length(spellerText_on);
    allsub_acc(:,sub)=sub_acc;
    
    for i=1:Nsequence
        t2(i)=one_seq_time*i;
    end
    
    speller_number = length(spellerText_on);
    no_b = sub_acc.*log2(sub_acc) + (1-sub_acc).*log2((1-sub_acc)/(speller_number-1)) + log2(speller_number);
    ind = find(isnan(no_b) == 1);
    no_b(ind) = log2(speller_number);
    no_itr = [no_b ./ (t2./60)];
    allsub_ITR(:,sub)=no_itr;
    
    init_speller_length=1;
    clear seq_acc i ind n nchar sub_acc no_b no_itr seq t2 t_char2 t_char22 speller_length  cnt_off cnt_off_filt smt_off smt_off_select
end
%% Acc
allsub_acc_mean=mean(allsub_acc,2);
allsub_itr_mean=mean(allsub_ITR,2);
 
 
 
 

