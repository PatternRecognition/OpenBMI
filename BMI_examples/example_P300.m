
%% load - see the example mat folder
load cnt_p300_off;
load cnt_p300_on;
 	 
CNT_off = cnt_p300_off;
CNT_on = cnt_p300_on;
clear cnt_p300_off cnt_p300_on

% cnt variables
% cnt.t  : time information 
% cnt.fs : sampling frequency
% cnt.y_dec : class information (e.g., target = 1, non-target = 2)
% cnt.y_logic : logical format of class inforamtion 
% cnt_y_class : class name (e.g., target, non-target)
% cnt.class : number of class 
% cnt.chan : number of electrodes
% cnt. x : raw eeg signals 
 	 
% revised 2017.11.11 - Oyeon Kwon (oy_kwon@korea.ac.kr)
%% Initialization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% for offline
ival = [-200 800];
baseTime =[-200 0];
selTime =[0 800];
nFeatures=10;
 	 
% for online 
char_seq= {'A', 'B', 'C', 'D', 'E', 'F', ...
    'G', 'H', 'I', 'J', 'K', 'L', ...
    'M', 'N', 'O', 'P', 'Q', 'R', ...
    'S', 'T', 'U', 'V', 'W', 'X', ...
    'Y', 'Z', '1', '2', '3', '4', ...
    '5', '6', '7', '8', '9', '_'};
 	 
fs = CNT_on.fs;
count_char=1;
spellerText_on='PATTERN_RECOGNITION_MACHINE_LEARNING';
load('random_cell_order.mat');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Make the classifier
cnt=prep_filter(CNT_off, {'frequency', [0.5 40]});
smt=prep_segmentation(cnt, {'interval', ival});
smt=prep_baseline(smt, {'Time',baseTime});
smt=prep_selectTime(smt, {'Time',selTime});
fv=func_featureExtraction(smt,{'feature','erpmean';'nMeans',nFeatures});
[nDat, nTrials, nChans]= size(fv.x);
fv.x= reshape(permute(fv.x,[1 3 2]), [nDat*nChans nTrials]);
[clf_param] = func_train(fv,{'classifier','LDA'});
clear cnt smt fv nDat nTrials nChans 
%%
cnt =prep_filter(CNT_on, {'frequency', [0.5 40]});
smt=prep_segmentation(cnt, {'interval', ival});
smt=prep_baseline(smt, {'Time',baseTime});
smt=prep_selectTime(smt, {'Time',selTime});
 	 
dat.x= smt.x;
dat.fs = smt.fs;
dat.ival = smt.ival;
dat.t = smt.t;
 	 
% divide for each character
for add=1:60:length(dat.t)
    dat_char_all.x(:,:,:,count_char)=dat.x(:,[add:add+59],:);            % 1 stimulus : 0.1875s ; 27 stimulus : 5.0625s // 2 sequences : 4.5 s
    dat_char_all.t(:,:,:,count_char)=dat.t(:,[add:add+59]);            % 1 stimulus : 0.1875s ; 27 stimulus : 5.0625s // 2 sequences : 4.5 s
    count_char=count_char+1;
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
        t_char2(char,nSeq)= char_seq{a2}; % for each sequence
        t_char22(char,nc)= char_seq{a2}; % for each run 
        
        nc=nc+1;
        in_nc=in_nc+1;
        if in_nc>12
            in_nc=1;
            nSeq=nSeq+1;
        end
    end
    clear DAT tm_Dat Y a b a1 a2
end
clear add baseTime count_char CF_PARAM char clf_param cnt dat dat_char dat_char_all tm_Dat Y DAT
clear ft_dat fv i i2 in_nc ival nc nCh nDat nFeature nFeatures nSeq nTrials segTime selTime smt char_seq nChans
clear BMI CNT EEG field file file3 marker rc_order

for seq=1:5
    for nchar=1:length(spellerText_on)
        acc2(nchar, seq)=strcmp(spellerText_on(nchar),t_char2(nchar,seq));
    end
end

no_=sum(acc2)/length(spellerText_on);
Accuracy=no_;

clear acc2 fs nchar no_ seq spellerText_on CNT_off CNT_on
