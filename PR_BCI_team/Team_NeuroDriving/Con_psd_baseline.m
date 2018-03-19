clear all; close all; clc;

% subjectList_Fatigue = { '20150210_kko'};
subjectList_Fatigue = { '20150210_kko','20150210_sjlee',...
    '20150212_bjkim','20150212_jswoo','20150223_hwkim',...
    '20150226_suyoon',...
    '20150304_dhkim','20150306_swpark'};
%     '20150223_hwkim',...성능완전구림

% subjectList_Distraction = {'20150402_kko'};
subjectList_Distraction = { '20150402_kko','20150408_sjlee',...
    '20150407_bjkim','20150327_jswoo',...
    '20150330_hwkim',...
    '20150401_suyoon',...
    '20150330_dhkim','20150403_swpark'};

% ldxSubjectlist = {'kko','sjlee','bjkim','jswoo','hwkim','suyoon','dhkim','swpark'};

load('BTB.mat');
file_Fatigue = 'D:\LG\Sleep\data\sleep_integrated(2)';
file_Dist = 'D:\LG\LG\LG_Integrated\Distraction';

for s = 1 : length(subjectList_Distraction)
    
    [cnt_dist, mrk_dist, mnt_dist] = eegfile_loadMatlab(strcat(file_Dist, '\', subjectList_Distraction{s}));
    [cnt_fati, mrk_fati, mnt_fati] = eegfile_loadMatlab(strcat(file_Fatigue, '\', subjectList_Fatigue{s}));
    
    %%  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Distraction>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    %% Segmentation of bio-signals with 1 seconds length of epoch
    % Epo included in exception condition will be abandoned during process
    epoch_dist = segmentationDistraction_eeg(cnt_dist, mrk_dist);
    
    Dist_epoch_target.x = epoch_dist.x;
    Dist_epoch_target.misc = epoch_dist.misc;
    Dist_epoch_target.fs = cnt_dist.fs;
    
    Dist_epoch_non.x = epoch_dist.non_x;
    Dist_epoch_non.misc = epoch_dist.non_misc;
    Dist_epoch_non.fs = cnt_dist.fs;
    
    for c = 1:4
        [Dist_trainTarget,Dist_trainNon, Dist_testTarget, Dist_testNon, Dist_trainLabel, Dist_testLabel] = cross_validation_set(Dist_epoch_target.x,Dist_epoch_non.x,4);
        
        Dist_Tr_epoch_non.x = Dist_epoch_non.x(:,:,Dist_trainNon(c,:));
        Dist_Tr_epoch_non.fs = Dist_epoch_non.fs;
        Dist_Tr_epoch_target.x = Dist_epoch_target.x(:,:,Dist_trainTarget(c,:));
        Dist_Tr_epoch_target.fs = Dist_epoch_target.fs;
        
        Dist_Te_epoch_non.x = Dist_epoch_non.x(:,:,Dist_testNon(c,:));
        Dist_Te_epoch_non.fs = Dist_epoch_non.fs;
        Dist_Te_epoch_target.x = Dist_epoch_target.x(:,:,Dist_testTarget(c,:));
        Dist_Te_epoch_target.fs = Dist_epoch_target.fs;
        
        %% Power spectrum analysis for EEG signals (1-30)
        % Two kinds of spectrum features were extracted (Averaged and
        % whole channel values of power spectrum density
        
        %% Training data
        
        Dist_Tr_Non_psd = proc_spectrum(Dist_Tr_epoch_non, [4 8]);
        Dist_Tr_Non_psd = proc_subtractMean(Dist_Tr_Non_psd,'median',3);
        Dist_Tr_Non_psd_theta = squeeze(mean(Dist_Tr_Non_psd.x,1));
        Dist_Tr_Target_psd = proc_spectrum(Dist_Tr_epoch_target, [4 8]);
        Dist_Tr_Target_psd = proc_subtractMean(Dist_Tr_Target_psd, 'median',3);
        Dist_Tr_Target_psd_theta = squeeze(mean(Dist_Tr_Target_psd.x,1));
        
        Dist_Tr_Non_psd = proc_spectrum(Dist_Tr_epoch_non, [8 13]);
                Dist_Tr_Non_psd = proc_subtractMean(Dist_Tr_Non_psd, 'median',3);
        Dist_Tr_Non_psd_alpha = squeeze(mean(Dist_Tr_Non_psd.x,1));
        Dist_Tr_Target_psd = proc_spectrum(Dist_Tr_epoch_target, [8 13]);
        Dist_Tr_Target_psd = proc_subtractMean(Dist_Tr_Target_psd, 'median',3);
        Dist_Tr_Target_psd_alpha = squeeze(mean(Dist_Tr_Target_psd.x,1));
        
        Dist_Tr_Non_psd = proc_spectrum(Dist_Tr_epoch_non, [13 20]);
                Dist_Tr_Non_psd = proc_subtractMean(Dist_Tr_Non_psd, 'median',3);
        Dist_Tr_Non_psd_lowBeta = squeeze(mean(Dist_Tr_Non_psd.x,1));
        Dist_Tr_Target_psd = proc_spectrum(Dist_Tr_epoch_target, [13 20]);
        Dist_Tr_Target_psd = proc_subtractMean(Dist_Tr_Target_psd, 'median',3);
        Dist_Tr_Target_psd_lowBeta = squeeze(mean(Dist_Tr_Target_psd.x,1));
        
        Dist_Tr_Non_psd = proc_spectrum(Dist_Tr_epoch_non, [20 30]);
                Dist_Tr_Non_psd = proc_subtractMean(Dist_Tr_Non_psd, 'median',3);
        Dist_Tr_Non_psd_highBeta = squeeze(mean(Dist_Tr_Non_psd.x,1));
        Dist_Tr_Target_psd = proc_spectrum(Dist_Tr_epoch_target, [20 30]);
        Dist_Tr_Target_psd = proc_subtractMean(Dist_Tr_Target_psd, 'median',3);
        Dist_Tr_Target_psd_highBeta = squeeze(mean(Dist_Tr_Target_psd.x,1));
        
        Dist_Tr_Non_psd = proc_spectrum(Dist_Tr_epoch_non, [13 30]);
                Dist_Tr_Non_psd = proc_subtractMean(Dist_Tr_Non_psd, 'median',3);
        Dist_Tr_Non_psd_beta = squeeze(mean(Dist_Tr_Non_psd.x,1));
        Dist_Tr_Target_psd = proc_spectrum(Dist_Tr_epoch_target, [13 30]);
        Dist_Tr_Target_psd = proc_subtractMean(Dist_Tr_Target_psd, 'median',3);
        Dist_Tr_Target_psd_beta = squeeze(mean(Dist_Tr_Target_psd.x,1));
        
        %       Permuting each frequency band
        Dist_Tr_Non_psd_theta = permute(Dist_Tr_Non_psd_theta ,[2 1]);
        Dist_Tr_Target_psd_theta  = permute(Dist_Tr_Target_psd_theta ,[2 1]);
        Dist_Tr_Non_psd_alpha = permute(Dist_Tr_Non_psd_alpha ,[2 1]);
        Dist_Tr_Target_psd_alpha  = permute(Dist_Tr_Target_psd_alpha ,[2 1]);
        Dist_Tr_Non_psd_lowBeta = permute(Dist_Tr_Non_psd_lowBeta ,[2 1]);
        Dist_Tr_Target_psd_lowBeta  = permute(Dist_Tr_Target_psd_lowBeta ,[2 1]);
        Dist_Tr_Non_psd_highBeta = permute(Dist_Tr_Non_psd_highBeta ,[2 1]);
        Dist_Tr_Target_psd_highBeta  = permute(Dist_Tr_Target_psd_highBeta ,[2 1]);
        Dist_Tr_Non_psd_beta = permute(Dist_Tr_Non_psd_beta ,[2 1]);
        Dist_Tr_Target_psd_beta  = permute(Dist_Tr_Target_psd_beta ,[2 1]);
        
        
        Dist_Tr_Non_psd_total = cat(3,Dist_Tr_Non_psd_theta,Dist_Tr_Non_psd_alpha,Dist_Tr_Non_psd_lowBeta,Dist_Tr_Non_psd_highBeta,Dist_Tr_Non_psd_beta );
        Dist_Tr_Target_psd_total = cat(3,Dist_Tr_Target_psd_theta,Dist_Tr_Target_psd_alpha,Dist_Tr_Target_psd_lowBeta,Dist_Tr_Target_psd_highBeta,Dist_Tr_Target_psd_beta );
        
        %% Test data
        
        Dist_Te_Non_psd = proc_spectrum(Dist_Te_epoch_non, [4 8]);
         Dist_Te_Non_psd = proc_subtractMean(Dist_Te_Non_psd, 'median',3);
        Dist_Te_Non_psd_theta = squeeze(mean(Dist_Te_Non_psd.x,1));
        Dist_Te_Target_psd = proc_spectrum(Dist_Te_epoch_target, [4 8]);
         Dist_Te_Target_psd = proc_subtractMean(Dist_Te_Target_psd, 'median',3);
        Dist_Te_Target_psd_theta = squeeze(mean(Dist_Te_Target_psd.x,1));
        
        Dist_Te_Non_psd = proc_spectrum(Dist_Te_epoch_non, [8 13]);
        Dist_Te_Non_psd = proc_subtractMean(Dist_Te_Non_psd, 'median',3);
        Dist_Te_Non_psd_alpha = squeeze(mean(Dist_Te_Non_psd.x,1));
        Dist_Te_Target_psd = proc_spectrum(Dist_Te_epoch_target, [8 13]);
        Dist_Te_Target_psd = proc_subtractMean(Dist_Te_Target_psd, 'median',3);
        Dist_Te_Target_psd_alpha = squeeze(mean(Dist_Te_Target_psd.x,1));
        
        Dist_Te_Non_psd = proc_spectrum(Dist_Te_epoch_non, [13 20]);
        Dist_Te_Non_psd = proc_subtractMean(Dist_Te_Non_psd, 'median',3);
        Dist_Te_Non_psd_lowBeta = squeeze(mean(Dist_Te_Non_psd.x,1));
        Dist_Te_Target_psd = proc_spectrum(Dist_Te_epoch_target, [13 20]);
        Dist_Te_Target_psd = proc_subtractMean(Dist_Te_Target_psd, 'median',3);
        Dist_Te_Target_psd_lowBeta = squeeze(mean(Dist_Te_Target_psd.x,1));
        
        Dist_Te_Non_psd = proc_spectrum(Dist_Te_epoch_non, [20 30]);
        Dist_Te_Non_psd = proc_subtractMean(Dist_Te_Non_psd, 'median',3);
        Dist_Te_Non_psd_highBeta = squeeze(mean(Dist_Te_Non_psd.x,1));
        Dist_Te_Target_psd = proc_spectrum(Dist_Te_epoch_target, [20 30]);
        Dist_Te_Target_psd = proc_subtractMean(Dist_Te_Target_psd, 'median',3);
        Dist_Te_Target_psd_highBeta = squeeze(mean(Dist_Te_Target_psd.x,1));
        
        Dist_Te_Non_psd = proc_spectrum(Dist_Te_epoch_non, [13 30]);
        Dist_Te_Non_psd = proc_subtractMean(Dist_Te_Non_psd, 'median',3);
        Dist_Te_Non_psd_beta = squeeze(mean(Dist_Te_Non_psd.x,1));
        Dist_Te_Target_psd = proc_spectrum(Dist_Te_epoch_target, [13 30]);
        Dist_Te_Target_psd = proc_subtractMean(Dist_Te_Target_psd, 'median',3);
        Dist_Te_Target_psd_beta = squeeze(mean(Dist_Te_Target_psd.x,1));
        
        % permuting each frequency band
        Dist_Te_Non_psd_theta = permute(Dist_Te_Non_psd_theta ,[2 1]);
        Dist_Te_Target_psd_theta  = permute(Dist_Te_Target_psd_theta ,[2 1]);
        Dist_Te_Non_psd_alpha = permute(Dist_Te_Non_psd_alpha ,[2 1]);
        Dist_Te_Target_psd_alpha  = permute(Dist_Te_Target_psd_alpha ,[2 1]);
        Dist_Te_Non_psd_lowBeta = permute(Dist_Te_Non_psd_lowBeta ,[2 1]);
        Dist_Te_Target_psd_lowBeta  = permute(Dist_Te_Target_psd_lowBeta ,[2 1]);
        Dist_Te_Non_psd_highBeta = permute(Dist_Te_Non_psd_highBeta ,[2 1]);
        Dist_Te_Target_psd_highBeta  = permute(Dist_Te_Target_psd_highBeta ,[2 1]);
        Dist_Te_Non_psd_beta = permute(Dist_Te_Non_psd_beta ,[2 1]);
        Dist_Te_Target_psd_beta  = permute(Dist_Te_Target_psd_beta ,[2 1]);
        
        
        Dist_Te_Non_psd_total = cat(3,Dist_Te_Non_psd_theta,Dist_Te_Non_psd_alpha,Dist_Te_Non_psd_lowBeta,Dist_Te_Non_psd_highBeta,Dist_Te_Non_psd_beta );
        Dist_Te_Target_psd_total = cat(3,Dist_Te_Target_psd_theta,Dist_Te_Target_psd_alpha,Dist_Te_Target_psd_lowBeta,Dist_Te_Target_psd_highBeta,Dist_Te_Target_psd_beta );
        
        
        %% >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Fatigue>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        %% Synchronization of KSS
        % Interpolation of KSS scores
        interKSS = zeros(length(cnt_fati.x), 1) + 1; %interKSS 초기화
        % Decision of KSS score of start and end point based on posterior and
        % prior KSS scores within range 1 ~ 9 (-1 and +1)
        interKSS(1) = mrk_fati.kss.toe(1) - 1; interKSS(end) = mrk_fati.kss.toe(end) + 1;
        interKSS(find(interKSS < 1)) = 1; interKSS(find(interKSS > 9)) = 9;
        for i = 1 : length(mrk_fati.kss.toe)
            if i == length(mrk_fati.kss.toe)
                interKSS(1 : mrk_fati.kss.pos(1)) = linspace(interKSS(1), mrk_fati.kss.toe(1), length(1 : length(mrk_fati.kss.pos(1))));
                interKSS(mrk_fati.kss.pos(i) : end) = linspace(mrk_fati.kss.toe(i), interKSS(end), length(mrk_fati.kss.pos(i) : length(interKSS)));
            else
                % increasing 1ms interval from the recent KSS position to the next KSS position
                interKSS(mrk_fati.kss.pos(i) : mrk_fati.kss.pos(i + 1)) = linspace(mrk_fati.kss.toe(i), mrk_fati.kss.toe(i + 1), length(mrk_fati.kss.pos(i) : mrk_fati.kss.pos(i + 1)));
            end
        end
        
        %% Segmentation of eeg-signals with 1 seconds length of epoch
        % Epo included in exception condition will be abandoned during process
        epoch_fati = segmentationSleep_eeg(cnt_fati,mrk_fati,[1 30],interKSS);
        
        %   average of KSS value of one trial from 32 channels
        %   initializing the avg_kss vector
        avg_kss=randperm(120);
        end_trial=length(epoch_fati.x);
        
        %   class labeling at one second=one trial
        %   j=1:200 because this signals recorded as 200 Hz
        for i=1:end_trial
            for j=1:200
                avg_kss(1,i)=mean(epoch_fati.misc(j,1,i));
            end
        end
        
        %   average of KSS value each trial when KSS<5 (alert state) and KSS>=6 (fatigue state) from 32 channels
        clear lowldx_rd highldx_rd lowldx highldx;
        %     lowldx= find(avg_kss< 5); highldx = find(avg_kss>= 6);
        lowldx = find(avg_kss<=(min(avg_kss)+1));
        lowldx_rd = lowldx(1:120);
        
        highldx = find(avg_kss>=(max(avg_kss)-1));
        highldx_rd = highldx(length(highldx)-119:end);
        
        Fati_epoch_target.x = epoch_fati.x(:,:,highldx_rd);
        Fati_epoch_target.misc = epoch_fati.misc;
        Fati_epoch_target.fs = epoch_fati.fs;
        
        Fati_epoch_non.x = epoch_fati.x(:,:,lowldx_rd);
        Fati_epoch_non.misc = epoch_fati.misc;
        Fati_epoch_non.fs = epoch_fati.fs;
        
        [Fati_trainTarget,Fati_trainNon, Fati_testTarget, Fati_testNon, Fati_trainLabel, Fati_testLabel] = cross_validation_set(Fati_epoch_target.x,Fati_epoch_non.x,10);
        
        Fati_Tr_epoch_non.x = Fati_epoch_non.x(:,:,Fati_trainNon(c,:));
        Fati_Tr_epoch_non.fs = Fati_epoch_non.fs;
        Fati_Tr_epoch_target.x = Fati_epoch_target.x(:,:,Fati_trainTarget(c,:));
        Fati_Tr_epoch_target.fs = Fati_epoch_target.fs;
        
        Fati_Te_epoch_non.x = Fati_epoch_non.x(:,:,Fati_testNon(c,:));
        Fati_Te_epoch_non.fs = Fati_epoch_non.fs;
        Fati_Te_epoch_target.x = Fati_epoch_target.x(:,:,Fati_testTarget(c,:));
        Fati_Te_epoch_target.fs = Fati_epoch_target.fs;
        
        %% Power spectrum analysis for EEG signals (1-30)
        % Two kinds of spectrum features were extracted (Averaged and
        % whole channel values of power spectrum density
        
        %% Training data
        
        Fati_Tr_Non_psd = proc_spectrum(Fati_Tr_epoch_non, [4 8]);
        Fati_Tr_Non_psd = proc_subtractMean(Fati_Tr_Non_psd, 'median',3);
        Fati_Tr_Non_psd_theta = squeeze(mean(Fati_Tr_Non_psd.x,1));
        Fati_Tr_Target_psd = proc_spectrum(Fati_Tr_epoch_target, [4 8]);
        Fati_Tr_Target_psd = proc_subtractMean(Fati_Tr_Target_psd, 'median',3);
        Fati_Tr_Target_psd_theta = squeeze(mean(Fati_Tr_Target_psd.x,1));
        
        Fati_Tr_Non_psd = proc_spectrum(Fati_Tr_epoch_non, [8 13]);
        Fati_Tr_Non_psd = proc_subtractMean(Fati_Tr_Non_psd, 'median',3);
        Fati_Tr_Non_psd_alpha = squeeze(mean(Fati_Tr_Non_psd.x,1));
        Fati_Tr_Target_psd = proc_spectrum(Fati_Tr_epoch_target, [8 13]);
        Fati_Tr_Target_psd = proc_subtractMean(Fati_Tr_Target_psd, 'median',3);
        Fati_Tr_Target_psd_alpha = squeeze(mean(Fati_Tr_Target_psd.x,1));
        
        Fati_Tr_Non_psd = proc_spectrum(Fati_Tr_epoch_non, [13 20]);
        Fati_Tr_Non_psd = proc_subtractMean(Fati_Tr_Non_psd, 'median',3);
        Fati_Tr_Non_psd_lowBeta = squeeze(mean(Fati_Tr_Non_psd.x,1));
        Fati_Tr_Target_psd = proc_spectrum(Fati_Tr_epoch_target, [13 20]);
        Fati_Tr_Target_psd = proc_subtractMean(Fati_Tr_Target_psd, 'median',3);
        Fati_Tr_Target_psd_lowBeta = squeeze(mean(Fati_Tr_Target_psd.x,1));
        
        Fati_Tr_Non_psd = proc_spectrum(Fati_Tr_epoch_non, [20 30]);
        Fati_Tr_Non_psd = proc_subtractMean(Fati_Tr_Non_psd, 'median',3);
        Fati_Tr_Non_psd_highBeta = squeeze(mean(Fati_Tr_Non_psd.x,1));
        Fati_Tr_Target_psd = proc_spectrum(Fati_Tr_epoch_target, [20 30]);
        Fati_Tr_Target_psd = proc_subtractMean(Fati_Tr_Target_psd, 'median',3);
        Fati_Tr_Target_psd_highBeta = squeeze(mean(Fati_Tr_Target_psd.x,1));
        
        Fati_Tr_Non_psd = proc_spectrum(Fati_Tr_epoch_non, [13 30]);
        Fati_Tr_Non_psd = proc_subtractMean(Fati_Tr_Non_psd, 'median',3);
        Fati_Tr_Non_psd_beta = squeeze(mean(Fati_Tr_Non_psd.x,1));
        Fati_Tr_Target_psd = proc_spectrum(Fati_Tr_epoch_target, [13 30]);
        Fati_Tr_Target_psd = proc_subtractMean(Fati_Tr_Target_psd, 'median',3);
        Fati_Tr_Target_psd_beta = squeeze(mean(Fati_Tr_Target_psd.x,1));
        
        %       Permuting each frequency band
        Fati_Tr_Non_psd_theta = permute(Fati_Tr_Non_psd_theta ,[2 1]);
        Fati_Tr_Target_psd_theta  = permute(Fati_Tr_Target_psd_theta ,[2 1]);
        Fati_Tr_Non_psd_alpha = permute(Fati_Tr_Non_psd_alpha ,[2 1]);
        Fati_Tr_Target_psd_alpha  = permute(Fati_Tr_Target_psd_alpha ,[2 1]);
        Fati_Tr_Non_psd_lowBeta = permute(Fati_Tr_Non_psd_lowBeta ,[2 1]);
        Fati_Tr_Target_psd_lowBeta  = permute(Fati_Tr_Target_psd_lowBeta ,[2 1]);
        Fati_Tr_Non_psd_highBeta = permute(Fati_Tr_Non_psd_highBeta ,[2 1]);
        Fati_Tr_Target_psd_highBeta  = permute(Fati_Tr_Target_psd_highBeta ,[2 1]);
        Fati_Tr_Non_psd_beta = permute(Fati_Tr_Non_psd_beta ,[2 1]);
        Fati_Tr_Target_psd_beta  = permute(Fati_Tr_Target_psd_beta ,[2 1]);
        
        Fati_Tr_Non_psd_total = cat(3,Fati_Tr_Non_psd_theta,Fati_Tr_Non_psd_alpha,Fati_Tr_Non_psd_lowBeta,Fati_Tr_Non_psd_highBeta,Fati_Tr_Non_psd_beta );
        Fati_Tr_Target_psd_total = cat(3,Fati_Tr_Target_psd_theta,Fati_Tr_Target_psd_alpha,Fati_Tr_Target_psd_lowBeta,Fati_Tr_Target_psd_highBeta,Fati_Tr_Target_psd_beta );
        %
        %% Test data
        
        Fati_Te_Non_psd = proc_spectrum(Fati_Te_epoch_non, [4 8]);
        Fati_Te_Non_psd = proc_subtractMean(Fati_Te_Non_psd, 'median',3);
        Fati_Te_Non_psd_theta = squeeze(mean(Fati_Te_Non_psd.x,1));
        Fati_Te_Target_psd = proc_spectrum(Fati_Te_epoch_target, [4 8]);
        Fati_Te_Target_psd = proc_subtractMean(Fati_Te_Target_psd, 'median',3);
        Fati_Te_Target_psd_theta = squeeze(mean(Fati_Te_Target_psd.x,1));
        
        Fati_Te_Non_psd = proc_spectrum(Fati_Te_epoch_non, [8 13]);
        Fati_Te_Non_psd = proc_subtractMean(Fati_Te_Non_psd, 'median',3);
        Fati_Te_Non_psd_alpha = squeeze(mean(Fati_Te_Non_psd.x,1));
        Fati_Te_Target_psd = proc_spectrum(Fati_Te_epoch_target, [8 13]);
        Fati_Te_Target_psd = proc_subtractMean(Fati_Te_Target_psd, 'median',3);
        Fati_Te_Target_psd_alpha = squeeze(mean(Fati_Te_Target_psd.x,1));
        %
        Fati_Te_Non_psd = proc_spectrum(Fati_Te_epoch_non, [13 20]);
        Fati_Te_Non_psd = proc_subtractMean(Fati_Te_Non_psd, 'median',3);
        Fati_Te_Non_psd_lowBeta = squeeze(mean(Fati_Te_Non_psd.x,1));
        Fati_Te_Target_psd = proc_spectrum(Fati_Te_epoch_target, [13 20]);
        Fati_Te_Target_psd = proc_subtractMean(Fati_Te_Target_psd, 'median',3);
        Fati_Te_Target_psd_lowBeta = squeeze(mean(Fati_Te_Target_psd.x,1));
        
        Fati_Te_Non_psd = proc_spectrum(Fati_Te_epoch_non, [20 30]);
        Fati_Te_Non_psd = proc_subtractMean(Fati_Te_Non_psd, 'median',3);
        Fati_Te_Non_psd_highBeta = squeeze(mean(Fati_Te_Non_psd.x,1));
        Fati_Te_Target_psd = proc_spectrum(Fati_Te_epoch_target, [20 30]);
        Fati_Te_Target_psd = proc_subtractMean(Fati_Te_Target_psd, 'median',3);
        Fati_Te_Target_psd_highBeta = squeeze(mean(Fati_Te_Target_psd.x,1));
        
        Fati_Te_Non_psd = proc_spectrum(Fati_Te_epoch_non, [13 30]);
        Fati_Te_Non_psd = proc_subtractMean(Fati_Te_Non_psd, 'median',3);
        Fati_Te_Non_psd_beta = squeeze(mean(Fati_Te_Non_psd.x,1));
        Fati_Te_Target_psd = proc_spectrum(Fati_Te_epoch_target, [13 30]);
        Fati_Te_Target_psd = proc_subtractMean(Fati_Te_Target_psd, 'median',3);
        Fati_Te_Target_psd_beta = squeeze(mean(Fati_Te_Target_psd.x,1));
        
        %       permuting each frequency band
        Fati_Te_Non_psd_theta = permute(Fati_Te_Non_psd_theta ,[2 1]);
        Fati_Te_Target_psd_theta  = permute(Fati_Te_Target_psd_theta ,[2 1]);
        Fati_Te_Non_psd_alpha = permute(Fati_Te_Non_psd_alpha ,[2 1]);
        Fati_Te_Target_psd_alpha  = permute(Fati_Te_Target_psd_alpha ,[2 1]);
        Fati_Te_Non_psd_lowBeta = permute(Fati_Te_Non_psd_lowBeta ,[2 1]);
        Fati_Te_Target_psd_lowBeta  = permute(Fati_Te_Target_psd_lowBeta ,[2 1]);
        Fati_Te_Non_psd_highBeta = permute(Fati_Te_Non_psd_highBeta ,[2 1]);
        Fati_Te_Target_psd_highBeta  = permute(Fati_Te_Target_psd_highBeta ,[2 1]);
        Fati_Te_Non_psd_beta = permute(Fati_Te_Non_psd_beta ,[2 1]);
        Fati_Te_Target_psd_beta  = permute(Fati_Te_Target_psd_beta ,[2 1]);
        
        
        Fati_Te_Non_psd_total = cat(3,Fati_Te_Non_psd_theta,Fati_Te_Non_psd_alpha,Fati_Te_Non_psd_lowBeta,Fati_Te_Non_psd_highBeta,Fati_Te_Non_psd_beta );
        Fati_Te_Target_psd_total = cat(3,Fati_Te_Target_psd_theta,Fati_Te_Target_psd_alpha,Fati_Te_Target_psd_lowBeta,Fati_Te_Target_psd_highBeta,Fati_Te_Target_psd_beta );
        
        %% Concatenating both data
        %%         Training data
        
        Con_Tr_Non_psd_total = cat(1,Fati_Tr_Non_psd_total, Dist_Tr_Non_psd_total);
        Con_Tr_Target_psd_total = cat(1,Fati_Tr_Target_psd_total, Dist_Tr_Target_psd_total);
        
        %         nontarget
        Con_Tr_Non_psd_theta = cat(1, Fati_Tr_Non_psd_theta, Dist_Tr_Non_psd_theta);
        Con_Tr_Non_psd_alpha = cat(1, Fati_Tr_Non_psd_alpha, Dist_Tr_Non_psd_alpha);
        Con_Tr_Non_psd_lowBeta = cat(1, Fati_Tr_Non_psd_lowBeta, Dist_Tr_Non_psd_lowBeta);
        Con_Tr_Non_psd_highBeta = cat(1, Fati_Tr_Non_psd_highBeta, Dist_Tr_Non_psd_highBeta);
        Con_Tr_Non_psd_beta = cat(1, Fati_Tr_Non_psd_beta, Dist_Tr_Non_psd_beta);
        
        %         target
        Con_Tr_Target_psd_theta = cat(1, Fati_Tr_Target_psd_theta, Dist_Tr_Target_psd_theta);
        Con_Tr_Target_psd_alpha = cat(1, Fati_Tr_Target_psd_alpha, Dist_Tr_Target_psd_alpha);
        Con_Tr_Target_psd_lowBeta = cat(1, Fati_Tr_Target_psd_lowBeta, Dist_Tr_Target_psd_lowBeta);
        Con_Tr_Target_psd_highBeta = cat(1, Fati_Tr_Target_psd_highBeta, Dist_Tr_Target_psd_highBeta);
        Con_Tr_Target_psd_beta = cat(1, Fati_Tr_Target_psd_beta, Dist_Tr_Target_psd_beta);
        
        %%         Test data
        Con_Te_Non_psd_total = cat(1,Fati_Te_Non_psd_total, Dist_Te_Non_psd_total);
        Con_Te_Target_psd_total = cat(1,Fati_Te_Target_psd_total, Dist_Te_Target_psd_total);
        
        %         nontarget
        Con_Te_Non_psd_theta = cat(1, Fati_Te_Non_psd_theta, Dist_Te_Non_psd_theta);
        Con_Te_Non_psd_alpha = cat(1, Fati_Te_Non_psd_alpha, Dist_Te_Non_psd_alpha);
        Con_Te_Non_psd_lowBeta = cat(1, Fati_Te_Non_psd_lowBeta, Dist_Te_Non_psd_lowBeta);
        Con_Te_Non_psd_highBeta = cat(1, Fati_Te_Non_psd_highBeta, Dist_Te_Non_psd_highBeta);
        Con_Te_Non_psd_beta = cat(1, Fati_Te_Non_psd_beta, Dist_Te_Non_psd_beta);
        
        %         target
        Con_Te_Target_psd_theta = cat(1, Fati_Te_Target_psd_theta, Dist_Te_Target_psd_theta);
        Con_Te_Target_psd_alpha = cat(1, Fati_Te_Target_psd_alpha, Dist_Te_Target_psd_alpha);
        Con_Te_Target_psd_lowBeta = cat(1, Fati_Te_Target_psd_lowBeta, Dist_Te_Target_psd_lowBeta);
        Con_Te_Target_psd_highBeta = cat(1, Fati_Te_Target_psd_highBeta, Dist_Te_Target_psd_highBeta);
        Con_Te_Target_psd_beta = cat(1, Fati_Te_Target_psd_beta, Dist_Te_Target_psd_beta);
        
        
        %         Labeling
        Con_trainLabel = [zeros(1,size(Con_Tr_Target_psd_theta,1)) ones(1, size(Con_Tr_Non_psd_theta,1));
            ones(1,size(Con_Tr_Target_psd_theta,1)) zeros(1,size(Con_Tr_Non_psd_theta,1))];
        Con_testLabel = [zeros(1,size(Con_Te_Target_psd_theta,1)) ones(1,size(Con_Te_Non_psd_theta,1));
            ones(1,size(Con_Te_Target_psd_theta,1)) zeros(1,size(Con_Te_Non_psd_theta,1))];
        
        %% Classification
        %         Distraction
        
        %         alpha
        Dist_train.x = [ Dist_Tr_Target_psd_alpha ; Dist_Tr_Non_psd_alpha]';
        Dist_train.y = Dist_trainLabel;
        C_eeg = trainClassifier(Dist_train, {'train_RLDAshrink'});
        
        Dist_test.x = [Dist_Te_Target_psd_alpha; Dist_Te_Non_psd_alpha]';
        Dist_test.y = Dist_testLabel;
        pred = applyClassifier(Dist_test, C_eeg);
        
        %       test data 성능
        [X,Y,T,AUC]=performance(pred,Dist_test.y);
        Dist_alpha_acc(c)=AUC;
        Dist_alpha_mean_acc = mean(Dist_alpha_acc);
        
        %         _theta
        Dist_train.x = [ Dist_Tr_Target_psd_theta ; Dist_Tr_Non_psd_theta]';
        Dist_train.y = Dist_trainLabel;
        C_eeg = trainClassifier(Dist_train, {'train_RLDAshrink'});
        
        Dist_test.x = [Dist_Te_Target_psd_theta; Dist_Te_Non_psd_theta]';
        Dist_test.y = Dist_testLabel;
        pred = applyClassifier(Dist_test, C_eeg);
        
        %       test data 성능
        [X,Y,T,AUC]=performance(pred,Dist_test.y);
        Dist_theta_acc(c)=AUC;
        Dist_theta_mean_acc = mean(Dist_theta_acc);
        
        
        %         _lowBeta
        Dist_train.x = [ Dist_Tr_Target_psd_lowBeta ; Dist_Tr_Non_psd_lowBeta]';
        Dist_train.y = Dist_trainLabel;
        C_eeg = trainClassifier(Dist_train, {'train_RLDAshrink'});
        
        Dist_test.x = [Dist_Te_Target_psd_lowBeta; Dist_Te_Non_psd_lowBeta]';
        Dist_test.y = Dist_testLabel;
        pred = applyClassifier(Dist_test, C_eeg);
        
        %       test data 성능
        [X,Y,T,AUC]=performance(pred,Dist_test.y);
        Dist_lowBeta_acc(c)=AUC;
        Dist_lowBeta_mean_acc = mean(Dist_lowBeta_acc);
        
        %         _highBeta
        Dist_train.x = [ Dist_Tr_Target_psd_highBeta ; Dist_Tr_Non_psd_highBeta]';
        Dist_train.y = Dist_trainLabel;
        C_eeg = trainClassifier(Dist_train, {'train_RLDAshrink'});
        
        Dist_test.x = [Dist_Te_Target_psd_highBeta; Dist_Te_Non_psd_highBeta]';
        Dist_test.y = Dist_testLabel;
        pred = applyClassifier(Dist_test, C_eeg);
        
        
        %       test data 성능
        [X,Y,T,AUC]=performance(pred,Dist_test.y);
        Dist_highBeta_acc(c)=AUC;
        Dist_highBeta_mean_acc = mean(Dist_highBeta_acc);
        
        %         _beta
        Dist_train.x = [ Dist_Tr_Target_psd_beta ; Dist_Tr_Non_psd_beta]';
        Dist_train.y = Dist_trainLabel;
        C_eeg = trainClassifier(Dist_train, {'train_RLDAshrink'});
        
        Dist_test.x = [Dist_Te_Target_psd_beta; Dist_Te_Non_psd_beta]';
        Dist_test.y = Dist_testLabel;
        pred = applyClassifier(Dist_test, C_eeg);
        
        [X,Y,T,AUC]=performance(pred,Dist_test.y);
        Dist_beta_acc(c)=AUC;
        Dist_beta_mean_acc = mean(Dist_beta_acc);
        
        %         Fatigue
        
        %         _theta
        Fati_train.x = [ Fati_Tr_Target_psd_theta ; Fati_Tr_Non_psd_theta]';
        Fati_train.y = Fati_trainLabel;
        C_eeg = trainClassifier(Fati_train, {'train_RLDAshrink'});
        
        Fati_test.x = [Fati_Te_Target_psd_theta; Fati_Te_Non_psd_theta]';
        Fati_test.y = Fati_testLabel;
        pred = applyClassifier(Fati_test, C_eeg);
        
        %       test data 성능
        [X,Y,T,AUC]=performance(pred,Fati_test.y);
        Fati_theta_acc(c)=AUC;
        Fati_theta_mean_acc = mean(Fati_theta_acc);
        
        %         _alpha
        Fati_train.x = [ Fati_Tr_Target_psd_alpha ; Fati_Tr_Non_psd_alpha]';
        Fati_train.y = Fati_trainLabel;
        C_eeg = trainClassifier(Fati_train, {'train_RLDAshrink'});
        
        Fati_test.x = [Fati_Te_Target_psd_alpha; Fati_Te_Non_psd_alpha]';
        Fati_test.y = Fati_testLabel;
        pred = applyClassifier(Fati_test, C_eeg);
        
        %       test data 성능
        [X,Y,T,AUC]=performance(pred,Fati_test.y);
        Fati_alpha_acc(c)=AUC;
        Fati_alpha_mean_acc = mean(Fati_alpha_acc);
        
        %         _lowBeta
        Fati_train.x = [ Fati_Tr_Target_psd_lowBeta ; Fati_Tr_Non_psd_lowBeta]';
        Fati_train.y = Fati_trainLabel;
        C_eeg = trainClassifier(Fati_train, {'train_RLDAshrink'});
        
        Fati_test.x = [Fati_Te_Target_psd_lowBeta; Fati_Te_Non_psd_lowBeta]';
        Fati_test.y = Fati_testLabel;
        pred = applyClassifier(Fati_test, C_eeg);
        
        %       test data 성능
        [X,Y,T,AUC]=performance(pred,Fati_test.y);
        Fati_lowBeta_acc(c)=AUC;
        Fati_lowBeta_mean_acc = mean(Fati_lowBeta_acc);
        
        %         _highBeta
        Fati_train.x = [ Fati_Tr_Target_psd_highBeta ; Fati_Tr_Non_psd_highBeta]';
        Fati_train.y = Fati_trainLabel;
        C_eeg = trainClassifier(Fati_train, {'train_RLDAshrink'});
        
        Fati_test.x = [Fati_Te_Target_psd_highBeta; Fati_Te_Non_psd_highBeta]';
        Fati_test.y = Fati_testLabel;
        pred = applyClassifier(Fati_test, C_eeg);
        
        %       test data 성능
        [X,Y,T,AUC]=performance(pred,Fati_test.y);
        Fati_highBeta_acc(c)=AUC;
        Fati_highBeta_mean_acc = mean(Fati_highBeta_acc);
        
        
        %         _beta
        Fati_train.x = [ Fati_Tr_Target_psd_beta ; Fati_Tr_Non_psd_beta]';
        Fati_train.y = Fati_trainLabel;
        C_eeg = trainClassifier(Fati_train, {'train_RLDAshrink'});
        
        Fati_test.x = [Fati_Te_Target_psd_beta; Fati_Te_Non_psd_beta]';
        Fati_test.y = Fati_testLabel;
        pred = applyClassifier(Fati_test, C_eeg);
        
        %       test data 성능
        [X,Y,T,AUC]=performance(pred,Fati_test.y);
        Fati_beta_acc(c)=AUC;
        Fati_beta_mean_acc = mean(Fati_beta_acc);
        
        
        %         Fatigue&Distraction
        
        %         _theta
        Con_train.x = [ Con_Tr_Target_psd_theta ; Con_Tr_Non_psd_theta]';
        Con_train.y = Con_trainLabel;
        C_eeg = trainClassifier(Con_train, {'train_RLDAshrink'});
        
        Con_test.x = [Con_Te_Target_psd_theta; Con_Te_Non_psd_theta]';
        Con_test.y = Con_testLabel;
        pred = applyClassifier(Con_test, C_eeg);
        
        %       test data 성능
        [X,Y,T,AUC]=performance(pred,Con_test.y);
        Con_theta_acc(c)=AUC;
        Con_theta_mean_acc = mean(Con_theta_acc);
        
        %         _alpha
        Con_train.x = [ Con_Tr_Target_psd_alpha ; Con_Tr_Non_psd_alpha]';
        Con_train.y = Con_trainLabel;
        C_eeg = trainClassifier(Con_train, {'train_RLDAshrink'});
        
        Con_test.x = [Con_Te_Target_psd_alpha; Con_Te_Non_psd_alpha]';
        Con_test.y = Con_testLabel;
        pred = applyClassifier(Con_test, C_eeg);
        
        %       test data 성능
        [X,Y,T,AUC]=performance(pred,Con_test.y);
        Con_alpha_acc(c)=AUC;
        Con_alpha_mean_acc = mean(Con_alpha_acc);
        
        %         _lowBeta
        Con_train.x = [ Con_Tr_Target_psd_lowBeta ; Con_Tr_Non_psd_lowBeta]';
        Con_train.y = Con_trainLabel;
        C_eeg = trainClassifier(Con_train, {'train_RLDAshrink'});
        
        Con_test.x = [Con_Te_Target_psd_lowBeta; Con_Te_Non_psd_lowBeta]';
        Con_test.y = Con_testLabel;
        pred = applyClassifier(Con_test, C_eeg);
        
        %       test data 성능
        [X,Y,T,AUC]=performance(pred,Con_test.y);
        Con_lowBeta_acc(c)=AUC;
        Con_lowBeta_mean_acc = mean(Con_lowBeta_acc);
        
        %         _highBeta
        Con_train.x = [ Con_Tr_Target_psd_highBeta ; Con_Tr_Non_psd_highBeta]';
        Con_train.y = Con_trainLabel;
        C_eeg = trainClassifier(Con_train, {'train_RLDAshrink'});
        
        Con_test.x = [Con_Te_Target_psd_highBeta; Con_Te_Non_psd_highBeta]';
        Con_test.y = Con_testLabel;
        pred = applyClassifier(Con_test, C_eeg);
        
        %       test data 성능
        [X,Y,T,AUC]=performance(pred,Con_test.y);
        Con_highBeta_acc(c)=AUC;
        Con_highBeta_mean_acc = mean(Con_highBeta_acc);
        
        %         _beta
        Con_train.x = [ Con_Tr_Target_psd_beta ; Con_Tr_Non_psd_beta]';
        Con_train.y = Con_trainLabel;
        C_eeg = trainClassifier(Con_train, {'train_RLDAshrink'});
        
        Con_test.x = [Con_Te_Target_psd_beta; Con_Te_Non_psd_beta]';
        Con_test.y = Con_testLabel;
        pred = applyClassifier(Con_test, C_eeg);
        
        %       test data 성능
        [X,Y,T,AUC]=performance(pred,Con_test.y);
        Con_beta_acc(c)=AUC;
        Con_beta_mean_acc = mean(Con_beta_acc);
        
    end
    Dist_theta_subject_acc(s) = Dist_theta_mean_acc;
    Fati_theta_subject_acc(s) = Fati_theta_mean_acc;
    Con_theta_subject_acc(s) = Con_theta_mean_acc;
    
    Dist_alpha_subject_acc(s) = Dist_alpha_mean_acc;
    Fati_alpha_subject_acc(s) = Fati_alpha_mean_acc;
    Con_alpha_subject_acc(s) = Con_alpha_mean_acc;
    
    Dist_lowBeta_subject_acc(s) = Dist_lowBeta_mean_acc;
    Fati_lowBeta_subject_acc(s) = Fati_lowBeta_mean_acc;
    Con_lowBeta_subject_acc(s) = Con_lowBeta_mean_acc;
    
    Dist_highBeta_subject_acc(s) = Dist_highBeta_mean_acc;
    Fati_highBeta_subject_acc(s) = Fati_highBeta_mean_acc;
    Con_highBeta_subject_acc(s) = Con_highBeta_mean_acc;
    
    Dist_beta_subject_acc(s) = Dist_beta_mean_acc;
    Fati_beta_subject_acc(s) = Fati_beta_mean_acc;
    Con_beta_subject_acc(s) = Con_beta_mean_acc;
end
% grandaverage_ = mean(subject_acc);