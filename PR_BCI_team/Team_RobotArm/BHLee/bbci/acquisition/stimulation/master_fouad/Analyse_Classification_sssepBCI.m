% Load data and downsampling 
% clear all;clc;close all;
date='_08_06_19\';
VP_CODE='VPip';
EEG_dir ='C:\eeg_data\'; 

% data recorded with different stimuli frequencies (21 and 27 Hz) 
%session = 'sssep_BCI_27_new_lev7VPiz';
 session = 'sssep_BCI_27_new_lev8_difVPip';

file = [EEG_dir,VP_CODE,date,session];
file2 = [EEG_dir,VP_CODE,date,session,'02'];
file3 = [EEG_dir,VP_CODE,date,session,'03'];
filename = {file,file2,file3};
band= [20 22];
warning('off','all')

% Apply lowpass-filter to avoid aliasing
Wps= [40 49]/1000*2;
[n, Ws]= cheb2ord(Wps(1), Wps(2), 3, 50);
[filt.b, filt.a]= cheby2(n, 50, Ws);

%[filt.b,filt.a]= butter(5, band/1000*2);
[cnt,mrk]= eegfile_loadBV(filename,'filt',filt,'fs',100);
%% 
% Define markers 
stimDef = {'S  1','S  2';'Left','Right'};
mrk_stim= mrk_defineClasses(mrk, stimDef);
ival = [1750 4200];
% Artifact removal  
figure('Name','artifacts')
[mrk_clean, RCLAB, RTRIALS]= reject_varEventsAndChannels(cnt, mrk_stim, ival, 'visualize',1,'whiskerlength', 3);

% Bandpass filtering before CSP around stimuli frequency
 [b,a]= butter(9, band/100*2);
 cnt_flt= proc_filt(cnt, b, a);

num_filt = 3;
% Epochs around visual cue onset (750 ms after)
epo = cntToEpo(cnt_flt, mrk_clean, ival);

% Remove some channels
epo= proc_selectChannels(epo, {'not','E*','FP*','AF*','O*','PO*'});

% % % epo = proc_pr_pca(epo)
% % % epo= proc_selectChannels(epo, {'CP*'});
% % %  epo= proc_movingAverage(epo, 100,'causal');
% % %fv =proc_laplace(epo,'large','lap','filter all');
% % 
% % %fv = proc_bandPower(epo,[26 28]);
% % % fv = proc_variance(fv);
% % % fv = proc_logarithm(fv);
% % 
% % %[loss, loss_std, out_test, memo] = xvalidation(fv, 'LDA');
% % 
% % 
% % % mnt= getElectrodePositions(cnt.clab);
% % % mnt= mnt_setGrid(mnt, 'medium');
% % % mnt= mnt_shrinkNonEEGchans(mnt);
% % % 
% % % ival_ers=[-250 4200];
% % % ers = cntToEpo(cnt_flt, mrk_clean, ival_ers);
% % % ers = proc_selectChannels(ers, 'not','E*');
% % % 
% % % ers= proc_envelope(ers);
% % % figure;grid_plot(ers, mnt);grid_markTimePoint(1000);grid_markIval([3700 3825])
% % % 
% % % 
% % % epo_csp_spec= proc_spectrum(epo_csp, band, kaiser(cnt.fs,2));

% Only for plotting the filters and patterns
[epo_csp, csp_w, csp_eig, csp_a]= proc_csp3(epo,num_filt);
opt_scalp_csp = defopt_scalp_csp;
mnt= getElectrodePositions(cnt.clab);
mnt= mnt_setGrid(mnt, 'tiny');
mnt= mnt_shrinkNonEEGchans(mnt);

figure;
plotCSPanalysis(epo, mnt, csp_w, csp_a, csp_eig, opt_scalp_csp, 'colorOrder',[]);

%%
% Classifier validation 
proc= struct('memo', 'csp_w');
proc.train= ['[fv,csp_w]= proc_csp3(fv, 3); ' ...
             'fv = proc_variance(fv); ' ...
             'fv = proc_logarithm(fv); '];
proc.apply= ['fv= proc_linearDerivation(fv, csp_w); ' ...
             'fv = proc_variance(fv); ' ...
             'fv = proc_logarithm(fv); ']; 
% ten cross ten validation
         [loss, loss_std, out_test, memo] = xvalidation(epo, 'LDA', 'proc',proc);
