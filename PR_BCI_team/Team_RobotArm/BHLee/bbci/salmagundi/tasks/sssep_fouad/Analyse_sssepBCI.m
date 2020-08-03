clear all;clc;close all;
date='_08_06_12\';
VP_CODE='VPiz';
session ='sssep_BCI_27_new_lev6VPiz';

%date='_08_06_19\';
%VP_CODE='VPip';
% session = 'sssep_BCI_27_new_lev8_difVPip';

file = ['C:\eeg_data\',VP_CODE,date,session];
file2 = ['C:\eeg_data\',VP_CODE,date,session,'02'];
file3 = ['C:\eeg_data\',VP_CODE,date,session,'03'];
filename = {file,file2,file3};

warning('off','all')

% Wps= [40 49]/1000*2;
% [n, Ws]= cheb2ord(Wps(1), Wps(2), 3, 50);
% [filt.b, filt.a]= cheby2(n, 50, Ws);
[filt.b,filt.a]= butter(5, [5 40]/1000*2);

[cnt,mrk]= eegfile_loadBV(filename,'fs',100,'filt',filt);
%%
saving = 0; % saves figures if = 1.
modfreq =27;
modfreq1 =27;
stimDef = {'S  1','S  2';'Left','Right'};
mrk_stim= mrk_defineClasses(mrk, stimDef);
ival = [1750 4200];
fft_band = [6 35];

figure('Name','artifacts')
[mrk_clean, RCLAB, RTRIALS]= reject_varEventsAndChannels(cnt, mrk_stim, ival, 'visualize',1);

    stimDef_ref = {'S  1','S  2';'Ref_Left','Ref_Right'};
    mrk_stim_ref= mrk_defineClasses(mrk, stimDef_ref);
    ival_ref = [0 1000];
    figure('Name','artifacts_ref');
    [mrk_clean_ref, RCLAB, RTRIALS]= reject_varEventsAndChannels(cnt, mrk_stim_ref, ival_ref, 'visualize',1);

channel_list ={'FC3-CP3','FCz-CPz','FC4-CP4'};

epo_ref = cntToEpo(cnt, mrk_clean_ref, ival_ref);
epo_ref_lap =proc_laplace(epo_ref);

epo = cntToEpo(cnt, mrk_clean, ival);

epo_lap = proc_laplace(epo,'large',' lap','filter all');
epo_bi= proc_bipolarChannels(epo, channel_list);

epo_spec= proc_spectrum(epo, fft_band, kaiser(cnt.fs,2));
epo_spec_ref= proc_spectrum(epo_ref, fft_band, kaiser(cnt.fs,2));

epo_lap_spec= proc_spectrum(epo_lap, fft_band, kaiser(cnt.fs,2));
epo_spec_ref_lap = proc_spectrum(epo_ref_lap, fft_band, kaiser(cnt.fs,2));

epo_bi_spec= proc_spectrum(epo_bi, fft_band, kaiser(cnt.fs,2));

epo_lap_spec_r= proc_r_square_signed(proc_selectClasses(epo_lap_spec, 'Left','Right'));
epo_spec_r= proc_r_square_signed(proc_selectClasses(epo_spec,  'Left','Right'));
epo_spec_bi_r= proc_r_square_signed(proc_selectClasses(epo_bi_spec,  'Left','Right'));

[DAT, CSP_W, CSP_EIG, CSP_A]= proc_csp3(epo,3);
DAT_spec= proc_spectrum(DAT, fft_band, kaiser(cnt.fs,2));

mnt= getElectrodePositions(cnt.clab);
mnt= mnt_setGrid(mnt, 'tiny');
mnt= mnt_shrinkNonEEGchans(mnt);

mnt1= getElectrodePositions(cnt.clab);
mnt1= mnt_setGrid(mnt1, 'medium');
mnt1= mnt_shrinkNonEEGchans(mnt1);

mnt_lap= getElectrodePositions(epo_lap.clab);
mnt_lap= mnt_setGrid(mnt_lap, 'tiny');

mnt_bi= getElectrodePositions(epo_bi.clab);
mnt_bi= mnt_setGrid(mnt_bi, 'tiny');
mnt_bi= mnt_restrictMontage(mnt_bi, epo_bi.clab);

mnt2= getElectrodePositions(DAT.clab);
mnt2= mnt_setGrid(mnt2, 'medium');

opt_spec= defopt_spec;
opt_grid= defopt_erps;

%% Produce figures
save_path = 'C:\Documents and Settings\Min Konto\Dokumenter\Fouads Mappe\DTU\tiendesem\BCI-SSSEP Projekt\Matlab\master_fouad\';
h_t = figure('Name','Epochs - Timeseries'); grid_plot(epo, mnt)
if saving == 1
    saveas(h_t,[save_path,VP_CODE,'_figs\Epochs - Timeseries'],'png')
end

h_raw=figure('Name','Epochs_raw - FFT');H=grid_plot(epo_spec, mnt,opt_spec);
grid_markIval([modfreq-1 modfreq+1]);
hgrid= grid_addBars(epo_spec_r, 'h_scale',H.scale, 'colormap',cmap_posneg(21),'cLim','sym');
if saving == 1
    saveas(h_raw,[save_path,VP_CODE,'_figs\Epochs_raw_FFT'],'png')
end

h_lap=figure('Name','Epochs_lap - FFT');H=grid_plot(epo_lap_spec, mnt_lap,opt_spec);
grid_markIval([modfreq-1 modfreq+1]);
grid_markIval([modfreq1-1 modfreq1+1]);
grid_addBars(epo_lap_spec_r, 'h_scale',H.scale, 'colormap',cmap_posneg(21),'cLim','sym');
if saving==1
    saveas(h_lap,[save_path,VP_CODE,'_figs\Epochs_lap - FFT'],'png')
end

h_raw_full = figure('Name','Epochs_raw - FFT (Full Montage)');H=grid_plot(epo_spec, mnt1,opt_spec);
grid_markIval([modfreq-1 modfreq+1]);
grid_markIval([modfreq1-1 modfreq1+1]);
if saving==1
    saveas(h_raw_full,[save_path,VP_CODE,'_figs\Epochs_raw - FFT (Full Montage)'],'png')
end

h_lap_full = figure('Name','Epochs_lap - FFT (Full Montage)');H=grid_plot(epo_lap_spec, mnt1,opt_spec);
grid_markIval([modfreq-1 modfreq+1]);
grid_markIval([modfreq1-1 modfreq1+1]);
if saving == 1
    saveas(h_lap_full,[save_path,VP_CODE,'_figs\Epochs_lap - FFT (Full Montage)'],'png')
end

h_csp_full = figure('Name','Epochs_csp - FFT (Full Montage)');H=grid_plot(DAT_spec, mnt2,opt_spec);
grid_markIval([modfreq-1 modfreq+1]);
grid_markIval([modfreq1-1 modfreq1+1]);
if saving == 1
    saveas(h_lap_full,[save_path,VP_CODE,'_figs\Epochs_lap - FFT (Full Montage)'],'png')
end

 figure;
 subplot(2,2,1);plotChannel(epo_lap_spec,'C3 lap','yLim',[-10 10]);
 subplot(2,2,2);plotChannel(epo_lap_spec,'C4 lap','yLim',[-10 10]);
 subplot(2,2,3);plotChannel(epo_lap_spec,'CP3 lap','yLim',[-10 10]);
 subplot(2,2,4);plotChannel(epo_lap_spec,'CP4 lap','yLim',[-10 10]);
grid_markIval([modfreq-1 modfreq+1]);
grid_markIval([modfreq1-1 modfreq1+1]);

figure;grid_plot(epo_spec_ref_lap,mnt1,opt_spec)
grid_markIval([modfreq-1 modfreq+1]);
grid_markIval([modfreq1-1 modfreq1+1]);

figure;grid_plot(epo_spec_ref_lap,mnt,opt_spec)
grid_markIval([modfreq-1 modfreq+1]);
grid_markIval([modfreq1-1 modfreq1+1]);

h_bi = figure('Name','Epochs_bipol - FFT');H=grid_plot(epo_bi_spec, mnt_bi,opt_spec);
grid_markIval([modfreq-1 modfreq+1]);
grid_markIval([modfreq1-1 modfreq1+1]);
% grid_addBars(epo_spec_bi_r, 'h_scale',H.scale, 'colormap',cmap_posneg(21),'cLim','sym');
if saving ==1
    saveas(h_bi,[save_path,VP_CODE,'_figs\Epochs_bipol - FFT'],'png')
end

h_ref = figure('Name','REF - FFT');H=grid_plot(epo_spec_ref, mnt,opt_spec);
if saving==1
    saveas(h_ref,[save_path,VP_CODE,'_figs\REF - FFT'],'png')
end
