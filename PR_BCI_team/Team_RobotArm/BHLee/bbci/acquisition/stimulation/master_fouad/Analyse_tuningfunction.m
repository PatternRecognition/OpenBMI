close all;clear all;
date='_08_05_16\';
VP_CODE='VPMichael';
load(['Para_palm_right_tap_incr_lev8',VP_CODE,'.mat'])
file = ['C:\eeg_data\',VP_CODE,date,Para.filename,VP_CODE];
%%
[filt.b,filt.a]= butter(5, [5 40]/(1000/2));
[cnt,mrk]= eegfile_loadBV(file,'fs',100,'filt',filt);
%%
modfreq =19;
modfreq1 =25;
stimDef = {['S ',int2str(modfreq)],['S ',int2str(modfreq1)];['Mod.freq',int2str(modfreq)],['Mod.freq',int2str(modfreq1)]};
mrk_stim= mrk_defineClasses(mrk, stimDef);
ival = [0 2000];
fft_band = [6 35];

    stimDef_ref = {'S102';'REF'};
    mrk_stim_ref= mrk_defineClasses(mrk, stimDef_ref);
    ival_ref = [3000 5000];
    epo_ref = cntToEpo(cnt, mrk_stim_ref, ival_ref);
    epo_spec_ref = proc_spectrum(epo_ref, fft_band, kaiser(cnt.fs,2));
    
epo = cntToEpo(cnt, mrk_stim, ival);
epo_lap = proc_laplace(epo);
epo_spec= proc_spectrum(epo, fft_band, kaiser(cnt.fs,2));
epo_lap_spec= proc_spectrum(epo_lap, fft_band, kaiser(cnt.fs,2));

epo_lap_spec_r= proc_r_square_signed(proc_selectClasses(epo_lap_spec, ['Mod.freq',int2str(modfreq)],['Mod.freq',int2str(modfreq1)]));
epo_spec_r= proc_r_square_signed(proc_selectClasses(epo_spec, ['Mod.freq',int2str(modfreq)],['Mod.freq',int2str(modfreq1)]));

channel_list ={'FC3-CP3','FCz-CPz','FC4-CP4'};
epo_bi= proc_bipolarChannels(epo, channel_list);
epo_bi_spec= proc_spectrum(epo_bi, fft_band, kaiser(cnt.fs,2));
epo_spec_bi_r= proc_r_square_signed(proc_selectClasses(epo_bi_spec, ['Mod.freq',int2str(modfreq)],['Mod.freq',int2str(modfreq1)]));

mnt= getElectrodePositions(cnt.clab);
mnt= mnt_setGrid(mnt, 'tiny');
mnt= mnt_shrinkNonEEGchans(mnt);

mnt1= getElectrodePositions(cnt.clab);
mnt1= mnt_setGrid(mnt1, 'medium');
mnt1= mnt_shrinkNonEEGchans(mnt1);

mnt_bi= getElectrodePositions(epo_bi.clab);
mnt_bi= mnt_restrictMontage(mnt_bi, epo_bi.clab);
mnt_bi= mnt_setGrid(mnt_bi, 'tiny');

opt_spec= defopt_spec;
opt_grid= defopt_erps;

warning('off','all')

figure('Name','Epochs - Timeseries'); grid_plot(epo, mnt,opt_grid)

figure('Name','Epochs_raw - FFT');H=grid_plot(epo_spec, mnt,opt_spec);
grid_markIval([modfreq-1 modfreq+1]);
grid_markIval([modfreq1-1 modfreq1+1]);
grid_addBars(epo_spec_r, 'h_scale',H.scale, 'colormap',cmap_posneg(21),'cLim','sym');


figure('Name','Epochs_lap - FFT');H=grid_plot(epo_lap_spec, mnt,opt_spec);
grid_markIval([modfreq-1 modfreq+1]);
grid_markIval([modfreq1-1 modfreq1+1]);
grid_addBars(epo_lap_spec_r, 'h_scale',H.scale, 'colormap',cmap_posneg(21),'cLim','sym');
 

figure('Name','Epochs_raw - FFT (Full Montage)');H=grid_plot(epo_spec, mnt1,opt_spec);
grid_markIval([modfreq-1 modfreq+1]);
grid_markIval([modfreq1-1 modfreq1+1]);

figure('Name','Epochs_lap - FFT (Full Montage)');H=grid_plot(epo_lap_spec, mnt1,opt_spec);
grid_markIval([modfreq-1 modfreq+1]);
grid_markIval([modfreq1-1 modfreq1+1]);

figure('Name','Epochs_bipol - FFT');H=grid_plot(epo_bi_spec, mnt_bi,opt_spec);
grid_markIval([modfreq-1 modfreq+1]);
grid_markIval([modfreq1-1 modfreq1+1]);
grid_addBars(epo_spec_bi_r, 'h_scale',H.scale, 'colormap',cmap_posneg(21),'cLim','sym');
 
figure('Name','REF - FFT');H=grid_plot(epo_spec_ref, mnt,opt_spec);



% figure;plotChannel(epo_spec,'CCP6')
% figure;plotChannel(epo_lap_spec,'C4 lap' )

% %%
% plot_classes = Para.modfreq;
% 
% plot_chan_raw ='CP3';
% ind_raw = chanind(epo,plot_chan_raw);
% 
% plot_chan_lap ='C3 lap';
% ind_lap = chanind(epo_lap,plot_chan_lap);
%     
% for i = 1:length(plot_classes),
%     
%     epo_band_ref = proc_Bandpower(epo_ref,[plot_classes(i)-1 plot_classes(i)+1]);
%     epo_avg_band_ref = proc_average(epo_band_ref);
%     epo_band_chan_ref = epo_avg_band_ref.x(:,ind_raw);
%     
%     stimDef1 = {['S ',int2str(plot_classes(i))];['Mod-freq ',int2str(plot_classes(i))]};
%     mrk_stim= mrk_defineClasses(mrk, stimDef1);
%     
%     epo_raw = cntToEpo(cnt, mrk_stim, ival);
%     epo_band_raw = proc_Bandpower(epo_raw,[plot_classes(i)-1 plot_classes(i)+1]);
%     epo_avg_band_raw = proc_average(epo_band_raw);
%     epo_band_chan_raw(i) = ((epo_avg_band_raw.x(:,ind_raw)-epo_band_chan_ref)/epo_band_chan_ref)*100 ;
% 
%     epo_lap = proc_laplace(epo_raw);
%     epo_band_lap = proc_Bandpower(epo_lap,[plot_classes(i)-1 plot_classes(i)+1]);
%     epo_avg_band_lap = proc_average(epo_band_lap);
%     epo_band_chan_lap(i) = epo_avg_band_lap.x(:,ind_lap);
% end
% 
% figure('Name',['Tuningfunction_rawdata',plot_chan_raw])
% plot(plot_classes,epo_band_chan_raw,'--rs','LineWidth',2,'MarkerEdgeColor','k','MarkerFaceColor','g','MarkerSize',8)
% grid on
% xlabel('Modulation Frequency [Hz]')
% ylabel('Band Power')
% 
% figure('Name',['Tuningfunction_lapdata',plot_chan_lap])
% plot(plot_classes,epo_band_chan_lap,'--rs','LineWidth',2,'MarkerEdgeColor','k','MarkerFaceColor','g','MarkerSize',8)
% grid on
% xlabel('Modulation Frequency [Hz]')
% ylabel('Band Power')