file= 'VPiz_08_06_12/sssep_BCI_27_new_lev6VPiz';
%file= 'VPip_08_06_19/sssep_BCI_27_new_lev8_difVPip';

[cnt,mrk,mnt]= eegfile_loadMatlab(file);

modfreq =27;

ival = [1750 4200];
[mrk_clean]= reject_varEventsAndChannels(cnt, mrk, ival, 'visualize',1);

ival_ref = [0 1000];
mrk_ref= mrk;
mrk_ref.y= sum(mrk.y);
mrk_ref.className= {'ref'};
[mrk_clean_ref]= reject_varEventsAndChannels(cnt, mrk_ref, ival_ref, 'visualize',1);

epo_ref= cntToEpo(cnt, mrk_clean_ref, ival_ref);
epo_ref_lap= proc_laplace(epo_ref, 'small',' lap', 'filter all');

epo= cntToEpo(cnt, mrk_clean, ival);
epo_lap= proc_laplace(epo, 'small',' lap', 'filter all');

spec= proc_spectrum(epo_lap, [5 40], kaiser(cnt.fs,2));
spec_ref= proc_spectrum(epo_ref_lap, [5 40], kaiser(cnt.fs,2));
spec_r= proc_r_square_signed(spec);
spec_ref= proc_average(spec_ref);
spec_baseline= proc_subtractReferenceClass(spec, spec_ref);

%H= grid_plot(spec_baseline, mnt, defopt_spec);
H= grid_plot(spec, mnt, defopt_spec);
grid_addBars(spec_r, 'h_scale',H.scale, ...
             'colormap',cmap_posneg(21), ...
             'cLim', 'sym', ...
             'box','on');


scalpEvolution(spec_r, mnt, 21:2:33);



return


%VPip:
[N,Ws]= cheb2ord([20 22]/100*2, [19 23]/100*2, 1, 40);
[b,a]= cheby2(N, 40, Ws);
cnt_flt= proc_filt(cnt, b, a);

erd= cntToEpo(cnt_flt, mrk, [0 5000]);
erd= proc_laplace(erd, 'small',' lap', 'filter all');
erd= proc_envelope(erd, 'ma_msec', 200);
erd= proc_baseline(erd, [], 'trialwise', 0);
erd_r= proc_r_square_signed(erd);

H= grid_plot(erd, mnt, defopt_erps);
grid_addBars(erd_r, 'h_scale',H.scale, ...
             'colormap',cmap_posneg(21), ...
             'cLim', 'sym', ...
             'box','on');


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
