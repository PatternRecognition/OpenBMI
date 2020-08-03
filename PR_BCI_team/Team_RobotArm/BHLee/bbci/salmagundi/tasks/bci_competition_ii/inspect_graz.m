file= 'bci_competition_ii/graz_train';
%fig_dir=  %% relavtive to global variable EEG_FIG_DIR !

[epo, mrk, mnt]= loadProcessedEEG(file);

grid_opt= struct('yDir','normal');
grid_opt.colorOrder= [1 0 0; 0 0.7 0];
grid_opt.xTick= 0:1000:9000;
grid_opt.scaleGroup= {{'not', 'E*'}};

%% no ERPs due to high-pass filtering during recording
%showERPgrid(epo, mnt);
%grid_plot(epo, mnt, grid_opt);

mnt_spec= mnt;
mnt_spec.box_sz= 0.9*mnt_spec.box_sz;

grid_spec_opt= grid_opt;
grid_spec_opt.yUnit= 'power';
grid_spec_opt.xTickMode= 'auto';
spec= proc_spectrum(epo, [1 40], epo.fs);
grid_plot(spec, mnt_spec, grid_spec_opt);
%saveFigure([fig_dir 'graz_spec'], [10 5]*2);
pause

epot= proc_t_scale(spec, 0.01);
tsc_opt= grid_spec_opt;
tsc_opt.yUnit= '';
tsc_opt.titleAppendix= sprintf('  (\\alpha=%.2f)', epot.alpha);
tsc_opt.colorOrder= [1 0 0];
tsc_opt.scalePolicy= 'sym';
grid_plot(epot, mnt_spec, tsc_opt);
grid_markRange(epot.crit, [], 'color',0.4*[1 1 1]);
%saveFigure([fig_dir 'graz_spectrum_tScaled'], [10 5]*2);
pause

band= [7 14];
refIval= [0 2000];
[b,a]= getButterFixedOrder(band, epo.fs, 6);
erd= proc_filtfilt(epo, b, a);
erd.title= sprintf('%s [%d %d] Hz', epo.title, band);
erd= proc_squareChannels(erd);
erd= proc_average(erd);
erd= proc_calcERD(erd, refIval, 100);
grid_plot(erd, mnt, grid_opt);
%saveFigure([fig_dir 'graz_alpha_erd'], [10 5]*2);
pause

band= [15 26];
refIval= [0 2000];
[b,a]= getButterFixedOrder(band, epo.fs, 6);
erd= proc_filtfilt(epo, b, a);
erd.title= sprintf('%s [%d %d] Hz', epo.title, band);
erd= proc_squareChannels(erd);
erd= proc_classMean(erd, 1:2);
erd= proc_calcERD(erd, refIval, 100);
grid_plot(erd, mnt, grid_opt);
%saveFigure([fig_dir 'graz_beta_erd'], [10 5]*2);
pause

band= [26 40];
refIval= [0 2000];
[b,a]= getButterFixedOrder(band, epo.fs, 6);
erd= proc_filtfilt(epo, b, a);
erd.title= sprintf('%s [%d %d] Hz', epo.title, band);
erd= proc_squareChannels(erd);
erd= proc_classMean(erd, 1:2);
erd= proc_calcERD(erd, refIval, 100);
grid_plot(erd, mnt, grid_opt);
%saveFigure([fig_dir 'graz_gamma_erd'], [10 5]*2);
