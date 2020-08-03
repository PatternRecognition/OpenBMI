file= 'bci_competition_ii/tuebingen1_train';
%fig_dir=  %% relavtive to global variable EEG_FIG_DIR !

[Epo, mrk, mnt]= loadProcessedEEG(file);

grid_opt= struct('yDir','reverse');
grid_opt.colorOrder= [0 0 1; 0.9 0.6 0];
grid_opt.xTick= 0:1000:8000;
grid_opt.scaleGroup= {{'not', 'E*'}};

epo= Epo;
%epo= proc_baseline(epo, [2000 2300]);
grid_plot(epo, mnt, grid_opt);
%saveFigure([fig_dir 'tuebingen1_erp'], [10 5]*2);

epot= proc_t_scale(epo, 0.01);
tsc_opt= grid_opt;
tsc_opt.yUnit= '';
tsc_opt.titleAppendix= sprintf('  (\\alpha=%.2f)', epot.alpha);
tsc_opt.colorOrder= [1 0 0];
tsc_opt.scalePolicy= 'sym';
grid_plot(epot, mnt, tsc_opt);
grid_markRange(epot.crit, [], 'color',0.4*[1 1 1]);
%saveFigure([fig_dir 'tuebingen1_tScaled'], [10 5]*2);

mnt_spec= mnt;
mnt_spec.box_sz= 0.9*mnt_spec.box_sz;
grid_spec_opt= struct('yUnit','power', 'xTickMode','auto');

spec= proc_spectrum(epo, [1 40], epo.fs);
grid_plot(spec, mnt_spec, grid_spec_opt);
