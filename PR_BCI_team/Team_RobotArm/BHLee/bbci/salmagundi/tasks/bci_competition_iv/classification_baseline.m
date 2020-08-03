fig_dir= '/home/blanker/neuro_cvs/Tex/bci/bci_competition_iv/berlin/pics/';

dd= [BCI_DIR 'salmagundi/tasks/bci_competition_iv/'];
subdir_list= textread([dd 'session_list'], '%s');

grd= sprintf(['scale,F3,Fz,F4,legend\n' ...
              'C3,C1,Cz,C2,C4\n' ...
              'CP3,CP1,CPz,CP2,CP4\n' ...
              'P5,P3,Pz,P4,P6']);
opt_grid= defopt_erps('scale_leftshift',0.075);
opt_grid_spec= defopt_spec('scale_leftshift',0.075, ...
                           'xTickAxes','Pz');
opt_scalp_bp= defopt_scalp_power('resolution',30);
opt_scalp_erd= defopt_scalp_erp('resolution',30);
opt_scalp_r= defopt_scalp_r('resolution', 30);
colDef= {'left','right','foot','rest'; ...
         [1 0 0], [0 0.7 0], [0 0 1], [0.8 0 0.8]};


for vp= 1:length(subdir_list),

subdir= [subdir_list{vp} '/']
is= min(find(subdir=='_'));
sbj= subdir(1:is-1);
sd= subdir;
sd(find(ismember(sd,'_/')))= [];
fig_pre= [fig_dir sd '_'];

%% default settings
clab= 'C4';
clab_rm= {};
band_list= [7 10; 10 14; 15 26; 28 35];
band_erd= [];
ival_spec= [750 4500];
ival_erd= [-500 6000];
ival_list= 500:1000:4500;
reject_opts= {};

%% subject-specific settings
switch(sbj),
 case 'VPja',
  ival_spec= [1250 4500];
  band_list= [7 10; 10 13; 13 20; 20 26];
  ival_list= [500 1000; 1000 2000; 2000 4000; 5000 5750];
 case 'VPip',
  clab= 'C3';
  ival_spec= [1250 4500];
  band_list= [7 10; 10 13; 23 28; 28 35];
  ival_list= [500 1000; 1000 2000; 2000 4500; 5000 5750];
 case 'VPik',
  band_erd= [8 12];
  band_list= [8 12; 12 15; 18 22; 27 33];
  ival_list= [500 1000; 1000 2000; 2000 4500; 5000 5750];
 case 'VPzk',  
  clab= 'FC4';
  band_erd= [8 12];
  band_list= [7 12; 12 17; 18 23; 26 32];
  ival_list= [500 750; 1000 2000; 2000 4500; 5000 5750];
end
if isempty(band_erd),
  band_erd= band_list(2,:);
end

file= [subdir 'imag_arrow' sbj];
[cnt, mrk, mnt]= eegfile_loadMatlab(file, 'clab',{'not',clab_rm{:}});
mrk= mrk_removeVoidClasses(mrk);

figure(1);
ct= proc_selectChannels(cnt, scalpChannels); %% order nicely
[mrk, rClab]= reject_varEventsAndChannels(ct, mrk, ival_spec, ...
                                          'visualize', 1, ...
                                          reject_opts{:});
clear ct
printFigure([fig_pre 'artifact_rejection'], [19 12]);

mnt= setElectrodeMontage(cnt.clab);
mnt= mnt_setGrid(mnt, grd);
mnt_red= mnt_restrictMontage(mnt, 'not', rClab);
classtags= apply_cellwise(mrk.className, inline('upper(x(1))','x'));
opt_grid.colorOrder= choose_colors(mrk.className, colDef);
opt_grid.lineWidthOrder= [1 1 1];
opt_grid_spec.colorOrder= opt_grid.colorOrder;
opt_grid_spec.lineWidthOrder= opt_grid.lineWidthOrder;

spec= makeEpochs(cnt, mrk, ival_spec);
disp_clab= getClabOfGrid(mnt);
requ_clab= getClabForLaplace(strukt('clab',scalpChannels), disp_clab);
spec_lap= proc_selectChannels(spec, requ_clab);
spec_lap= proc_laplace(spec_lap, 'small', ' lap', 'filter all');
spec_lap= proc_spectrum(spec_lap, [5 40], kaiser(cnt.fs,2));
spec= proc_spectrum(spec, [5 40], kaiser(cnt.fs,2));
spec_r= proc_r_square_signed(spec);
spec_lap_r= proc_r_square_signed(spec_lap);

mrk_ref= mrk;
mrk_ref.y= ones(1, length(mrk_ref.pos));
mrk_ref.className= {'ref'};
spec_baseline= makeEpochs(cnt, mrk_ref, [-2000 0]);
spec_baseline= proc_spectrum(spec_baseline, [5 40], kaiser(cnt.fs,2));
spec_baseline= proc_average(spec_baseline);
spec_ref= proc_subtractReferenceClass(spec, spec_baseline);

H= grid_plot(spec_baseline, mnt, opt_grid_spec, 'colorOrder',0.3*[1 1 1]);
grid_markIval(band_erd);
printFigure([fig_pre 'spec_baseline'], [19 12]);

H= scalpEvolutionPlusChannel(spec_baseline, mnt_red, clab, band_list, ...
                             opt_scalp_bp, ...
                             'colorOrder',0.3*[1 1 1], ...
                             'scalePos','horiz', ...
                             'globalCLim',0, ...
                             'legend_pos',1);
printFigure([fig_pre 'spec_baseline_topo'], [20 4+6]);

H= grid_plot(spec, mnt, opt_grid_spec);
grid_markIval(band_erd);
grid_addBars(proc_rectifyChannels(spec_r), 'h_scale',H.scale, ...
             'colormap',cmap_posneg(21), ...
             'cLim', 'sym', ...
             'box','on');
printFigure([fig_pre 'spec'], [19 12]);

H= grid_plot(spec_lap, mnt, opt_grid_spec);
grid_markIval(band_erd);
grid_addBars(proc_rectifyChannels(spec_lap_r), 'h_scale',H.scale, ...
             'colormap',cmap_posneg(21), ...
             'cLim', 'sym', ...
             'box','on');
printFigure([fig_pre 'spec_lap'], [19 12]);

H= scalpEvolutionPlusChannel(spec_ref, mnt_red, clab, band_list, ...
                             opt_scalp_bp, ...
                             'colorOrder',opt_grid_spec.colorOrder, ...
                             'lineWidthOrder',opt_grid.lineWidthOrder*3, ...
                             'scalePos','horiz', ...
                             'globalCLim',0, ...
                             'colormap', cmap_posneg(51), ...
                             'legend_pos',1);
grid_addBars(proc_rectifyChannels(spec_r), ...
             'box','on', ...
             'cLim', '0tomax', ...
             'vpos',1);
set(H.text(clidx), 'FontWeight','normal');
printFigure([fig_pre 'spec_topo'], [20 4+5.8*size(spec.y,1)]);

figure(2);
spec_r= proc_r_square_signed(spec);
spec_r.className= {sprintf('\\pm r^2 (%s,%s)', classtags{:})};
scalpEvolution(spec_r, mnt, band_list, opt_scalp_r);
shiftAxesUp; drawnow;
printFigure([fig_pre 'spec_topo_r'], [20 6]);
clear spec spec_r spec_lap spec_lap_r

[b,a]= butter(5, band_erd/cnt.fs*2);
cnt= proc_channelwise(cnt, 'filt', b, a);
epo= makeEpochs(cnt, mrk, ival_erd);
erd_lap= proc_selectChannels(epo, requ_clab);
erd_lap= proc_laplace(erd_lap, 'small', ' lap', 'filter all');
erd_lap= proc_envelope(erd_lap, 'ma_msec', 200);
erd_lap= proc_baseline(erd_lap, [], 'trialwise', 0);
%erd_lap= proc_baseline(erd_lap, [ival_erd(1) 0]);
erd= proc_envelope(epo, 'ma_msec', 200);
erd= proc_baseline(erd, [], 'trialwise', 0);
erd_lap_r= proc_r_square_signed(erd_lap);
erd_r= proc_r_square_signed(erd);

erd_baseline= makeEpochs(cnt, mrk_ref, [-4500 500]);
erd_baseline= proc_laplace(erd_baseline);
erd_baseline= proc_envelope(erd_baseline, 'ma_msec', 200);
erd_baseline= proc_baseline(erd_baseline, [], 'trialwise', 0);

figure(1);
H= grid_plot(erd, mnt, opt_grid);
grid_markIval(ival_spec);
grid_addBars(proc_rectifyChannels(erd_r), 'h_scale',H.scale, ...
             'colormap',cmap_posneg(21), ...
             'cLim', 'sym', ...
             'box','on');
printFigure([fig_pre 'erd'], [19 12]);

H= grid_plot(erd_lap, mnt, opt_grid);
yLim= get(H.ax(1),'YLim');
grid_markIval(ival_spec);
grid_addBars(proc_rectifyChannels(erd_lap_r), 'h_scale',H.scale, ...
             'colormap',cmap_posneg(21), ...
             'cLim', 'sym', ...
             'box','on');
printFigure([fig_pre 'erd_lap'], [19 12]);

grid_plot(erd_baseline, mnt, opt_grid, 'colorOrder',0.3*[1 1 1], ...
          'scalePolicy', yLim);
grid_markIval(ival_spec);
printFigure([fig_pre 'erd_baseline'], [19 12]);

H= scalpEvolutionPlusChannel(erd, mnt_red, clab, ival_list, opt_scalp_erd, ...
                             'colorOrder',opt_grid.colorOrder, ...
                             'lineWidthOrder',opt_grid.lineWidthOrder*3, ...
                             'legend_pos',2);
grid_addBars(proc_rectifyChannels(erd_r), ...
             'box','on', ...
             'cLim', '0tomax', ...
             'vpos',1);
set(H.text(clidx), 'FontWeight','normal');
printFigure([fig_pre 'erd_topo'], [20 4+5.8*size(erd.y,1)]);

figure(2);
erd_r= proc_r_square_signed(erd);
erd_r.className= {sprintf('\\pm r^2 (%s,%s)', classtags{:})};
scalpEvolution(erd_r, mnt, ival_list, opt_scalp_r);
shiftAxesUp; drawnow;
printFigure([fig_pre 'erd_topo_r'], [20 6]);
clear erd erd_r erd_lap erd_lap_r

end
