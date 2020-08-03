grd= sprintf('EOGh,scale,Fz,legend,EOGv\nC3,C1,Cz,C2,C4\nP3,P1,Pz,P2,P4\nEMGl,O1,EMGf,O2,EMGr');

mnt= getElectrodePositions(Cnt.clab);
mnt= mnt_setGrid(mnt, grd);
mnt= mnt_excenterNonEEGchans(mnt, 'E*');

fig_opt= {'numberTitle','off', 'menuBar','none'};

scalp_opt= struct('shading','flat', 'resolution',30, 'contour',[-40:2:40]);
head= setDisplayMontage(mnt, 'visible_128');
head= mnt_adaptMontage(head, Cnt.clab);
colDef= {'Target','Non-target';
         [0 0.7 0], [1 0 0]};
grid_opt = defopt_erps;
grid_opt.colorOrder = choose_colors(mrk.className,colDef);
grid_opt.scaleGroup= {scalpChannels, {'EMG*'}, {'EOG*'}};
scalp_opt.colorOrder = grid_opt.colorOrder;

handlefigures('use','ARTIFACT');
[mrk_rej, rclab, rtrials] = reject_varEventsAndChannels(Cnt, mrk, [0 800], 'visualize', 1, 'do_multipass', 1, 'do_bandpass', 0);

fig_H = handlefigures('use','ERP');
set(fig_H, fig_opt{:}, 'name',sprintf('%s: ERPs in [%d %d] ms', epo.title, opt.dar_ival));  
h = grid_plot(epo, mnt, grid_opt);  
hh = cmap_posneg(81);
colormap(hh)
grid_markIval(opt.selectival);
grid_addBars(epo_r,...
             'colormap',hh,'height',0.12,'h_scale',h.scale);
 
fig_H = handlefigures('use','ERPscalps');
set(fig_H, fig_opt{:}, ...
         'name',sprintf('%s: ERP-Pattern', epo.title));
scalpEvolutionPlusChannel(epo, head, 'Cz', opt.selectival, scalp_opt);
grid_addBars(epo_r, 'colormap',hh,'height',0.12);

fig_H = handlefigures('use','ERP r-value scalps');  
scalpEvolution(epo_r, head, opt.selectival, scalp_opt, 'colormap', cmap_posneg(81));

