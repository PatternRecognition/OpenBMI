grd= sprintf('EOGh,scale,Fz,legend,EOGv\nC3,C1,Cz,C2,C4\nCP3,CP1,CPz,CP2,CP4\nEMGl,O1,EMGf,O2,EMGr');

if isempty(chanind(Cnt,'EOGh')),
  eog_bip= proc_bipolarChannels(Cnt,'EOGhp-EOGhn','EOGvp-EOGvn');
  eog_bip.clab= {'EOGh','EOGv'};
  Cnt= proc_selectChannels(Cnt,'not','EOG*');
  Cnt= proc_appendChannels(Cnt,eog_bip);
end

mnt= getElectrodePositions(Cnt.clab);
mnt= mnt_setGrid(mnt, grd);
mnt= mnt_excenterNonEEGchans(mnt, 'E*');

fig_opt= {'numberTitle','off', 'menuBar','none'};


scalp_opt= struct('shading','flat', 'resolution',30, 'contour',[-40:2:40]);
head= setDisplayMontage(mnt, 'visible_128');
head= mnt_adaptMontage(head, Cnt.clab);
colDef= {'hit','miss';
         [0 0.7 0], [1 0 0]};
grid_opt= struct('colorOrder', choose_colors(colDef,mrk.className));
grid_opt.scaleGroup= {scalpChannels, {'EMG*'}, {'EOG*'}};


if bbci.withgraphics
  handlefigures('use','ERP');
  set(gcf, fig_opt{:}, 'name',sprintf('%s: ERPs in [%d %d] ms', Cnt.title, opt.dar_ival));
  
  epo= makeEpochs(Cnt, mrk, opt.dar_ival);
  epo= proc_baseline(epo, opt.dar_base);
  epo_r= proc_r_square_signed(proc_selectClasses(epo,'miss','hit'));
  h = grid_plot(epo, mnt, grid_opt);
  
  hh = hot(64); hh = hh(end:-1:1,:);
  hh(1,:) = [0.999999,0.999999,0.999999];
  colormap(hh)
  grid_markIval(opt.ival);
  grid_addBars(proc_rectifyChannels(epo_r),...
               'colormap',hh,'height',0.12,'h_scale',h.scale);
 
  handlefigures('use','ERPscalps');
  set(gcf, fig_opt{:}, ...
           'name',sprintf('%s: ERP-Pattern', epo.title));
  scalpEvolutionPlusChannel(epo, head, 'FCz', opt.dar_scalps, scalp_opt);
  grid_addBars(proc_rectifyChannels(epo_r));

  handlefigures('use','ERP r-value scalps');  
  scalpEvolution(epo_r, head, opt.dar_scalps, scalp_opt);
end

cnt = proc_selectChannels(Cnt, opt.clab{:});
epo = makeEpochs(cnt,mrk,opt.ival);
if isfield(opt,'baseline') & ~isempty(opt.baseline)
  epo = proc_baseline(epo,opt.baseline);
  epo = proc_selectIval(epo,opt.selectival);
end

features= proc_jumpingMeans(epo, opt.jMeans);

if bbci.withclassification
  opt_xv= strukt('xTrials',[5 5], 'loss','rocArea');
  [dum,dum,outTe] = xvalidation(features, opt.model, opt_xv);
  me= val_confusionMatrix(features, outTe, 'mode','normalized');
  remainmessage = sprintf('Correct Hits: %2.1f, Correct Miss: %2.1f\n',100*me(1,1),100*me(2,2));
  bbci_bet_message(remainmessage,0);
end

% What do we need later?
analyze = struct('features', features, 'message', remainmessage);

bbci_bet_message('Finished analysis\n');
